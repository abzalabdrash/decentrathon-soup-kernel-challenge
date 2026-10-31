"""Task G1 measured on the T4 before any integration: one CUDA graph template of the fused MLP (the
training path of fused_proj._MLP: NF4 dequantization into the shared buffer, gate|up GEMMs, offload_int8's
SwiGLU + int8 form, down GEMM; backward int8_bwd.mlp_backward reading the int8 form) captured once on
static buffers and replayed for every layer, against the eager path. Measurement only, nothing here is
used by apply().

Per layer, at real token counts N and their bucket B (N <= 512 -> 512, else N rounded up to 128):
- CPU issue time (perf_counter around 16 layer calls between synchronizations, the launch queue never
  fills): eager core; replay + what stays eager around it (input copy, padding rows, per-layer constants,
  output copy; for the backward also the int8 (gate|up) copied into the static buffer);
- GPU time (the same calls queued behind a long sleep kernel, so the CPU never starves the GPU);
- capture time, memory of the static buffers and of the graph's private pool, and the same for the
  full autograd node (`mlp(x)` forward + backward in a 16-layer chain under the offload context, as in
  training) to see which share of a layer's CPU the template can remove;
- the traps: a timing event recorded during capture (what the harness's TritonWatch does around every
  Triton launch of solution/ while it times steps 3+), an external event recorded in the middle of a
  graph and waited for by a side stream after the replay (could the weight ring's release stay inside
  the template?), an external wait inside the graph on an event recorded before the replay.

`python -m solution.bench_mlp_graph [--n 296 416 512 640 832 1440] [--layers 16]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_mlp_graph` runs it on Kaggle)."""

import argparse
import statistics
import time

import torch

HIDDEN, INTER = 1536, 8960
SLEEP_CYCLES = int(2e8)  # ~0.13 s on the T4: covers the issue of 16 eager layers


def _bucket(n):
    return 512 if n <= 512 else -(-n // 128) * 128


def _layers(count):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution.test_fused_proj import _config, _quantize_mlp

    config = _config(HIDDEN, INTER)
    return [_quantize_mlp(mq.Qwen2MLP(config), seed=20 + 3 * i).cuda().train() for i in range(count)]


def _patch():
    from solution import fused_proj
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm

    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    fused_proj.patch_qwen2_mlp_fused()
    fused_proj.set_mlp_recompute(False)


def cpu_issue(fn, layers, reps=7):
    """Median CPU ms per call of fn(i) over `layers` calls between synchronizations."""
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(layers):
            fn(i)
        times.append((time.perf_counter() - t0) * 1e3 / layers)
        torch.cuda.synchronize()
    return statistics.median(times)


def gpu_time(fn, layers, reps=7):
    """Median GPU ms per call of fn(i), the calls queued behind a sleep kernel (GPU-bound timing)."""
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        torch.cuda._sleep(SLEEP_CYCLES)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(layers):
            fn(i)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / layers)
    return statistics.median(times)


class _Forward:
    """The forward template at bucket B: static x16 (B, D), (gate|up) (2I, B), hidden (I, B), output
    (B, D); the int8 scales are made inside the capture (graph pool). The dequantizations read the
    weights of `plan` (the type ring's fixed slots in training; here one layer's own weights)."""

    def __init__(self, plan, bucket, stream, pool=None):
        from solution import fused_proj as fp

        self.plan, self.bucket, self.stream = plan, bucket, stream
        dev, half = "cuda", plan.half
        self.x16 = torch.zeros((bucket, HIDDEN), dtype=half, device=dev)
        self.gate_up = torch.empty((2 * INTER, bucket), dtype=half, device=dev)
        self.hidden = torch.empty((INTER, bucket), dtype=half, device=dev)
        self.out = torch.empty((bucket, HIDDEN), dtype=half, device=dev)
        self.consts = torch.zeros(3 * 273, dtype=torch.float32, device=dev)  # offsets + code tables
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):  # cuBLAS workspace of the capture stream, Triton loads
            self.body()
        torch.cuda.synchronize()
        fp._HELD["key"] = fp._HELD["views"] = None  # every dequantization goes into the graph
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            self.graph.capture_begin(pool=pool) if pool is not None else self.graph.capture_begin()
            self.wire = self.body()
            self.graph.capture_end()
        self.capture_ms = (time.perf_counter() - t0) * 1e3
        self.pool_bytes = torch.cuda.memory_allocated() - before
        fp._HELD["key"] = fp._HELD["views"] = None

    def body(self):
        from solution import fused_proj as fp
        from solution.offload_int8 import swiglu_quant

        plan = self.plan
        fp._gate_up_matmul(plan, None, self.x16.t(), self.gate_up)
        wire = swiglu_quant(self.gate_up, self.hidden)
        w = fp._weights((plan.down,), plan.half, self.x16.device)
        torch.mm(self.hidden.t(), fp._t(w), out=self.out)
        return wire

    def static_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.x16, self.gate_up, self.hidden,
                                                           self.out, self.consts))

    def __call__(self, x, layer_consts):
        n = x.shape[0]
        self.x16[:n].copy_(x)
        if n < self.bucket:
            self.x16[n:].zero_()
        self.consts.copy_(layer_consts)
        self.graph.replay()
        return self.out[:n].clone()


class _Backward:
    """The int8 backward template at bucket B: static incoming gradient (B, D) half, the int8 form of
    (gate|up) for B tokens (uint8, offload_int8.wire_bytes), per-layer column scales; dL/dx (B, D)
    fp32 made inside the capture (graph pool)."""

    def __init__(self, plan, bucket, stream, buffer, pool=None):
        from solution import offload_int8

        self.plan, self.bucket, self.buffer = plan, bucket, buffer
        dev, half = "cuda", plan.half
        self.grad16 = torch.zeros((bucket, HIDDEN), dtype=half, device=dev)
        self.wire = torch.zeros(offload_int8.wire_bytes(2 * INTER, bucket), dtype=torch.uint8, device=dev)
        self.consts = torch.zeros(2 * INTER + 2 * HIDDEN, dtype=torch.float32, device=dev)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            self.body()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            self.graph.capture_begin(pool=pool) if pool is not None else self.graph.capture_begin()
            self.grad_x = self.body()
            self.graph.capture_end()
        self.capture_ms = (time.perf_counter() - t0) * 1e3
        self.pool_bytes = torch.cuda.memory_allocated() - before

    def body(self):
        from solution import int8_bwd

        return int8_bwd.mlp_backward(self.plan, self.wire, self.grad16, self.buffer)

    def static_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.grad16, self.wire, self.consts))

    def __call__(self, grad, wire, layer_consts):
        n = grad.shape[0]
        self.grad16[:n].copy_(grad)
        if n < self.bucket:
            self.grad16[n:].zero_()
        self.wire.copy_(wire)
        self.consts.copy_(layer_consts)
        self.graph.replay()
        return self.grad_x[:n].to(torch.float16)


def _eager_forward(plan, x):
    """fused_proj._MLP.forward's ops on the fp16 GEMM path with offload_int8's SwiGLU (P1), no autograd."""
    from solution import fused_proj as fp
    from solution.offload_int8 import swiglu_quant

    n = x.shape[0]
    padded = -(-n // 8) * 8
    x16 = fp._padded_half(x, n, padded, plan.half)
    gate_up = torch.empty((2 * INTER, padded), dtype=plan.half, device=x.device)
    fp._gate_up_matmul(plan, None, x16.t(), gate_up)
    hidden = torch.empty((INTER, padded), dtype=plan.half, device=x.device)
    wire = swiglu_quant(gate_up, hidden)
    w = fp._ring_weights(plan, (plan.down,), x.device, 1)
    out = torch.mm(hidden.t(), fp._t(w))
    return out[:n], wire


def _eager_backward(plan, grad, wire, buffer):
    from solution import fused_proj as fp
    from solution import int8_bwd

    n = grad.shape[0]
    padded = -(-n // 8) * 8
    grad16 = fp._padded_half(grad, n, padded, plan.half)
    return int8_bwd.mlp_backward(plan, wire, grad16, buffer)[:n].to(torch.float16)


def _wire(gate_up):
    from solution import offload_int8

    wire, _ = offload_int8.encode(gate_up)
    return wire


def templates(ns, layers, mlps, plans, bwd=True):
    from solution import fused_proj as fp

    stream = torch.cuda.Stream()
    k = len(mlps)
    buffer = fp._buffer(torch.device("cuda"), torch.float16, INTER * HIDDEN)
    g = torch.Generator("cuda").manual_seed(0)
    print(f"\nper layer, {layers} layers per timing, median of 7; CPU = issue time on this box's CPU, "
          "GPU = queued behind a sleep kernel")
    print(f"{'N':>5s} {'B':>5s} | {'fwd CPU eager':>13s} {'replay':>7s} {'saved':>6s} | {'fwd GPU eager':>13s} "
          f"{'replay':>7s} | {'bwd CPU eager':>13s} {'replay':>7s} {'saved':>6s} | {'bwd GPU eager':>13s} "
          f"{'replay':>7s} | capture f/b ms | static+pool MiB f/b")
    for n in ns:
        bucket = _bucket(n)
        xs = [torch.randn((n, HIDDEN), device="cuda", generator=g).half() for _ in range(k)]
        grads = [torch.randn((n, HIDDEN), device="cuda", generator=g).half() * 0.01 for _ in range(k)]
        padded = -(-n // 8) * 8
        wires_n = [_wire(torch.randn((2 * INTER, padded), device="cuda", generator=g).half()) for _ in range(k)]
        wires_b = wires_n if padded == bucket else [
            _wire(torch.randn((2 * INTER, bucket), device="cuda", generator=g).half()) for _ in range(k)]
        consts_f = [torch.randn(3 * 273, device="cuda", generator=g) for _ in range(k)]
        consts_b = [torch.randn(2 * INTER + 2 * HIDDEN, device="cuda", generator=g) for _ in range(k)]
        fwd = _Forward(plans[0], bucket, stream)
        # correctness: one replay against the eager path on the same layer (bit for bit)
        ref, ref_wire = _eager_forward(plans[0], xs[0])
        got = fwd(xs[0], consts_f[0])
        same = torch.equal(ref, got)
        q_ref, s_ref = ref_wire
        q_got, s_got = fwd.wire
        same_wire = (torch.equal(q_ref[:, :n], q_got[:, :n]) and torch.equal(s_ref[:, :n], s_got[:, :n]))
        e_fc = cpu_issue(lambda i: _eager_forward(plans[i % k], xs[i % k]), layers)
        r_fc = cpu_issue(lambda i: fwd(xs[i % k], consts_f[i % k]), layers)
        e_fg = gpu_time(lambda i: _eager_forward(plans[i % k], xs[i % k]), layers)
        r_fg = gpu_time(lambda i: fwd(xs[i % k], consts_f[i % k]), layers)
        line = (f"{n:5d} {bucket:5d} | {e_fc:13.3f} {r_fc:7.3f} {e_fc - r_fc:6.3f} | {e_fg:13.3f} {r_fg:7.3f} | ")
        cap_b = mem_b = float("nan")
        if bwd:
            bw = _Backward(plans[0], bucket, stream, buffer)
            ref_g = _eager_backward(plans[0], grads[0], wires_n[0], buffer) if padded == bucket else None
            got_g = bw(grads[0], wires_b[0], consts_b[0])
            same_b = torch.equal(ref_g, got_g) if ref_g is not None else None
            e_bc = cpu_issue(lambda i: _eager_backward(plans[i % k], grads[i % k], wires_n[i % k], buffer), layers)
            r_bc = cpu_issue(lambda i: bw(grads[i % k], wires_b[i % k], consts_b[i % k]), layers)
            e_bg = gpu_time(lambda i: _eager_backward(plans[i % k], grads[i % k], wires_n[i % k], buffer), layers)
            r_bg = gpu_time(lambda i: bw(grads[i % k], wires_b[i % k], consts_b[i % k]), layers)
            line += f"{e_bc:13.3f} {r_bc:7.3f} {e_bc - r_bc:6.3f} | {e_bg:13.3f} {r_bg:7.3f} | "
            cap_b, mem_b = bw.capture_ms, (bw.static_bytes() + bw.pool_bytes) / 2**20
            del bw
        else:
            same_b = None
            line += f"{'':13s} {'':7s} {'':6s} | {'':13s} {'':7s} | "
        line += (f"{fwd.capture_ms:6.2f} / {cap_b:6.2f} | {(fwd.static_bytes() + fwd.pool_bytes) / 2**20:5.1f}"
                 f" / {mem_b:5.1f}   fwd bitwise {same} wire {same_wire} bwd bitwise {same_b}")
        print(line, flush=True)
        del fwd
        torch.cuda.synchronize()


def shared_pool(mlps, plans, bucket=512):
    """Forward and backward templates in one private pool (captured in order, replayed in order):
    what the pair keeps resident, and the allocator's numbers before and after."""
    from solution import fused_proj as fp

    torch.cuda.synchronize()
    a0, r0 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    stream = torch.cuda.Stream()
    buffer = fp._buffer(torch.device("cuda"), torch.float16, INTER * HIDDEN)
    pool = torch.cuda.graph_pool_handle()
    fwd = _Forward(plans[0], bucket, stream, pool=pool)
    bw = _Backward(plans[0], bucket, stream, buffer, pool=pool)
    torch.cuda.synchronize()
    a1, r1 = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    print(f"\nshared pool, bucket {bucket}: static fwd {fwd.static_bytes() / 2**20:.1f} MiB + bwd "
          f"{bw.static_bytes() / 2**20:.1f} MiB; pool during capture fwd {fwd.pool_bytes / 2**20:.1f} bwd "
          f"{bw.pool_bytes / 2**20:.1f} MiB; allocated +{(a1 - a0) / 2**20:.1f} MiB, reserved "
          f"+{(r1 - r0) / 2**20:.1f} MiB (incl. the capture stream's cuBLAS workspace)", flush=True)
    del fwd, bw
    torch.cuda.synchronize()


def autograd_chain(ns, mlps, layers):
    """The training node: h = h + mlp(h) over `layers` layers (fused_proj's patched forward, autocast
    fp16, offload_saved_tensors with the early int8 copy, int8 backward), CPU of the forward loop and
    of backward() per layer, and the GPU time of the same queued behind a sleep."""
    from solution import offload

    weights = [p for m in mlps for p in m.parameters()]
    k = len(mlps)
    print(f"\nautograd node, {layers} layers (h = h + mlp(h)), offload context with the early int8 copy, "
          "int8 backward; ms per layer")
    print(f"{'N':>5s} | {'fwd CPU':>8s} {'bwd CPU':>8s} | {'fwd+bwd GPU':>11s}")
    for n in ns:
        x0 = torch.randn((1, n, HIDDEN), device="cuda") * 0.5
        grad = torch.randn((1, n, HIDDEN), device="cuda") * 0.01

        def run(sleep):
            x = x0.detach().requires_grad_(True)
            with offload.offload_saved_tensors(weights=weights), torch.autocast("cuda", dtype=torch.float16):
                torch.cuda.synchronize()
                if sleep:
                    torch.cuda._sleep(SLEEP_CYCLES)
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                t0 = time.perf_counter()
                h = x
                for i in range(layers):
                    h = h + mlps[i % k](h)
                t1 = time.perf_counter()
                h.backward(grad)
                t2 = time.perf_counter()
                if sleep:
                    end.record()
                torch.cuda.synchronize()
            return (t1 - t0) * 1e3 / layers, (t2 - t1) * 1e3 / layers, (
                start.elapsed_time(end) / layers if sleep else None)

        for _ in range(2):
            run(False)
        cpu = [run(False) for _ in range(5)]
        gpu = [run(True)[2] for _ in range(3)]
        print(f"{n:5d} | {statistics.median(c[0] for c in cpu):8.3f} {statistics.median(c[1] for c in cpu):8.3f} | "
              f"{statistics.median(gpu):11.3f}", flush=True)


def traps():
    print("\ntraps")
    s = torch.cuda.Stream()
    x = torch.ones(1 << 20, device="cuda")
    with torch.cuda.stream(s):
        x.mul_(1)
    torch.cuda.synchronize()
    # 1. the harness's TritonWatch records a pair of timing events around each Triton launch of solution/
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g.capture_begin()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        x.mul_(1)
        b.record()
        g.capture_end()
    g.replay()
    torch.cuda.synchronize()
    try:
        print(f"  timing events recorded during capture: elapsed_time {a.elapsed_time(b):.3f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"  timing events recorded during capture: elapsed_time raises {type(exc).__name__}: "
              f"{str(exc).splitlines()[0]}")
    # 2. an external event recorded between two halves of a graph, waited for by a side stream
    ev = torch.cuda.Event(external=True)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g2.capture_begin()
        torch.cuda._sleep(SLEEP_CYCLES)
        ev.record()
        torch.cuda._sleep(SLEEP_CYCLES)
        g2.capture_end()
    side = torch.cuda.Stream()
    cur = torch.cuda.current_stream()
    t0, t_side, t_end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    torch.cuda.synchronize()
    t0.record(cur)
    g2.replay()
    t_end.record(cur)
    side.wait_event(ev)
    with torch.cuda.stream(side):
        x.add_(1)
        t_side.record(side)
    torch.cuda.synchronize()
    print(f"  external event recorded mid-graph: graph {t0.elapsed_time(t_end):.1f} ms, the side stream's "
          f"op after its wait done at {t0.elapsed_time(t_side):.1f} ms (half = released mid-graph)")
    # 3. an external wait inside the graph on an event recorded (after a sleep) before the replay
    ev2 = torch.cuda.Event(external=True)
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        g3.capture_begin()
        torch.cuda.current_stream().wait_event(ev2)
        x.add_(1)
        g3.capture_end()
    torch.cuda.synchronize()
    with torch.cuda.stream(side):
        torch.cuda._sleep(SLEEP_CYCLES)
        ev2.record(side)
    t0.record(cur)
    g3.replay()
    t_end.record(cur)
    torch.cuda.synchronize()
    print(f"  external wait inside the graph: replay done at {t0.elapsed_time(t_end):.1f} ms "
          "(~the sleep = it waited)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[296, 416, 512, 640, 832, 1440])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--chain-n", type=int, nargs="+", default=[416, 832])
    parser.add_argument("--skip-chain", action="store_true")
    parser.add_argument("--skip-bwd", action="store_true")
    args = parser.parse_args()
    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    from solution import fused_proj, int8_bwd, offload_int8

    traps()
    _patch()
    int8_bwd._STATE["on"] = offload_int8._STATE["on"] = True
    int8_bwd.compile_now()
    offload_int8.compile_now()
    mlps = _layers(4)
    probe = torch.zeros((512, HIDDEN), device="cuda", dtype=torch.float16)
    plans = [fused_proj._mlp_plan(m, probe) for m in mlps]
    assert all(p is not None for p in plans), "fused MLP plan declined"
    # column scales of every layer (cached on the modules) before any capture
    buffer = fused_proj._buffer(torch.device("cuda"), torch.float16, INTER * HIDDEN)
    for p in plans:
        for base in (p.gate, p.up, p.down):
            int8_bwd._column_scales(base, p.half, buffer)
    templates(args.n, args.layers, mlps, plans, bwd=not args.skip_bwd)
    if not args.skip_bwd:
        shared_pool(mlps, plans)
    if not args.skip_chain:
        autograd_chain(args.chain_n, mlps, args.layers)


if __name__ == "__main__":
    main()
