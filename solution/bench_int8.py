"""int8 tensor-core GEMMs on the T4 for the backward's input-gradient products (night task N2, step 1:
measurement only, nothing here touches the training path).

The question: does `torch._int_mm` (cuBLASLt, int8 x int8 -> int32, IMMA on sm_75) run on the T4,
and how much faster than the fp16 GEMMs the backward runs today is it on the training's shapes?
The input gradient of a frozen 4-bit linear layer is grad_x (M, K) = g (M, N) @ W (N, K); with
per-token scales for g and per-column scales for W (the contraction runs over N, so the scales stay
outside the sum) it can be computed as s_g[m] * s_w[k] * (g_q @ W_q)[m, k] in int32.

For every backward GEMM of a decoder layer (q|k|v, o, gate|up, down) and M in --m:
- fp16 as trained: torch.mm in fp16 with fp32 accumulation (the operands' layouts of fused_proj.py);
- int8: torch._int_mm(a int8 (M, N) row-major, b int8 (N, K) column-major) -> int32;
- int8 total: the GEMM plus the per-token quantization of g and the int32 -> fp32 scaling, as torch
  ops (an own kernel would fuse each into one pass);
- the relative error of the int8 product against fp32 on random data with a few large columns.
Kernel names come from torch.profiler (IMMA kernels have 'imma' or 'i8816' in their names); the SM
clock is sampled by nvidia-smi while the bench runs (the T4 throttles under GEMM load).

`python -m solution.bench_int8 [--m 832 1440] [--iters 30] [--rounds 7] [--out int8.json]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_int8` runs it on Kaggle).
"""

import argparse
import json
import statistics
import subprocess
import threading

import torch

D, INTER, QKV = 1536, 8960, 2048


def backward_shapes():
    """(name, N contracted, K out): grad_x (M, K) = g (M, N) @ W (N, K) of each frozen layer."""
    return [("qkv_bwd", QKV, D), ("o_bwd", D, D), ("gate_up_bwd", 2 * INTER, D),
            ("down_bwd", D, INTER)]


def forward_shapes():
    """(name, K contracted, N out) of the forward GEMMs, for reference: y (M, N) = x (M, K) @ W.T."""
    return [("qkv_fwd", D, QKV), ("o_fwd", D, D), ("gate_up_fwd", D, 2 * INTER),
            ("down_fwd", INTER, D)]


def _time(fn, iters, rounds):
    """Median over rounds of the mean time per call (us), CUDA events around `iters` calls."""
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(out)


def _kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for evt in prof.events():
        if evt.device_type.name == "CUDA" and evt.name not in names:
            names.append(evt.name)
    return names


def quantize_rows(t):
    """Per-row absmax int8 of a 2-D tensor: (q int8, scale fp32 (rows, 1))."""
    scale = t.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127.0
    q = torch.round(t.float() / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


def quantize_cols(t):
    """Per-column absmax int8 of a 2-D tensor: (q int8, scale fp32 (1, cols))."""
    scale = t.abs().amax(dim=0, keepdim=True).float().clamp_min(1e-12) / 127.0
    q = torch.round(t.float() / scale).clamp_(-127, 127).to(torch.int8)
    return q, scale


class _Clock:
    """nvidia-smi's SM clock and power, sampled every 100 ms while active."""

    def __init__(self):
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                text = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm,power.draw", "--format=csv,noheader,nounits",
                     "-i", str(torch.cuda.current_device())],
                    capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                mhz, watts = (float(v) for v in text.split(","))
                self.samples.append((mhz, watts))
            except Exception:  # noqa: BLE001 - no nvidia-smi (local CPU box): no clock column
                return
            self._stop.wait(0.1)

    def __enter__(self):
        self.samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop.clear()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def mean(self):
        if not self.samples:
            return None, None
        return (statistics.mean(s[0] for s in self.samples), statistics.mean(s[1] for s in self.samples))


def bench_backward(m, n, k, iters, rounds, gen):
    """fp16 grad_x = g @ W against int8 with per-token / per-column scales."""
    g = torch.randn(m, n, generator=gen, device="cuda")
    g[:, :: max(1, n // 7)] *= 20.0  # a few large columns, as gradients have
    w = torch.randn(n, k, generator=gen, device="cuda") * 0.02
    g16, w16 = g.half(), w.half()
    w16_t = w16.t().contiguous()  # (K, N) contiguous: fused_proj's dequantized weight is (out, in)
    ref = g16.float() @ w16.float()

    def fp16():  # as fused_proj's backward: grad (M, N) @ W (N, K), W a row-major (out, in) buffer
        return torch.mm(g16, w16)

    gq, gs = quantize_rows(g16)
    wq, ws = quantize_cols(w16)
    wq_cm = wq.t().contiguous().t()  # (N, K) column-major, as cuBLASLt's int8 kernels want "TN"
    try:
        out32 = torch._int_mm(gq, wq_cm)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    def int8():
        return torch._int_mm(gq, wq_cm)

    def int8_rm():  # both operands row-major ("NN"), in case cuBLASLt takes it as well
        return torch._int_mm(gq, wq)

    try:
        int8_rm()
        rm_ok = True
    except Exception:  # noqa: BLE001
        rm_ok = False

    def int8_total():
        q, s = quantize_rows(g16)
        return torch._int_mm(q, wq_cm).float() * s * ws

    approx = out32.float() * gs * ws
    err = ((approx - ref).norm() / ref.norm()).item()
    err16 = ((fp16().float() - ref).norm() / ref.norm()).item()
    with _Clock() as clock:
        t16 = _time(fp16, iters, rounds)
        t8 = _time(int8, iters, rounds)
        t8_total = _time(int8_total, iters, rounds)
        t8_rm = _time(int8_rm, iters, rounds) if rm_ok else None
        t16b = _time(fp16, iters, rounds)  # again: the clock drifts during a run
    mhz, watts = clock.mean()
    flops = 2.0 * m * n * k
    t16m = min(t16, t16b)
    return {"fp16_us": round(t16m, 1), "int8_us": round(t8, 1), "int8_total_us": round(t8_total, 1),
            "int8_rowmajor_us": None if t8_rm is None else round(t8_rm, 1),
            "speedup_gemm": round(t16m / t8, 2), "speedup_total": round(t16m / t8_total, 2),
            "fp16_tflops": round(flops / t16m / 1e6, 1), "int8_tops": round(flops / t8 / 1e6, 1),
            "rel_err_int8": round(err, 5), "rel_err_fp16": round(err16, 6),
            "sm_mhz": None if mhz is None else round(mhz), "watts": None if watts is None else round(watts, 1),
            "kernels_fp16": _kernels(fp16), "kernels_int8": _kernels(int8)}


def bench_forward(m, k, n, iters, rounds, gen):
    """fp16 y = x @ W.T against int8 (per-token x, per-row W), for reference only."""
    x16 = (torch.randn(m, k, generator=gen, device="cuda")).half()
    w16 = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).half()
    xq, _ = quantize_rows(x16)
    wq, _ = quantize_rows(w16)

    def fp16():
        return torch.mm(x16, w16.t())

    try:
        torch._int_mm(xq, wq.t())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}

    def int8():
        return torch._int_mm(xq, wq.t())

    t16 = _time(fp16, iters, rounds)
    t8 = _time(int8, iters, rounds)
    return {"fp16_us": round(t16, 1), "int8_us": round(t8, 1), "speedup_gemm": round(t16 / t8, 2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[832, 1440])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--out")
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, sm_{props.major}{props.minor}, {props.multi_processor_count} SMs, "
          f"torch {torch.__version__}, cuda {torch.version.cuda}", flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    results = {"device": props.name, "backward": {}, "forward": {}}
    for m in args.m:
        for name, n, k in backward_shapes():
            r = bench_backward(m, n, k, args.iters, args.rounds, gen)
            results["backward"][f"{name}@{m}"] = r
            if "error" in r:
                print(f"{name:12s} M={m:5d}: {r['error']}", flush=True)
                continue
            print(f"{name:12s} M={m:5d}: fp16 {r['fp16_us']:8.1f} us ({r['fp16_tflops']:5.1f} TF)  "
                  f"int8 {r['int8_us']:8.1f} us ({r['int8_tops']:5.1f} TOPS)  x{r['speedup_gemm']:.2f}  "
                  f"with quant+scale {r['int8_total_us']:8.1f} us x{r['speedup_total']:.2f}  "
                  f"row-major b {r['int8_rowmajor_us']} us  "
                  f"err int8 {r['rel_err_int8']:.4f} fp16 {r['rel_err_fp16']:.6f}  "
                  f"{r['sm_mhz']} MHz {r['watts']} W", flush=True)
            print(f"{'':12s}   fp16 kernels {r['kernels_fp16']}\n{'':12s}   int8 kernels {r['kernels_int8']}",
                  flush=True)
        for name, k, n in forward_shapes():
            r = bench_forward(m, k, n, args.iters, args.rounds, gen)
            results["forward"][f"{name}@{m}"] = r
            if "error" in r:
                print(f"{name:12s} M={m:5d}: {r['error']}", flush=True)
            else:
                print(f"{name:12s} M={m:5d}: fp16 {r['fp16_us']:8.1f} us  int8 {r['int8_us']:8.1f} us  "
                      f"x{r['speedup_gemm']:.2f}", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
