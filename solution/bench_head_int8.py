"""The int8 streamed head (head_int8.py) against the fp16 streamed head on the real checkpoint's weight:
the build, the GEMMs alone (fp16 mm, int8 TN, int8 NN), and the whole head per step at the run's
token counts and labelled fractions, with the copies (the ring of 2 slots, prefetched right before
the head: the worst case) and without them (every chunk resident: the compute alone).

    python -m solution.bench_head_int8 [--tokens 512/830/1440/2048] [--frac 0.3/0.6] [--reps 5]

(lists separated by "," or "/": tools/kaggle_run.py --bench takes no commas)
"""

import argparse
import re
import statistics
import time

import torch


def load_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", allow_patterns=["model.safetensors"])
    with safe_open(f"{path}/model.safetensors", "pt") as handle:
        return handle.get_tensor("model.embed_tokens.weight").half()


def _timed(fn, reps):
    times = []
    for i in range(reps + 2):
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        if i >= 2:
            times.append(begin.elapsed_time(end))
    return statistics.median(times)


def bench_gemms(n, n_lab, rows=4096, d=1536, chunks=37, reps=5):
    """One head's worth (37 full chunks) of each GEMM, ms."""
    dev = "cuda"
    h16 = torch.randn(n, d, device=dev).half()
    w16 = (torch.randn(rows, d, device=dev) * 0.05).half()
    p16 = torch.rand(n_lab, rows, device=dev).half()
    acc = torch.empty(n_lab, d, device=dev)
    logits16 = torch.empty(n, rows, dtype=torch.float16, device=dev)
    pad = max(-(-n // 8) * 8, 24)
    lab_pad = max(-(-n_lab // 8) * 8, 24)
    hq = torch.randint(-127, 127, (pad, d), dtype=torch.int8, device=dev)
    q = torch.randint(-127, 127, (rows, d), dtype=torch.int8, device=dev)
    qt = q.t().contiguous()
    pq = torch.randint(0, 127, (lab_pad, rows), dtype=torch.int8, device=dev)
    l32 = torch.empty(pad, rows, dtype=torch.int32, device=dev)
    g32 = torch.empty(lab_pad, d, dtype=torch.int32, device=dev)
    res = {}
    res["fwd fp16"] = _timed(lambda: [torch.mm(h16, w16.t(), out=logits16) for _ in range(chunks)], reps)
    res["bwd fp16"] = _timed(lambda: [torch.addmm(acc, p16, w16, out_dtype=torch.float32, out=acc)
                                      for _ in range(chunks)], reps)
    res["fwd int8"] = _timed(lambda: [torch._int_mm(hq, q.t(), out=l32) for _ in range(chunks)], reps)
    res["bwd int8 TN"] = _timed(lambda: [torch._int_mm(pq, qt.t(), out=g32) for _ in range(chunks)], reps)
    try:
        res["bwd int8 NN"] = _timed(lambda: [torch._int_mm(pq, q, out=g32) for _ in range(chunks)], reps)
    except RuntimeError as exc:
        res["bwd int8 NN"] = f"error {exc}"
    from .head_int8 import T_TILE, int8_transpose_kernel

    import triton

    res["transpose"] = _timed(lambda: [int8_transpose_kernel[(triton.cdiv(rows, T_TILE), triton.cdiv(d, T_TILE))](
        q, qt, rows, d, T=T_TILE, num_warps=4) for _ in range(chunks)], reps)
    assert torch.equal(qt, q.t())
    return res


def time_heads(streamer, n, frac, reps, resident):
    dev, d = streamer.device, streamer.hidden
    gen = torch.Generator(device=dev).manual_seed(n)
    hidden = (torch.randn(n, d, device=dev, generator=gen) * 2.0).half()
    targets = torch.randint(0, streamer.vocab, (n,), device=dev, generator=gen)
    targets[torch.rand(n, device=dev, generator=gen) >= frac] = -100
    order = torch.argsort((targets != -100).to(torch.int8), descending=True, stable=True)
    h16, t = hidden[order].contiguous(), targets[order].contiguous()
    n_lab = int((t != -100).sum())
    int8 = streamer._int8
    out = {"fp16": [], "int8": []}
    peak, outputs = {}, {}
    for r in range(reps + 1):
        for mode in (("fp16", "int8") if r % 2 == 0 else ("int8", "fp16")):
            int8.calls = 10**6 if mode == "int8" else -10**6
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            streamer.prefetch(n)
            if resident:
                torch.cuda.synchronize(dev)
            started = time.perf_counter()
            result = streamer.head(h16, t, n_lab, 1.0 / max(n_lab, 1))
            torch.cuda.synchronize(dev)
            if r:
                out[mode].append((time.perf_counter() - started) * 1000)
            peak[mode] = (torch.cuda.max_memory_allocated(dev) - base) / 2**20
            outputs[mode] = result
    lab = t != -100
    loss_rel = (outputs["int8"][0][lab].sum() / outputs["fp16"][0][lab].sum() - 1).item() if n_lab else 0.0
    ent_rel = (outputs["int8"][1].sum() / outputs["fp16"][1].sum() - 1).item()
    if n_lab:
        a, b = outputs["int8"][3][:n_lab], outputs["fp16"][3][:n_lab]
        dh_err = ((a - b).norm() / b.norm()).item()
    else:
        dh_err = 0.0
    return {k: statistics.median(v) for k, v in out.items()}, peak, n_lab, loss_rel, ent_rel, dh_err


def load_norm_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", allow_patterns=["model.safetensors"])
    with safe_open(f"{path}/model.safetensors", "pt") as handle:
        return handle.get_tensor("model.norm.weight").float()


def profile_heads(streamer, n, frac, resident, label):
    """GPU time per kernel name of one head (mean of 3), fp16 ring against int8 ring."""
    from torch.profiler import ProfilerActivity, profile

    dev, d = streamer.device, streamer.hidden
    gen = torch.Generator(device=dev).manual_seed(n)
    hidden = (torch.randn(n, d, device=dev, generator=gen) * 2.0).half()
    targets = torch.randint(0, streamer.vocab, (n,), device=dev, generator=gen)
    targets[torch.rand(n, device=dev, generator=gen) >= frac] = -100
    order = torch.argsort((targets != -100).to(torch.int8), descending=True, stable=True)
    h16, t = hidden[order].contiguous(), targets[order].contiguous()
    n_lab = int((t != -100).sum())
    int8 = streamer._int8
    for mode in ("fp16", "int8"):
        int8.calls = 10**6 if mode == "int8" else -10**6

        def once():
            streamer.prefetch(n)
            if resident:
                torch.cuda.synchronize(dev)
            streamer.head(h16, t, n_lab, 1.0 / max(n_lab, 1))
            torch.cuda.synchronize(dev)

        once()
        once()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                once()
        rows = []
        for evt in prof.key_averages():
            ms = getattr(evt, "self_device_time_total", 0) / 1000 / 3
            if ms > 0.05:
                rows.append((ms, evt.key[:70], evt.count // 3))
        rows.sort(reverse=True)
        total = sum(r[0] for r in rows if not r[1].lower().startswith("memcpy"))
        print(f"profile {label} N {n} labelled {n_lab} {mode}: kernels {total:.2f} ms per head; " + "; ".join(
            f"{name} x{cnt} {ms:.2f}" for ms, name, cnt in rows[:9]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", default="512,830,1440,2048")
    parser.add_argument("--frac", default="0.3,0.6")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--gemms", type=int, default=1)
    parser.add_argument("--profile", type=int, default=0, help="1: per-kernel GPU time instead of the sweep")
    parser.add_argument("--fwd-rows", default="512")
    args = parser.parse_args()
    import types

    from . import head_int8
    from .head_stream import HeadStreamer

    tokens = [int(x) for x in re.split(r"[,/]", args.tokens)]
    fracs = [float(x) for x in re.split(r"[,/]", args.frac)]
    print(torch.cuda.get_device_name(0), flush=True)
    if args.gemms:
        for n in tokens:
            for frac in fracs:
                res = bench_gemms(n, max(int(n * frac), 1), rows=args.rows)
                print(f"GEMMs per head N {n} lab {frac}: " + ", ".join(
                    f"{k} {v:.2f} ms" if isinstance(v, float) else f"{k} {v}" for k, v in res.items()), flush=True)
    started = time.perf_counter()
    weight = load_weight()
    print(f"weight {tuple(weight.shape)} {weight.dtype} loaded in {time.perf_counter() - started:.1f} s", flush=True)
    param = torch.nn.Parameter(weight.cuda(), requires_grad=False)
    streamer = HeadStreamer(param, rows=args.rows, slots=2)
    streamer.norm = types.SimpleNamespace(weight=load_norm_weight())  # the smoothing (head_int8.smoothing)
    del param
    torch.cuda.empty_cache()
    started = time.perf_counter()
    int8 = streamer.start_int8(log=print)
    int8.join()
    print(f"int8 copy ready {int8.ready} ({int8.error}) in {time.perf_counter() - started:.2f} s "
          f"(build + compile in the thread {int8.seconds:.2f} s), smoothing {int8.col is not None}", flush=True)
    fwd_rows = [int(x) for x in re.split(r"[,/]", args.fwd_rows)]
    configs = [(f"ring of 2, forward in {r}-row blocks", 2, False, r) for r in fwd_rows]
    configs.append(("resident", streamer.n_chunks, False, fwd_rows[0]))
    for label, slots, nn, rows in configs:
        head_int8.NN, head_int8.FWD_ROWS = nn, rows
        streamer.configure(rows=args.rows, slots=slots)
        for n in tokens:
            for frac in fracs:
                if args.profile:
                    profile_heads(streamer, n, frac, slots > 2, label)
                    continue
                times, peak, n_lab, loss_rel, ent_rel, dh_err = time_heads(streamer, n, frac, args.reps, slots > 2)
                print(f"{label}: N {n:5d} labelled {n_lab:5d}: head fp16 {times['fp16']:7.2f} ms, int8 "
                      f"{times['int8']:7.2f} ms ({times['int8'] - times['fp16']:+.2f}); memory above the start "
                      f"fp16 {peak['fp16']:.1f} MiB, int8 {peak['int8']:.1f} MiB; int8 vs fp16: loss "
                      f"{loss_rel * 100:+.3f}%, entropy {ent_rel * 100:+.3f}%, dh {dh_err:.4f}", flush=True)
    head_int8.NN, head_int8.FWD_ROWS = False, 1024
    streamer.configure(rows=args.rows, slots=2)
    print(streamer.stats(), flush=True)
    streamer.restore()


if __name__ == "__main__":
    main()
