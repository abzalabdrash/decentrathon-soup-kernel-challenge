"""Microbenchmark of one layer's attention backward (task A2): aten's memory-efficient varlen
backward as unpad.py runs it (k/v expansion, the fmha backward, the group sums) against attn_bwd.py's
math path (batched GEMMs and own kernels), on packed batches of 4 rows plus the alignment tail,
12 query heads and 2 kv heads of 128 (Qwen2.5-1.5B), fp16.

    python -m solution.bench_attn_bwd [iters]

Prints per shape: GPU ms per call (median of CUDA-event timings), host ms to enqueue one call (GPU
idle), peak allocated MiB above the inputs, and the error of both against an fp32 reference
(max |err| / max |ref| per gradient); then bmm(out_dtype=fp32) against bmm + cast, a per-launch
breakdown of the math path at L=208 and a sweep of the row kernels' warps.
"""

import statistics
import sys
import time

import torch

from . import attn_bwd
from .unpad import _efficient_backward, _expand_heads, varlen_forward

H, G, D = 12, 2, 128
SCALE = D ** -0.5


def batch(lengths, seed=0):
    n_real = sum(lengths)
    pad = (-n_real) % 16
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    if pad:
        bounds.append(bounds[-1] + pad)
    n = bounds[-1]
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, n, H, D, device="cuda", generator=gen).half()
    k = torch.randn(1, n, G, D, device="cuda", generator=gen).half()
    v = torch.randn(1, n, G, D, device="cuda", generator=gen).half()
    dout = torch.randn(1, n, H, D, device="cuda", generator=gen).half()
    dout[:, n_real:] = 0  # the tail's grad_out is zero in training
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return q, k, v, dout, cu, tuple(bounds), len(lengths)


def original(q, k, v, dout, cu, bounds, out, lse, seed, offset):
    max_len = max(b - a for a, b in zip(bounds[:-1], bounds[1:]))
    gq, gk, gv, _ = _efficient_backward(dout, q, _expand_heads(k, H // G), _expand_heads(v, H // G),
                                        out, cu, max_len, lse, seed, offset, SCALE)
    gk = gk.unflatten(2, (G, H // G)).sum(3)
    gv = gv.unflatten(2, (G, H // G)).sum(3)
    return gq, gk, gv


def reference(q, k, v, dout, bounds, n_seq):
    gq = torch.zeros(q.shape, dtype=torch.float32, device="cuda")
    gk = torch.zeros(k.shape, dtype=torch.float32, device="cuda")
    gv = torch.zeros(v.shape, dtype=torch.float32, device="cuda")
    for b in range(n_seq):
        a, e = bounds[b], bounds[b + 1]
        qb = q[0, a:e].float().transpose(0, 1).requires_grad_()  # (H, L, D)
        kb = k[0, a:e].float().transpose(0, 1).requires_grad_()
        vb = v[0, a:e].float().transpose(0, 1).requires_grad_()
        ke, ve = kb.repeat_interleave(H // G, 0), vb.repeat_interleave(H // G, 0)
        s = qb @ ke.transpose(1, 2) * SCALE
        mask = torch.ones(e - a, e - a, dtype=torch.bool, device="cuda").tril()
        o = s.masked_fill(~mask, float("-inf")).softmax(-1) @ ve
        o.backward(dout[0, a:e].float().transpose(0, 1))
        gq[0, a:e], gk[0, a:e], gv[0, a:e] = (t.grad.transpose(0, 1) for t in (qb, kb, vb))
    return gq, gk, gv


def errors(got, want):
    return [((g.float() - w).abs().max() / w.abs().max().clamp_min(1e-30)).item() for g, w in zip(got, want)]


def gpu_ms(fn, iters):
    times = []
    for _ in range(3):
        fn()
    for _ in range(iters):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end))
    return statistics.median(times)


def host_ms(fn, iters):
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    return statistics.median(times)


def peak_mib(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    keep = fn()
    torch.cuda.synchronize()
    del keep
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    attn_bwd._STATE.update(mode="math", min_tokens=0, max_len=4096, scratch="own", scratch_mb=16)
    print(torch.cuda.get_device_name(0), torch.__version__, flush=True)
    shapes = []
    for big in (208, 256, 320, 384, 448, 512):
        shapes.append((f"L={big} even", [big, big - 4, big - 12, big - 20]))
        # the longest row and three of the task data's median length (140 tokens)
        shapes.append((f"L={big} +3x140", [big, 140, 140, 140]))
    # batches like the task's (row lengths: median 140, max 512)
    for lengths in ([380, 140, 120, 100], [300, 200, 150, 90], [250, 230, 180, 120], [512, 160, 120, 60],
                    [450, 300, 140, 80]):
        shapes.append(("data " + "/".join(map(str, lengths)), lengths))
    blocks = (1, 2, 4)
    print(f"{'shape':24s} {'N':>5s} {'orig':>7s} " + " ".join(f"{'nb=' + str(b):>7s}" for b in blocks)
          + f" {'host o/m2':>11s} {'MiB o/m2':>11s}  err orig (q k v) | err nb=2 (q k v)  bins", flush=True)
    for name, lengths in shapes:
        q, k, v, dout, cu, bounds, n_seq = batch(lengths)
        max_len = max(b - a for a, b in zip(bounds[:-1], bounds[1:]))
        out, lse, seed, offset = varlen_forward(q, k, v, cu, max_len, SCALE)
        run_o = lambda: original(q, k, v, dout, cu, bounds, out, lse, seed, offset)  # noqa: E731
        want = reference(q, k, v, dout, bounds, n_seq)
        eo = errors(run_o(), want)
        to = gpu_ms(run_o, iters)
        times, em, hm, mm, tail_zero = [], None, None, None, True
        for nb in blocks:
            plan = attn_bwd.Plan(bounds, n_seq, q.device, blocks=nb)
            run_m = lambda: attn_bwd.math_backward(dout, q, k, v, plan, SCALE)  # noqa: E731
            got = run_m()
            tail_zero &= all(bool((t[:, bounds[n_seq]:] == 0).all()) for t in got)
            err = errors(got, want)
            times.append(gpu_ms(run_m, iters))
            if nb == 2:
                em, hm, mm, bins = err, host_ms(run_m, 10), peak_mib(run_m), plan.bins
        ho, mo = host_ms(run_o, 10), peak_mib(run_o)
        print(f"{name:24s} {bounds[-1]:5d} {to:7.3f} " + " ".join(f"{t:7.3f}" for t in times)
              + f" r2={attn_bwd.work_ratio(list(lengths), 2):4.2f}"
              + f" {ho:5.2f}/{hm:5.2f} {mo:5.1f}/{mm:5.1f}  "
              f"{eo[0]:.1e} {eo[1]:.1e} {eo[2]:.1e} | {em[0]:.1e} {em[1]:.1e} {em[2]:.1e}  {bins}"
              f"{'' if tail_zero else '  TAIL NOT ZERO'}", flush=True)

    # bmm with an fp32 output against a half bmm and a cast (the scores' shape at L=208)
    a = torch.randn(8, 6 * 208, 128, device="cuda").half()
    b = torch.randn(8, 208, 128, device="cuda").half()
    out32 = torch.empty(8, 6 * 208, 208, device="cuda")
    t32 = gpu_ms(lambda: torch.bmm(a, b.transpose(1, 2), out_dtype=torch.float32, out=out32), iters)
    t16 = gpu_ms(lambda: torch.bmm(a, b.transpose(1, 2)).float(), iters)
    t16o = gpu_ms(lambda: torch.bmm(a, b.transpose(1, 2)), iters)
    print(f"scores GEMM L=208: bmm out_dtype=fp32 {t32:.3f} ms, half bmm + cast {t16:.3f} ms, "
          f"half bmm alone {t16o:.3f} ms", flush=True)

    # the row kernels' warps (nb=2, L=208 even)
    q, k, v, dout, cu, bounds, n_seq = batch([208, 204, 196, 188])
    plan = attn_bwd.Plan(bounds, n_seq, q.device, blocks=2)
    for warps in (1, 2, 4, 8):
        attn_bwd.NUM_WARPS_ROWS = warps
        ms = gpu_ms(lambda: attn_bwd.math_backward(dout, q, k, v, plan, SCALE), iters)
        print(f"  row kernels num_warps={warps}: math nb=2 {ms:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
