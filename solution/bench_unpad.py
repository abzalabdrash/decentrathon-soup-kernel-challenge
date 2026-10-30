"""Microbenchmark of the packed step's attention and token alignment (unpad.py) at the task's
shapes: 12 q heads and 2 kv heads of 128 in fp16, the rows of an average step (826 tokens in 4 rows)
and of the longest one (1427 tokens), each with its alignment tail.

    python solution/bench_unpad.py
    python solution/bench_unpad.py --iters 50

Attention, per path (one variable-length launch vs the per-row SDPA loop on the memory-efficient
backend with the kv heads repeated before it), forward and backward apart: "gpu" ms as the summed
time of the call's CUDA kernels from torch.profiler, "cpu" ms to issue the call from an idle GPU
(median of `--iters`), and the number of kernel launches. GEMMs: the MLP's transposed gate|up
product (17920, N) and the input gradient through it, with N as the real token count and as the
padded one, timed with CUDA events.
"""

import argparse
import statistics
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile

from solution import unpad

HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
STEPS = {"826 tokens": [212, 187, 310, 117], "1427 tokens": [512, 401, 311, 203]}


def _profile(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(events), sum(e.device_time for e in events) / 1000.0


def _cpu(fn, iters):
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    torch.cuda.synchronize()
    return statistics.median(times)


def _attention(lengths, varlen, iters):
    n = unpad._pad_count(sum(lengths)) + sum(lengths)
    bounds = unpad.sequences([(e - l, e) for e, l in zip(torch.tensor(lengths).cumsum(0).tolist(),
                                                         lengths)], n)
    width = (HEADS + 2 * KV_HEADS) * HEAD_DIM
    qkv = torch.randn(n, width, device="cuda").half().requires_grad_(True)
    nq, nk = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    q = qkv[:, :nq].view(1, n, HEADS, HEAD_DIM).transpose(1, 2)
    k = qkv[:, nq:nq + nk].view(1, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
    v = qkv[:, nq + nk:].view(1, n, KV_HEADS, HEAD_DIM).transpose(1, 2)
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda").half()
    unpad._STATE["varlen"] = varlen

    def forward():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return unpad.attention(q, k, v, bounds, HEAD_DIM ** -0.5)

    for _ in range(3):
        forward().backward(grad)
    holder = {}
    fwd = _profile(lambda: holder.update(out=forward()))
    bwd = _profile(lambda: holder.pop("out").backward(grad))
    cpu_fwd = _cpu(forward, iters)
    outs = [forward() for _ in range(iters)]
    times = []
    for out in outs:
        torch.cuda.synchronize()
        start = time.perf_counter()
        out.backward(grad)
        times.append((time.perf_counter() - start) * 1000.0)
    return n, fwd, bwd, cpu_fwd, statistics.median(times)


def _gemm_ms(fn, iters):
    for _ in range(5):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _gemms(n, iters):
    """The fused MLP's (gate|up)^T = W @ x^T into an (17920, N8) buffer (N8 = N rounded up to 8,
    as fused_proj allocates it) and the input gradient G^T @ W, at N tokens."""
    w = torch.randn(17920, 1536, device="cuda").half()
    x = torch.randn(n, 1536, device="cuda").half()
    ld = -(-n // 8) * 8
    out = torch.empty(17920, ld, device="cuda").half()
    forward = _gemm_ms(lambda: torch.mm(w, x.t(), out=out[:, :n]), iters)
    backward = _gemm_ms(lambda: torch.mm(out[:, :n].t(), w, out_dtype=torch.float32), iters)
    return forward, backward


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    print(torch.cuda.get_device_name())
    for name, lengths in STEPS.items():
        for varlen in (False, True):
            with unpad.varlen([]):  # cu_seqlens built once, as in a step
                n, (fl, fg), (bl, bg), fc, bc = _attention(lengths, varlen, args.iters)
            label = "varlen " if varlen else "per-row"
            print(f"attention {name} (N={n}) {label}: fwd gpu {fg:.3f} ms cpu {fc:.3f} ms "
                  f"{fl} launches; bwd gpu {bg:.3f} ms cpu {bc:.3f} ms {bl} launches")
    unpad._STATE["varlen"] = True
    for n in (826, 832, 1427, 1440):
        f, b = _gemms(n, args.iters)
        print(f"gate|up GEMMs N={n}: forward {f:.3f} ms, input gradient {b:.3f} ms")


if __name__ == "__main__":
    main()
