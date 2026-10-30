"""Microbenchmark of the RMSNorm kernels against eager Qwen2RMSNorm, forward and backward apart.

    python solution/bench_rmsnorm.py            # the chosen launch settings
    python solution/bench_rmsnorm.py --sweep    # also rows per program x warps

Two numbers per case, in ms per call. "wall": CUDA events around 50 back-to-back calls after a
warm-up, median of 5 repeats; on small shapes this is bound by the CPU (autograd and the launch from
Python) when nothing else keeps the GPU busy. "gpu": the summed time of the CUDA kernels of one
call from torch.profiler, which is what the call costs inside a training step where the CPU runs
ahead of the GPU. Inputs as on the task: fp32 rows of 1536, frozen fp32 weight; the fused add takes
an fp16 branch (o_proj's output under autocast).
"""

import argparse
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from solution import rmsnorm as rn

EPS = 1e-6


def timed(fn, iters=50, warmup=10, repeats=5):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return statistics.median(times)


def gpu_time(fn, iters=20):
    """Summed CUDA kernel time of one call, ms."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return total_us / 1000 / iters


def measure(fn):
    return {"wall": timed(fn), "gpu": gpu_time(fn)}


def norm_case(n_rows, n_cols=1536):
    torch.manual_seed(0)
    x = torch.randn(n_rows, n_cols, device="cuda", requires_grad=True)
    w = 1.0 + 0.1 * torch.randn(n_cols, device="cuda")
    dy = torch.randn(n_rows, n_cols, device="cuda")
    y_ref = rn.reference_rmsnorm(x, w, EPS)
    y_ours = rn.rmsnorm(x, w, EPS)
    return {
        "eager fwd": measure(lambda: rn.reference_rmsnorm(x, w, EPS)),
        "triton fwd": measure(lambda: rn.rmsnorm(x, w, EPS)),
        "eager bwd": measure(lambda: torch.autograd.grad(y_ref, x, dy, retain_graph=True)),
        "triton bwd": measure(lambda: torch.autograd.grad(y_ours, x, dy, retain_graph=True)),
    }


def add_norm_case(n_rows, n_cols=1536):
    torch.manual_seed(0)
    res = torch.randn(n_rows, n_cols, device="cuda", requires_grad=True)
    br = torch.randn(n_rows, n_cols, device="cuda", dtype=torch.float16, requires_grad=True)
    w = 1.0 + 0.1 * torch.randn(n_cols, device="cuda")
    ds = torch.randn(n_rows, n_cols, device="cuda")
    dy = torch.randn(n_rows, n_cols, device="cuda")

    def eager():
        total = res + br
        return total, rn.reference_rmsnorm(total, w, EPS)

    outs_ref = eager()
    outs = rn.add_rmsnorm(res, br, w, EPS)
    return {
        "eager fwd": measure(eager),
        "triton fwd": measure(lambda: rn.add_rmsnorm(res, br, w, EPS)),
        "eager bwd": measure(lambda: torch.autograd.grad(outs_ref, (res, br), (ds, dy),
                                                         retain_graph=True)),
        "triton bwd": measure(lambda: torch.autograd.grad(outs, (res, br), (ds, dy),
                                                          retain_graph=True)),
    }


def report(title, rows):
    print(f"\n{title}")
    for kind in ("gpu", "wall"):
        print(f"[{kind}] {'case':<28}{'eager fwd':>10}{'triton fwd':>11}{'x':>6}{'eager bwd':>11}"
              f"{'triton bwd':>11}{'x':>6}")
        for name, r in rows:
            ef, tf, eb, tb = (r[k][kind] for k in ("eager fwd", "triton fwd", "eager bwd",
                                                   "triton bwd"))
            print(f"[{kind}] {name:<28}{ef:>10.4f}{tf:>11.4f}{ef / tf:>6.2f}{eb:>11.4f}"
                  f"{tb:>11.4f}{eb / tb:>6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), f"FWD rows/warps {rn.FWD_ROWS}/{rn.FWD_WARPS}, "
          f"BWD rows/warps {rn.BWD_ROWS}/{rn.BWD_WARPS}")
    rows = []
    for n in (2048, 826):
        rows.append((f"rmsnorm ({n}, 1536)", norm_case(n)))
        rows.append((f"add+rmsnorm ({n}, 1536)", add_norm_case(n)))
    report("ms per call", rows)
    if args.sweep:
        chosen = (rn.FWD_ROWS, rn.FWD_WARPS, rn.BWD_ROWS, rn.BWD_WARPS)
        sweep = []
        for n_rows_prog, warps in [(1, 2), (1, 4), (1, 8), (1, 16), (2, 4), (2, 8), (4, 8),
                                   (4, 16), (8, 16)]:
            rn.FWD_ROWS = rn.BWD_ROWS = n_rows_prog
            rn.FWD_WARPS = rn.BWD_WARPS = warps
            for n in (2048, 826):
                sweep.append((f"rows {n_rows_prog} warps {warps:<2} ({n})", norm_case(n)))
        rn.FWD_ROWS, rn.FWD_WARPS, rn.BWD_ROWS, rn.BWD_WARPS = chosen
        report("sweep: rows per program x warps, rmsnorm only", sweep)


if __name__ == "__main__":
    main()
