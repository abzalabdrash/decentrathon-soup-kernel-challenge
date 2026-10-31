"""SOUP_SOLUTION_NORM_HALF on the GPU: one norm and its consumer's cast (rms_norm_rows_fwd, then
y.to(half)) against rms_norm_rows_fwd_half (the same norm writing the half copy too), for the plain
norm (layer 0's input norm) and the fused add (the other 55 norms of a step that get a copy), at the
token counts of the task's steps. Medians of CUDA-event times, every call bitwise checked.

    python -m solution.bench_norm_half [iters]
"""

import statistics
import sys

import torch

from . import rmsnorm

HIDDEN = 1536


def gpu_us(fn, iters):
    for _ in range(10):
        fn()
    times = []
    for _ in range(iters):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        times.append(begin.elapsed_time(end) * 1e3)
    return statistics.median(times)


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    print(torch.cuda.get_device_name(0), flush=True)
    g = torch.Generator("cuda").manual_seed(0)
    w = torch.rand(HIDDEN, device="cuda", generator=g) + 0.5
    print(f"{'tokens':>6s} {'norm':>5s} {'norm+cast us':>12s} {'norm_half us':>12s} {'delta us':>9s} "
          f"{'per step ms (1 plain + 55 add)':>30s}", flush=True)
    for n in (400, 600, 830, 1100, 1440, 2048):
        x = torch.randn(n, HIDDEN, device="cuda", generator=g) * 3
        branch = torch.randn(n, HIDDEN, device="cuda", generator=g).half()
        deltas = {}
        for add in (False, True):
            b = branch if add else None
            s = torch.empty_like(x) if add else None
            y = torch.empty_like(x)
            r = torch.empty(n, device="cuda")
            y16 = torch.empty(n, HIDDEN, device="cuda", dtype=torch.half)
            rmsnorm._forward(x, b, s, w, y, r, 1e-6)
            ref, ref_r = y.clone(), r.clone()
            rmsnorm._forward(x, b, s, w, y, r, 1e-6, y16)
            same = torch.equal(y, ref) and torch.equal(r, ref_r) and torch.equal(y16, ref.half())
            t_cast = gpu_us(lambda: (rmsnorm._forward(x, b, s, w, y, r, 1e-6), y.to(torch.half)), iters)
            t_half = gpu_us(lambda: rmsnorm._forward(x, b, s, w, y, r, 1e-6, y16), iters)
            deltas[add] = t_cast - t_half
            print(f"{n:6d} {'add' if add else 'plain':>5s} {t_cast:12.1f} {t_half:12.1f} "
                  f"{t_cast - t_half:9.1f}{'' if same else '  NOT BITWISE'}", flush=True)
        print(f"{n:6d} {'':5s} {'':12s} {'':12s} {'':9s} {(deltas[False] + 55 * deltas[True]) / 1e3:30.2f}",
              flush=True)


if __name__ == "__main__":
    main()
