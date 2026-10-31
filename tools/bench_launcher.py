"""Per-launch CPU cost of triton's launcher built at -O3 (triton's default) and at -O0
(SOUP_SOLUTION_LAUNCHER_OPT=0, warmup.install_launcher_build), on the GPU.

    python -m tools.bench_launcher            # both, each in a fresh process with its own cache

Each child compiles one small kernel with 12 arguments (8 pointers, like rms_norm_rows_fwd), launches
it N times in a loop and reports microseconds per launch (wall time of the loop, synchronized), best
of R repeats; launches are asynchronous, so this is the host-side cost the training loop pays per
launch. A benchmark only (it starts subprocesses); not used by the run, and outside solution/ so its
kernel is not one of the solution's."""

import os
import subprocess
import sys
import tempfile

N, R = 20000, 5


def child():
    import time

    import torch
    import triton
    import triton.language as tl

    from solution.warmup import install_launcher_build

    print(f"launcher build hook installed: {install_launcher_build()}", flush=True)

    @triton.jit
    def bench_kernel(a, b, c, d, e, f, g, h, n, stride, eps, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(a + offs, mask=mask)
        tl.store(h + offs, x * eps, mask=mask)

    ts = [torch.zeros(1024, device="cuda") for _ in range(8)]
    bench_kernel[(1,)](*ts, 1024, 1024, 0.5, BLOCK=1024)
    torch.cuda.synchronize()
    best = None
    for _ in range(R):
        start = time.perf_counter()
        for _ in range(N):
            bench_kernel[(1,)](*ts, 1024, 1024, 0.5, BLOCK=1024)
        torch.cuda.synchronize()
        per = (time.perf_counter() - start) / N * 1e6
        best = per if best is None else min(best, per)
    print(f"RESULT opt={os.environ.get('SOUP_SOLUTION_LAUNCHER_OPT', '3 (triton default)')}: "
          f"{best:.2f} us per launch (best of {R} x {N})", flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child()
        return
    for opt in (None, "0", None, "0"):
        env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="soup-launcher-bench-"))
        env.pop("SOUP_SOLUTION_LAUNCHER_OPT", None)
        env.pop("SOUP_SOLUTION_LAUNCHER_NICE", None)
        if opt is not None:
            env["SOUP_SOLUTION_LAUNCHER_OPT"] = opt
        subprocess.run([sys.executable, "-m", "tools.bench_launcher", "--child"], env=env, check=False)


if __name__ == "__main__":
    main()
