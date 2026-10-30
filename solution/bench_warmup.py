"""Measurements behind warmup.py. Every number comes from fresh processes with an empty triton cache
(a new TRITON_CACHE_DIR each), as in a measured run.

    python -m solution.bench_warmup imports     # import cost of triton, transformers, peft, trl, bnb
    python -m solution.bench_warmup compile     # compile time of every registered warm-up
    python -m solution.bench_warmup gil         # how much of the main thread the compiling thread takes
    python -m solution.bench_warmup first-call  # first training-shape launch, without / with warm-up
    python -m solution.bench_warmup all

Kernels are compiled for fp16, the T4's dtype, on any card.
"""

import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMPORTS = ("triton", "transformers", "peft", "trl", "bitsandbytes")


# ---------------------------------------------------------------- training-shape launches

def fused_head_call(n, n_labelled, num_items, chunk, d=64):
    """`fused_lm_head_loss` forward and backward as training calls it (hidden size `d`: the kernel
    never sees it)."""
    import torch

    from solution.fused_ce import fused_lm_head_loss

    v = 151936
    weight = (torch.randn(v, d, device="cuda") * 0.02).half()
    hidden = torch.randn(n, d, device="cuda", requires_grad=True)
    targets = torch.randint(0, v, (n,), device="cuda")
    targets[torch.randperm(n, device="cuda")[n_labelled:]] = -100
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, *_ = fused_lm_head_loss(hidden, weight, targets, num_items, chunk)
    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def swiglu_call(tokens):
    """The example SwiGLU forward and backward on a packed batch, fp16."""
    import torch

    from solution.swiglu import swiglu

    gate = torch.randn(1, tokens, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn(1, tokens, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    swiglu(gate, up).float().sum().backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


# ---------------------------------------------------------------- child processes

def _child_env(cache_root):
    return dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="triton-", dir=cache_root))


def _child(args, cache_root):
    out = subprocess.run([sys.executable, "-m", "solution.bench_warmup", *args], cwd=REPO,
                         env=_child_env(cache_root), capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"{args}: {out.stderr[-3000:]}")
    return dict(line.split("=", 1) for line in out.stdout.split("\n") if "=" in line)


def _prelude():
    """What harness/_train.py does before apply(): torch, the CUDA context, transformers."""
    import torch

    torch.empty(1, device="cuda")
    import transformers  # noqa: F401


def _setup_warmup():
    import torch

    from solution import fused_ce, swiglu, warmup  # noqa: F401 - what apply() has imported

    warmup.training_half_dtype = lambda: torch.float16
    return warmup


def child_import(ctx, name):
    import importlib

    if ctx == "torch":
        import torch  # noqa: F401
    else:
        _prelude()
    before = set(sys.modules)
    start = time.perf_counter()
    importlib.import_module(name)
    seconds = time.perf_counter() - start
    pulled = [m for m in IMPORTS if m != name and m not in before and m in sys.modules]
    print(f"seconds={seconds:.3f}\npulled={','.join(pulled)}")


def child_prelude():
    start = time.perf_counter()
    import torch

    torch_s = time.perf_counter() - start
    torch.empty(1, device="cuda")
    cuda_s = time.perf_counter() - start - torch_s
    import transformers  # noqa: F401

    print(f"torch={torch_s:.3f}\ncuda={cuda_s:.3f}\ntransformers="
          f"{time.perf_counter() - start - torch_s - cuda_s:.3f}")


def child_compile():
    """Every registered warm-up in the main thread, one after another, with the compile to cubin
    (`kernel.warmup`) and the launcher build + cubin load (`_init_handles`) timed apart."""
    _prelude()
    start = time.perf_counter()
    import triton  # noqa: F401

    print(f"import_triton={time.perf_counter() - start:.3f}")
    warmup = _setup_warmup()
    from triton.runtime.driver import driver

    start = time.perf_counter()
    driver.active.utils  # noqa: B018 - builds and loads triton's cuda_utils module
    print(f"cuda_utils={time.perf_counter() - start:.3f}")
    parts = {}
    original = warmup.compile_kernel

    def timed(kernel, *args, **kwargs):
        start = time.perf_counter()
        compiled = kernel.warmup(*args, grid=(1,), **kwargs)
        mid = time.perf_counter()
        compiled._init_handles()
        parts.setdefault("compile", []).append(mid - start)
        parts.setdefault("launcher", []).append(time.perf_counter() - mid)
        return compiled

    warmup.compile_kernel = timed
    for name, fn in warmup.WARMUPS:
        parts.clear()
        start = time.perf_counter()
        fn()
        total = time.perf_counter() - start
        print(f"{name.replace(' ', '_')}={total:.3f}|{len(parts['compile'])}|"
              f"{sum(parts['compile']):.3f}|{sum(parts['launcher']):.3f}")
    warmup.compile_kernel = original


def _spin(stop, samples):
    """Counts in the main thread; samples (time, count) every 4096 iterations until `stop(t)`."""
    n = 0
    while True:
        n += 1
        if not n & 4095:
            t = time.perf_counter()
            samples.append((t, n))
            if stop(t):
                return


def _rate(samples, t0, t1):
    inside = [(t, n) for t, n in samples if t0 <= t <= t1]
    if len(inside) < 2:
        return None
    return (inside[-1][1] - inside[0][1]) / (inside[-1][0] - inside[0][0])


def child_gil(mode):
    """The warm-up thread while the main thread spins a Python counter (mode spin) or just waits
    (mode wait). Per registration: its wall time and the counter's rate against the rate alone."""
    _prelude()
    import triton  # noqa: F401

    warmup = _setup_warmup()
    windows = {}

    def wrap(name, fn):
        def run():
            start = time.perf_counter()
            try:
                fn()
            finally:
                windows[name] = (start, time.perf_counter())
        return run

    warmup.WARMUPS[:] = [(name, wrap(name, fn)) for name, fn in warmup.WARMUPS]
    samples = []
    if mode == "spin":
        t_end = time.perf_counter() + 1.0
        _spin(lambda t: t >= t_end, samples)
        base = _rate(samples, samples[0][0], samples[-1][0])
        samples.clear()
    start = time.perf_counter()
    warmup.start_background_warmup(log=False)
    if mode == "spin":
        _spin(lambda t: warmup._DONE.is_set(), samples)
    warmup.wait_for_warmup()
    print(f"total={time.perf_counter() - start:.3f}")
    for name, (t0, t1) in windows.items():
        line = f"{name.replace(' ', '_')}={t1 - t0:.3f}"
        if mode == "spin":
            rate = _rate(samples, t0, t1)
            line += f"|{rate / base:.3f}" if rate else "|nan"
        print(line)
    if mode == "spin":
        t0 = min(t for t, _ in windows.values())
        t1 = max(t for _, t in windows.values())
        print(f"all={t1 - t0:.3f}|{_rate(samples, t0, t1) / base:.3f}")


def child_first_call(mode):
    """First and second training-shape launches in this fresh process, after a warm-up (warm) or
    without one (cold)."""
    _prelude()
    import triton  # noqa: F401

    warmup = _setup_warmup()
    if mode == "warm":
        start = time.perf_counter()
        warmup.start_background_warmup(log=False)
        warmup.wait_for_warmup()
        print(f"warmup_s={time.perf_counter() - start:.3f}")
    print(f"head_first={fused_head_call(826, 600, 900.0, 512):.4f}")
    print(f"head_second={fused_head_call(826, 600, 900.0, 512):.4f}")
    print(f"swiglu_first={swiglu_call(826):.4f}")
    print(f"swiglu_second={swiglu_call(826):.4f}")


# ---------------------------------------------------------------- reports

def report_imports(cache_root, repeats=3):
    pre = [_child(["child-prelude"], cache_root) for _ in range(repeats)]
    print("harness prelude, s (median of %d): import torch %.2f, CUDA context %.2f, "
          "import transformers %.2f" % (repeats, *(statistics.median(float(p[k]) for p in pre)
                                                   for k in ("torch", "cuda", "transformers"))))
    print(f"{'module':14s} {'after torch, s':>15s} {'after prelude, s':>17s}  also loads")
    for name in IMPORTS:
        row = []
        pulled = ""
        for ctx in ("torch", "harness"):
            runs = [_child(["child-import", ctx, name], cache_root) for _ in range(repeats)]
            row.append(statistics.median(float(r["seconds"]) for r in runs))
            pulled = runs[-1]["pulled"] if ctx == "harness" else pulled
        print(f"{name:14s} {row[0]:15.2f} {row[1]:17.2f}  {pulled or '-'}")


def report_compile(cache_root):
    res = _child(["child-compile"], cache_root)
    print(f"import triton {float(res.pop('import_triton')):.2f} s (after the prelude); "
          f"triton's cuda_utils build {float(res.pop('cuda_utils')):.2f} s")
    print(f"{'warm-up':38s} {'total, s':>9s} {'compiles':>9s} {'to cubin, s':>12s} "
          f"{'launcher+load, s':>17s}")
    for name, value in res.items():
        total, count, comp, launch = value.split("|")
        print(f"{name:38s} {float(total):9.2f} {count:>9s} {float(comp):12.2f} {float(launch):17.2f}")


def report_gil(cache_root):
    wait = _child(["child-gil", "wait"], cache_root)
    spin = _child(["child-gil", "spin"], cache_root)
    print("warm-up thread: wall time with the main thread waiting / spinning a Python counter, and "
          "the counter's rate during it against the counter alone")
    print(f"{'warm-up':38s} {'waiting, s':>11s} {'spinning, s':>12s} {'counter rate':>13s}")
    for name in wait:
        if name == "total":
            continue
        w = float(wait[name])
        s, rate = spin[name].split("|")
        print(f"{name:38s} {w:11.2f} {float(s):12.2f} {float(rate):13.2f}")
    s, rate = spin["all"].split("|")
    print(f"{'all':38s} {float(wait['total']):11.2f} {float(s):12.2f} {float(rate):13.2f}")


def report_first_call(cache_root):
    cold = _child(["child-first-call", "cold"], cache_root)
    warm = _child(["child-first-call", "warm"], cache_root)
    cold_first = float(cold["head_first"]) + float(cold["swiglu_first"])
    warm_first = float(warm["head_first"]) + float(warm["swiglu_first"])
    steady = float(warm["head_second"]) + float(warm["swiglu_second"])
    print(f"fused head (826 tokens, 512-row chunks) first call: cold {float(cold['head_first']):.3f} s, "
          f"after warm-up {float(warm['head_first']):.3f} s, steady {float(warm['head_second']):.3f} s")
    print(f"SwiGLU fwd+bwd (826 x 8960) first call: cold {float(cold['swiglu_first']):.3f} s, "
          f"after warm-up {float(warm['swiglu_first']):.3f} s, steady {float(warm['swiglu_second']):.3f} s")
    print(f"warm-up thread took {float(warm['warmup_s']):.2f} s")
    print(f"cold_first_s={cold_first:.4f} warm_first_s={warm_first:.4f} steady_s={steady:.4f}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    children = {"child-import": child_import, "child-prelude": child_prelude,
                "child-compile": child_compile, "child-gil": child_gil,
                "child-first-call": child_first_call}
    if cmd in children:
        children[cmd](*sys.argv[2:])
        return
    reports = {"imports": report_imports, "compile": report_compile, "gil": report_gil,
               "first-call": report_first_call}
    root = os.environ.get("TRITON_CACHE_DIR") or tempfile.mkdtemp(prefix="bench-warmup-")
    os.makedirs(root, exist_ok=True)
    for name in (reports if cmd == "all" else [cmd]):
        print(f"== {name}")
        reports[name](root)


if __name__ == "__main__":
    main()
