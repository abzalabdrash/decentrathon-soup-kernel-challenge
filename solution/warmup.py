"""Triton compilation of the run's kernels in a background thread while Soup loads the model.

Every measured run starts with empty compiler caches, so the first training step compiles every
kernel it meets (Kaggle T4, commit d649c7c: the first two steps took about 4.5 s against 0.9 s for
two ordinary ones; from the end of apply() to the first step there are about 6 s: the model load,
3.1 s, and the trainer setup, 2.9 s). Here a daemon thread compiles them ahead, so that the first
step finds them in Triton's in-memory cache: the 11 variants a real step compiles (rms_norm_rows_fwd
and _bwd two each, nf4_dequant_kernel, rope_qk_kernel forward and backward, fused_ce_rows_kernel two,
the example SwiGLU forward and backward) and a rarer third one of fused_ce_rows_kernel.

What makes a launch compile (triton 3.6.0, triton/runtime/jit.py)
-----------------------------------------------------------------
`JITFunction.run` (jit.py:695) looks the launch up in a dict per device (`device_caches[device]`,
jit.py:707, 771) under `compute_cache_key(specialization, options)` (jit.py:563-583, called at
:712) and compiles only on a miss (jit.py:713-721). The key holds, per argument, what the binder
built by `create_function_from_signature` gives (jit.py:402-421; `specialize_impl` is the C++
`native_specialize_impl`, jit.py:439; its Python twin is triton/backends/compiler.py:83-92), plus
`str(options)`:

- a tensor: `*fp16`, `*i64`, ... (the dtype) and "D" when `data_ptr() % 16 == 0`;
- an int: `i32` (`i64` past 2**31 - 1) and "D" when it is divisible by 16; the value 1 becomes a
  compile-time constant `("constexpr", 1)`, a compile of its own;
- a float: `fp32`, never specialized (a bool is `u1`, None a constant);
- a `tl.constexpr` parameter: its value (jit.py:404);
- the options: num_warps, num_stages, num_ctas, debug, ... (`do_not_specialize` and
  `do_not_specialize_on_alignment` switch the int and alignment parts off, jit.py:405-409).

The grid, the shapes and the values of the other ints are not in the key: a kernel with a runtime
loop `for start in range(0, V, BLOCK)` compiles once for every V of the same class. Checked on the
installed build (native_specialize_impl): 1 -> ("constexpr", 1), 0, 16, 512, 151936 -> ("i32",
"D"), 2, 387 -> ("i32", ""), 2**31 -> ("i64", "D"), 0.5 -> ("fp32", None), an fp16 tensor at an
offset of 1 element -> ("*fp16", ""), at 8 elements -> ("*fp16", "D").

So a warm-up has to repeat, for every argument, the dtype, the 16-byte alignment, the
divisible-by-16 class, the "== 1" class and the constexpr values of the training launch, with the
same num_warps / num_stages. Nothing else matters.

How it compiles without running anything
----------------------------------------
`kernel.warmup(*args, grid=..., **kwargs)` (jit.py:358) goes through the same `run` with
`warmup=True`: same key, same compile, no launch (jit.py:731). A torch dtype stands for a tensor
(`MockTensor`, jit.py:946; its `data_ptr()` is 0, so aligned). The launcher (a small C module built
with the system C compiler, triton/backends/nvidia/driver.py:685) and the cubin load are left for
the first real launch (`CompiledKernel.run` -> `_init_handles`, triton/compiler/compiler.py:436-476);
`compile_kernel` does them in the thread too. Nothing runs on the GPU, so the warm-up cannot change
what training computes, and the harness does not time it (its `JITFunction.run` wrapper skips
calls with `warmup=True`; it still counts them as launches in `triton_launches`).

Adding a kernel
---------------
In your module's patch function (so that a patch switched off warms nothing up, and the harness
does not count a warm-up of a kernel that training never launches):

    from .warmup import compile_kernel, register, training_half_dtype

    def _warm_my_kernel():
        half = training_half_dtype()
        compile_kernel(my_kernel, half, torch.float32, 1536, 826, BLOCK=1024, num_warps=4)

    def patch_something():
        ...
        register(_warm_my_kernel, name="my_kernel")

once per class the training launches can take (see above), all classes of one kernel in one
registration (the workers run registrations side by side). A registration that raises is logged
and skipped; the others still run. `start_background_warmup()` reads the registry when it is
called, so `apply()` calls it after the patches.

The thread and the GIL
----------------------
The first compile of the process hashes triton's own files, libtriton.so (415 MB) in 1 MB reads
(`triton_key`, triton/runtime/cache.py:272-298), and the compile, the ptxas and C compiler
subprocesses and the cubin load drop and retake the GIL many times. While the main thread runs
Python, each retake waits up to the switch interval (5 ms by default), so the thread lowers it to
1 ms while it runs and puts it back after (see `_run` for the numbers).

Where to start it
-----------------
At the end of apply(), with `patch_soup_warmup()` (the wait at `train` begin) and
`watch_compiles()`. RTX 3050 Ti, 3 workers, medians (bench_warmup.py load / apply): the real
model load takes 2.77 s alone and 2.96 s with the warm-up next to it, which is done 3.27 s after
the load starts; started before apply() instead, it makes apply() 1.28 s longer (3.05 -> 4.33 s),
since apply() only imports and leaves the GIL no gaps. Background imports do not pay either:
trl and datasets imported next to the model load end at 3.38 s, imported first then the load at
2.69 s. The log line "waited N s for the warmup" at `train` begin shows whether the warm-up fits.
"""

import atexit
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SWITCH_INTERVAL = 0.001  # seconds: the GIL switch interval while the warm-up runs (see _run)
HELPER_TIMEOUT = 120.0  # seconds the warm-up thread waits for the helper process before compiling itself
EXIT_TIMEOUT = 60.0  # seconds the interpreter's exit waits for the warm-up thread (see _shutdown)
# threads that take the registrations in turn (see _run). Whole warm-up on the RTX 3050 Ti with
# the main thread waiting / spinning a counter: 1 worker 4.8 / 6.6 s, 2 workers 3.2 / 3.7 s,
# 3 workers 2.5 / 3.4 s, 4 workers 2.5 / 3.1 s (bench_warmup.py workers); the main thread loses
# about 1 s of its time in every case. One core is left to the main thread.
WORKERS = max(1, min(3, (os.cpu_count() or 2) - 1))
VOCAB = 151936  # Qwen2.5's vocabulary: V and stride_row of the fused head (divisible by 16)
INTERMEDIATE = 8960  # Qwen2.5-1.5B's MLP width: 16 * 560
TOKENS = 826  # a typical packed batch; only its class matters (tokens * 8960 is always "D")

WARMUPS = []  # [(name, fn)]: run in this order by the background thread
NEEDS_GPU = set()  # names of registrations that launch on the GPU: never run in the helper process
COMPILES = []  # every compile seen by `watch_compiles`, as dicts (see there)

_LOCK = threading.Lock()
# set in the helper process only (see helper_main): the half dtype of the training process, and
# that compile_kernel has no GPU to load the cubin on
_HELPER = {"half": None, "offline": False}


def _new_state():
    return {"thread": None, "idents": set(), "begin": None, "seconds": None, "results": {},
            "errors": {}, "helper": None}


_STATE = _new_state()
_DONE = threading.Event()
_STOP = threading.Event()  # set at exit: the warm-up takes no new registration (see _shutdown)
_EXIT = {"registered": False}
_HOOKS = {"installed": False}


def register(fn=None, *, name=None, gpu=False):
    """Adds `fn` (no arguments) to WARMUPS; usable as `@register` or `@register(name=...)`. A second
    registration under the same name replaces the first (a reloaded module). `gpu=True` marks a
    warm-up that launches work on the GPU (it runs only in the training process, never in the
    helper)."""

    def add(f):
        key = name or f"{f.__module__}.{f.__qualname__}"
        if gpu:
            NEEDS_GPU.add(key)
        for i, (existing, _) in enumerate(WARMUPS):
            if existing == key:
                WARMUPS[i] = (key, f)
                return f
        WARMUPS.append((key, f))
        return f

    return add(fn) if fn is not None else add


def training_half_dtype():
    """The half dtype the run computes in: Soup's `bf16_fp16_flags` gives bf16 where the card has it
    and fp16 elsewhere (soup_cli/trainer/sft.py:1511-1512, soup_cli/utils/gpu.py:564), so fp16 on
    the T4. In the helper process (no GPU), the training process's answer."""
    import torch

    if _HELPER["half"] is not None:
        return _HELPER["half"]
    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # an older torch without the keyword
        bf16 = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16 else torch.float16


def compile_kernel(kernel, *args, **kwargs):
    """Compiles `kernel` (a @triton.jit function) for these arguments without launching it, then
    builds its launcher and loads the cubin on the current device. Pass a torch dtype for each
    tensor, and the ints, floats, constexprs and num_warps / num_stages as the training launch
    passes them (or values of the same class). Returns triton's CompiledKernel.

    In the helper process there is no GPU: the launcher is built (and cached on disk) but no cubin
    is loaded."""
    compiled = kernel.warmup(*args, grid=(1,), **kwargs)
    if compiled is not None and hasattr(compiled, "result"):  # triton's async compile mode
        compiled = compiled.result()
    if _HELPER["offline"]:
        if compiled is not None:
            from triton.runtime.driver import driver

            # what _init_handles does first (triton/compiler/compiler.py:452): the launcher's C
            # build, cached under the hash of its source (triton/runtime/build.py:compile_module_from_src)
            driver.active.launcher_cls(compiled.src, compiled.metadata)
        return compiled
    init = getattr(compiled, "_init_handles", None)
    if callable(init):
        init()
    return compiled


# The kernels in solution/ (their modules are not touched; the warm-ups live here), in the order
# the first step meets them. Each warms the classes its training launches take (checked against a
# real step in test_warmup.py) and skips itself when apply() leaves its patch off: the switches
# are read the way solution/__init__.py reads them, so the thread may start before the patches.
# The launches: fused_ce.py `_head_chunks`, swiglu.py `SwiGLU`, rmsnorm.py `_forward` /
# `_backward`, nf4.py `_launch`, rope.py `_launch`.

SKIPPED = "off"  # the result of a warm-up whose patch is switched off
HIDDEN = 1536  # Qwen2.5-1.5B: hidden size, 12 query heads and 2 key/value heads of 128
HEAD_DIM, Q_HEADS, KV_HEADS = 128, 12, 2


def _switched_on(name):
    from . import _on

    return _on(name)


@register(name="cublas", gpu=True)
def warm_cublas():
    # not Triton: the first half matmul of the process loads and sets up cuBLAS / cuBLASLt
    # (0.27 s on the RTX 3050 Ti, then 0.012 s for the main thread's first matmul: the library
    # state is per process, only the handle is per thread). Real launches on 64 x 64 garbage, on
    # the thread's stream; training reads none of it.
    import torch
    import torch.nn.functional as F

    half = training_half_dtype()
    x = torch.randn(64, 64, device="cuda", dtype=half)
    F.linear(x, x, x[0]).sum().item()  # addmm with a bias: the cuBLASLt path
    (x @ x).sum().item()


@register(name="rms_norm_rows_fwd")
def warm_rmsnorm_fwd():
    # rows of the fp32 residual stream (strides = 1536, "D"; n_rows is not specialized): the plain
    # norm (input_layernorm, the final norm) and the fused add with the half attention output
    if not _switched_on("SOUP_SOLUTION_RMSNORM"):
        return SKIPPED
    import torch
    import triton

    from .rmsnorm import FWD_ROWS, FWD_WARPS, rms_norm_rows_fwd

    f32, block = torch.float32, triton.next_power_of_2(HIDDEN)
    for add, add_dtype in ((False, f32), (True, training_half_dtype())):
        compile_kernel(rms_norm_rows_fwd, f32, HIDDEN, add_dtype, HIDDEN, f32, HIDDEN, f32,
                       f32, HIDDEN, f32, TOKENS, HIDDEN, 1e-6,
                       ADD=add, ROWS=FWD_ROWS, BLOCK=block, num_warps=FWD_WARPS)


@register(name="nf4_dequant_kernel")
def warm_nf4():
    # every 4-bit weight of the model has N * K and N * K / 64 divisible by 16: one class
    if not _switched_on("SOUP_SOLUTION_NF4"):
        return SKIPPED
    import torch

    from .nf4 import ABLOCK, BLOCKS_PER_PROGRAM, QBLOCK, nf4_dequant_kernel

    n = HIDDEN * HIDDEN
    f32, u8 = torch.float32, torch.uint8
    # the kernel reads the packed weight as int32 words; its arguments: words, qabsmax, code2,
    # absmax2, offset, nf4 table, out, n_blocks (see nf4.py's launch)
    compile_kernel(nf4_dequant_kernel, torch.int32, u8, f32, f32, f32, f32, torch.float16,
                   n // QBLOCK, QBLOCK=QBLOCK, ABLOCK=ABLOCK, BLOCKS=BLOCKS_PER_PROGRAM,
                   num_warps=4, enable_fp_fusion=False)


def _rope(backward, dtype):
    # q (1, 12, N, 128) and k (1, 2, N, 128) are transposed views of the projections: strides
    # (N * 1536, 128, 1536) and (N * 256, 128, 256), all multiples of 16 for any N; the outputs keep
    # them; cos and sin fp32 (1, N, 128) with the batch stride passed as 0. The backward's gradients
    # come contiguous, (12 * N * 128, N * 128, 128): also all "D". n_tokens is not specialized.
    # fused_proj.py launches it on views of its half qkv buffer, (N * 2048, 128, 2048) and the like:
    # the same classes, with q, k and the outputs in the half dtype (`dtype`); rope.py's own
    # attention path in fp32.
    import torch

    from .rope import BLOCK_TOKENS, NUM_WARPS, rope_qk_kernel

    f32, n = torch.float32, TOKENS
    if backward:
        q_in = (n * Q_HEADS * HEAD_DIM, n * HEAD_DIM, HEAD_DIM)
        k_in = (n * KV_HEADS * HEAD_DIM, n * HEAD_DIM, HEAD_DIM)
    else:
        q_in = (n * HIDDEN, HEAD_DIM, HIDDEN)
        k_in = (n * KV_HEADS * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM)
    q_out = (n * HIDDEN, HEAD_DIM, HIDDEN)
    k_out = (n * KV_HEADS * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM)
    compile_kernel(rope_qk_kernel, dtype, dtype, dtype, dtype, f32, f32,
                   *q_in, *k_in, *q_out, *k_out,
                   0, HEAD_DIM, n, Q_HEADS,
                   HALF=HEAD_DIM // 2, BLOCK_N=BLOCK_TOKENS, BACKWARD=backward,
                   num_warps=NUM_WARPS)


@register(name="rope_qk_kernel")
def warm_rope():
    if not _switched_on("SOUP_SOLUTION_ROPE"):
        return SKIPPED
    import torch

    dtypes = [torch.float32]
    if _switched_on("SOUP_SOLUTION_FUSED_PROJ"):
        dtypes.insert(0, training_half_dtype())  # fused_proj.py's launches, first
    for dtype in dtypes:
        _rope(backward=False, dtype=dtype)
        _rope(backward=True, dtype=dtype)


@register(name="example_swiglu_fwd_kernel")
def warm_swiglu_fwd():
    # n = tokens * 8960 is a multiple of 16 for every batch, so only the "D" class occurs
    from .swiglu import BLOCK, example_swiglu_fwd_kernel

    half = training_half_dtype()
    compile_kernel(example_swiglu_fwd_kernel, half, half, half, TOKENS * INTERMEDIATE, BLOCK=BLOCK)


@register(name="fused_ce_rows_kernel")
def warm_fused_ce():
    # logits half, stride_row = V = 151936 ("D"), targets int64, loss / entropy fp32, correct int32
    # (all chunk slices start at a multiple of the 512-row chunk: aligned), grad_scale float.
    # n_grad_rows: 387 stands for a count that is not a multiple of 16 (most chunks), 512 for one
    # that is (0 too: a chunk without labelled rows), 1 is a constant of its own
    import torch

    from .fused_ce import BLOCK_V, IGNORE_INDEX, fused_ce_rows_kernel

    half = training_half_dtype()
    # 1 last: a chunk that ends one labelled row past its start is rare
    for n_grad in (387, 512, 1):
        compile_kernel(fused_ce_rows_kernel, half, VOCAB, torch.int64, torch.float32,
                       torch.float32, torch.int32, 1.0, n_grad, VOCAB,
                       IGNORE=IGNORE_INDEX, BLOCK=BLOCK_V, num_warps=8)


@register(name="example_swiglu_bwd_kernel")
def warm_swiglu_bwd():
    from .swiglu import BLOCK, example_swiglu_bwd_kernel

    half = training_half_dtype()
    compile_kernel(example_swiglu_bwd_kernel, half, half, half, half, half,
                   TOKENS * INTERMEDIATE, BLOCK=BLOCK)


@register(name="rms_norm_rows_bwd")
def warm_rmsnorm_bwd():
    # frozen norm weights: no dweight, so the partial-sum buffer is rstd and tiles_per_program is 1
    # (a constant). The fused add's backward adds the residual's gradient and writes dx again in
    # the half dtype of the attention output
    if not _switched_on("SOUP_SOLUTION_RMSNORM"):
        return SKIPPED
    import torch
    import triton

    from .rmsnorm import BWD_ROWS, BWD_WARPS, rms_norm_rows_bwd

    f32, block = torch.float32, triton.next_power_of_2(HIDDEN)
    for fused, dx2_dtype in ((False, f32), (True, training_half_dtype())):
        compile_kernel(rms_norm_rows_bwd, f32, HIDDEN, f32, HIDDEN, f32, f32, f32, HIDDEN,
                       f32, HIDDEN, dx2_dtype, HIDDEN, f32, TOKENS, HIDDEN, 1,
                       ADD_GRAD=fused, SECOND=fused, DW=False, ROWS=BWD_ROWS, BLOCK=block,
                       num_warps=BWD_WARPS)


def _triton_key():
    """The hash of triton's own files (`triton_key`, triton/runtime/cache.py:272-298), part of every
    on-disk cache key: computed once per process, mostly outside the GIL (file reads, sha256)."""
    try:
        from triton.runtime.cache import triton_key

        triton_key()
    except Exception:  # noqa: BLE001 - another triton layout; the first compile does it then
        pass


def _prepare_triton():
    """What the first compile of the process does once, done before the workers start so that they
    do not all do it at the same time: the CUDA driver object and its cuda_utils module (built with
    the C compiler, triton/backends/nvidia/driver.py:61-64, or loaded from the on-disk cache) and
    `triton_key`."""
    from triton.runtime.driver import driver

    driver.active.get_current_device()
    _triton_key()


def _run_jobs(pending, workers, on_start=None, stop=None):
    """Runs the registrations in `pending` ([(name, fn)]) on `workers` threads (the caller's one
    included) and returns ({name: seconds or SKIPPED}, {name: error}). One registration per kernel,
    so two workers never fill the same kernel's cache. Once `stop` is set, no worker takes another
    registration."""
    pending = list(pending)
    results, errors = {}, {}
    take = threading.Lock()

    def worker():
        if on_start is not None:
            on_start()
        while True:
            with take:
                if not pending or (stop is not None and stop.is_set()):
                    return
                name, fn = pending.pop(0)
            start = time.perf_counter()
            try:
                skipped = fn() == SKIPPED
                results[name] = SKIPPED if skipped else round(time.perf_counter() - start, 3)
            except Exception as exc:  # noqa: BLE001 - a warm-up must never break training
                errors[name] = repr(exc)

    helpers = [threading.Thread(target=worker, name=f"solution-warmup-{i}", daemon=True)
               for i in range(1, min(workers, len(pending)))]
    for helper in helpers:
        helper.start()
    worker()
    for helper in helpers:
        helper.join()
    return results, errors


def _shutdown():
    """At the interpreter's exit: the warm-up takes no new registration, the helper is killed, and
    the exit waits for the thread to finish the registration it is in (at most EXIT_TIMEOUT s).
    Without it, a process that ends soon after apply() (Soup's tests end 0.6 s after it) could
    finalize while the thread compiles, and the compiler's C++ thread pool, destroyed under it,
    aborts the process ("terminate called without an active exception", exit code 134; T13 saw it
    in 3 of 6 runs of the tests). Waiting for the current registration only keeps the tests' exit
    short (about a second at most), where waiting for the whole warm-up would add its 5 s. The
    killed helper leaves no half-written entry in triton's cache: every file is written in a temp
    directory and moved into place with os.replace, and a kernel's index file (`__grp__*.json`)
    last (triton/runtime/cache.py, FileCacheManager.put / put_group); a kernel cut short has no
    index and is compiled again by whoever needs it. In a training run the warm-up is long done at
    exit: nothing to wait for."""
    _STOP.set()
    helper = _STATE["helper"]
    if helper is not None and helper["proc"].poll() is None:
        helper["proc"].kill()
    thread = _STATE["thread"]
    if thread is not None and thread is not threading.current_thread():
        thread.join(EXIT_TIMEOUT)


def _register_shutdown():
    if not _EXIT["registered"]:
        _EXIT["registered"] = True
        atexit.register(_shutdown)


# ---------------------------------------------------------------- the helper process

def start_compile_helper(log=True):
    """Starts the helper process that compiles every registration not marked `gpu` (those
    registered so far: the ones in this file) into triton's on-disk cache, and returns it (None
    without CUDA, or when it is already started). The warm-up thread waits for it, then runs every
    registration in this process, where the kernels the helper compiled are loaded from the cache.

    The helper sees no GPU (CUDA_VISIBLE_DEVICES is empty): it cannot create a CUDA context, which
    would hold GPU memory that nvidia-smi counts. Triton compiles for a target, not a device, so
    the helper gets the target of this process's device (see helper_main). Its output goes to a
    pipe that the warm-up thread reads; it is killed at exit if still running (see _shutdown)."""
    with _LOCK:
        if _STATE["helper"] is not None:
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            major, minor = torch.cuda.get_device_capability()
            half = str(training_half_dtype()).replace("torch.", "")
            names = [name for name, _ in WARMUPS if name not in NEEDS_GPU]
            package_root = str(Path(__file__).resolve().parents[1])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join(
                           p for p in (package_root, os.environ.get("PYTHONPATH")) if p))
            proc = subprocess.Popen(
                [sys.executable, "-m", __name__, "--helper", str(major * 10 + minor), half, *names],
                env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)
        except Exception as exc:  # noqa: BLE001 - the in-process warm-up compiles everything then
            _STATE["errors"]["(helper start)"] = repr(exc)
            if log:
                print(f"[solution] warmup helper not started: {exc!r}", flush=True)
            return None
        _STATE["helper"] = {"proc": proc, "begin": time.perf_counter(), "names": names,
                            "report": None}
        _register_shutdown()
        return proc


def _wait_for_helper(log):
    """Waits for the helper (at most HELPER_TIMEOUT s) and returns its report, or None."""
    helper = _STATE["helper"]
    if helper is None:
        return None
    proc = helper["proc"]
    try:
        out, err = proc.communicate(timeout=HELPER_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - a hung helper: kill it, compile here
        proc.kill()
        proc.communicate()
        _STATE["errors"]["(helper)"] = repr(exc)
        if log:
            print(f"[solution] warmup helper stopped: {exc!r}", flush=True)
        return None
    report = None
    for line in reversed((out or "").splitlines()):
        if line.startswith("{"):
            try:
                report = json.loads(line)
            except ValueError:
                pass
            break
    if report is None or proc.returncode != 0:
        _STATE["errors"]["(helper)"] = f"exit {proc.returncode}: {(err or '').strip()[-500:]}"
        if log and not _STOP.is_set():
            print(f"[solution] warmup helper failed (exit {proc.returncode}): "
                  f"{(err or '').strip()[-500:]}", flush=True)
        return None
    report["wall"] = round(time.perf_counter() - helper["begin"], 3)
    helper["report"] = report
    if log:
        done = ", ".join(f"{name} {s}" if s == SKIPPED else f"{name} {s:.2f} s"
                         for name, s in report["results"].items())
        print(f"[solution] warmup helper (pid {proc.pid}, no GPU) done {report['wall']:.2f} s "
              f"after its start: imports {report['import_s']:.2f} s, compiles "
              f"{report['compile_s']:.2f} s: {done or 'nothing'}"
              + (f"; failed: {', '.join(report['errors'])}" if report["errors"] else ""),
              flush=True)
    return report


def _offline_driver(arch):
    """Triton's CUDA driver answering for a device this process does not open: device 0, stream 0
    and the training device's compute capability. Compiling needs nothing else from the device:
    the cache key and the code are those of the target (JITFunction.create_binder,
    triton/runtime/jit.py:658-669; CudaDriver.get_current_target, nvidia/driver.py:724-729)."""
    from triton.backends.nvidia.driver import CudaDriver
    from triton.runtime.driver import driver

    offline = CudaDriver()  # builds or loads cuda_utils; touches no device
    offline.get_current_device = lambda: 0
    offline.get_current_stream = lambda device=None: 0
    offline.get_device_capability = lambda device=None: divmod(arch, 10)
    driver.set_active(offline)
    return offline


def helper_main(argv):
    """The helper process: `python -m solution.warmup --helper <arch> <half dtype> <names...>`.
    Prints one JSON line: seconds and errors per registration."""
    begin = time.perf_counter()
    arch, half, names = int(argv[1]), argv[2], set(argv[3:])
    import torch

    _HELPER["half"] = getattr(torch, half)
    _HELPER["offline"] = True
    _offline_driver(arch)
    _triton_key()
    import_s = time.perf_counter() - begin
    jobs = [(name, fn) for name, fn in WARMUPS if name in names and name not in NEEDS_GPU]
    results, errors = _run_jobs(jobs, WORKERS)
    print(json.dumps({"import_s": round(import_s, 3),
                      "compile_s": round(time.perf_counter() - begin - import_s, 3),
                      "results": results, "errors": errors,
                      "cuda_initialized": torch.cuda.is_initialized()}), flush=True)
    return 0


# ---------------------------------------------------------------- the warm-up thread

def _run(jobs, device, log):
    begin = time.perf_counter()
    previous = ours = None
    try:
        import torch

        _STATE["idents"].add(threading.get_ident())
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        # while the helper compiles: triton's own hash (mostly outside the GIL), then the wait
        _triton_key()
        helper = _wait_for_helper(log)
        if _STOP.is_set():  # the process is exiting (see _shutdown)
            return
        # The rest takes the GIL often (the compiles the helper did not do, and for the others the
        # cache reads, the launcher import and the cubin load); while the main thread runs Python,
        # every retake waits for the switch interval (5 ms by default). RTX 3050 Ti, one worker,
        # compiling, medians of 3 (bench_warmup.py gil, before rmsnorm / nf4 / rope): 3.1 s with
        # the main thread waiting; with it spinning a Python counter, 15.3 s at 5 ms (counter at
        # 100% of its rate alone), 5.5 s at 1 ms (89%), 4.0 s at 0.2 ms (79%). 1 ms keeps the
        # warm-up short and costs the main thread at most about 11% while it runs.
        previous = sys.getswitchinterval()
        sys.setswitchinterval(min(previous, SWITCH_INTERVAL))
        ours = sys.getswitchinterval()  # read back: CPython stores it in microseconds
        _prepare_triton()

        def on_start():
            # several workers overlap the parts outside the GIL (a compile's ptxas and C compiler
            # runs, the cubin loads); triton compiles from several threads by design (its
            # AsyncCompileMode takes any Executor, triton/runtime/_async_compile.py)
            _STATE["idents"].add(threading.get_ident())
            torch.cuda.set_device(device)
            torch.cuda.set_stream(stream)  # a registration that launches stays off the main stream

        # with the helper's kernels in the cache, loading them needs no extra workers
        results, errors = _run_jobs(jobs, 1 if helper is not None else WORKERS, on_start, _STOP)
        _STATE["results"].update(results)
        _STATE["errors"].update(errors)
        if log:
            for name, error in errors.items():
                print(f"[solution] warmup {name} failed: {error}", flush=True)
        stream.synchronize()
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"]["(thread)"] = repr(exc)
        if log:
            print(f"[solution] warmup thread failed: {exc!r}", flush=True)
    finally:
        if ours is not None and sys.getswitchinterval() == ours:  # unless changed meanwhile
            sys.setswitchinterval(previous)
        _STATE["seconds"] = round(time.perf_counter() - begin, 3)
        _DONE.set()
        if log and not _STOP.is_set():
            done = ", ".join(f"{name} {s}" if s == SKIPPED else f"{name} {s:.2f} s"
                             for name, s in _STATE["results"].items())
            compiled = [c for c in COMPILES if c["thread"] in _STATE["idents"]]
            loaded = sum(1 for c in compiled if c.get("cache_hit"))
            print(f"[solution] warmup done in {_STATE['seconds']:.2f} s"
                  + (f" ({loaded} kernels loaded from the on-disk cache, {len(compiled) - loaded} "
                     f"compiled here)" if _HOOKS["installed"] else "")
                  + f": {done or 'nothing'}"
                  + (f"; failed: {', '.join(_STATE['errors'])}" if _STATE["errors"] else ""),
                  flush=True)


def start_background_warmup(log=True):
    """Starts the daemon thread that runs every registered warm-up once (with up to WORKERS - 1
    daemon helpers), on the caller's CUDA device and on a stream of its own. Returns the thread
    (the same one on later calls), or None without CUDA. The registry is read now: register before
    calling this."""
    with _LOCK:
        if _STATE["thread"] is not None:
            return _STATE["thread"]
        try:
            import torch

            if not torch.cuda.is_available():
                _DONE.set()
                return None
            device = torch.cuda.current_device()
        except Exception as exc:  # noqa: BLE001
            _STATE["errors"]["(start)"] = repr(exc)
            _DONE.set()
            return None
        _STATE["begin"] = time.perf_counter()
        thread = threading.Thread(target=_run, args=(list(WARMUPS), device, log),
                                  name="solution-warmup", daemon=True)
        _STATE["thread"] = thread
        _register_shutdown()
        thread.start()
        return thread


def wait_for_warmup(timeout=None):
    """Blocks until the warm-up thread is done (True) or `timeout` seconds pass (False). False at
    once when it was never started."""
    if _STATE["thread"] is None:
        return _DONE.is_set()
    return _DONE.wait(timeout)


def warmup_report():
    """Diagnostics: per-registration seconds, errors, total seconds, and the compiles seen by
    `watch_compiles` inside and outside the warm-up threads."""
    idents = _STATE["idents"]
    return {
        "started": _STATE["thread"] is not None,
        "done": _DONE.is_set(),
        "seconds": _STATE["seconds"],
        "results": dict(_STATE["results"]),
        "errors": dict(_STATE["errors"]),
        "compiles_in_warmup": sum(1 for c in COMPILES if c["thread"] in idents),
        "compiles_outside": [c for c in COMPILES if c["thread"] not in idents],
        "helper": (_STATE["helper"] or {}).get("report"),
    }


def _describe(signature, constants, attrs):
    """`n_grad_rows:i32 D, IGNORE=-100, ...`: the parts of the key that differ between compiles."""
    parts = []
    for i, (param, ty) in enumerate(signature.items()):
        if ty == "constexpr":
            parts.append(f"{param}={constants.get((i,))}")
        else:
            # attrs has an entry for every specialized argument, [] when it is not divisible
            divisible = any(attr[0] == "tt.divisibility" for attr in attrs.get((i,)) or [])
            parts.append(f"{param}:{ty}{' D' if divisible else ''}")
    return ", ".join(parts)


def watch_compiles(log=True):
    """Records every Triton compile in COMPILES through triton's JIT hooks
    (`knobs.runtime.jit_cache_hook` before and `jit_post_compile_hook` after, triton/knobs.py:474-477;
    `JITFunction._do_compile` calls them only on an in-memory cache miss, jit.py:826-851, so a
    cached launch costs nothing extra) and its compilation listener (`knobs.compilation.listener`,
    called by `triton.compile` with `cache_hit`, compiler.py:268-276, 359-361: whether the miss was
    served by the on-disk cache, as for the kernels the helper process compiled). Each record:
    kernel, thread id, seconds (the compile to cubin or the cache read; the launcher comes on the
    first launch), cache_hit, the key's specialization, num_warps, num_stages. With `log`, a miss
    outside the warm-up threads is printed: after a complete warm-up the training steps should
    print none. Hooks already set are kept and called first."""
    from triton import knobs

    with _LOCK:
        if _HOOKS["installed"]:
            return COMPILES
        before_prev = knobs.runtime.jit_cache_hook
        after_prev = knobs.runtime.jit_post_compile_hook
        listener_prev = knobs.compilation.listener
        local = threading.local()

        def before(**kwargs):
            local.start = time.perf_counter()
            local.cache_hit = None
            # a true return skips the compile (jit.py:829): only the previous hook may decide that
            return before_prev(**kwargs) if before_prev is not None else None

        def after(**kwargs):
            try:
                info = kwargs.get("compile") or {}
                configs = info.get("configs") or [{}]
                record = {
                    "kernel": getattr(kwargs.get("fn"), "name", "?"),
                    "thread": threading.get_ident(),
                    "seconds": round(time.perf_counter() - getattr(local, "start", time.perf_counter()), 3),
                    "spec": _describe(info.get("signature") or {}, info.get("constants") or {},
                                      configs[0] or {}),
                    "num_warps": info.get("num_warps"),
                    "num_stages": info.get("num_stages"),
                    "is_warmup": bool(kwargs.get("is_manual_warmup")),
                    "cache_hit": getattr(local, "cache_hit", None),
                }
                COMPILES.append(record)
                if log and record["thread"] not in _STATE["idents"]:
                    what = ("kernel loaded from the on-disk cache" if record["cache_hit"]
                            else "compile")
                    print(f"[solution] triton {what} outside warmup: {record['kernel']} "
                          f"{record['seconds']:.2f} s ({record['spec']}; num_warps="
                          f"{record['num_warps']})", flush=True)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            return after_prev(**kwargs) if after_prev is not None else None

        def listener(**kwargs):
            local.cache_hit = bool(kwargs.get("cache_hit"))
            return listener_prev(**kwargs) if listener_prev is not None else None

        knobs.runtime.jit_cache_hook = before
        knobs.runtime.jit_post_compile_hook = after
        knobs.compilation.listener = listener
        _HOOKS["installed"] = True
    return COMPILES


def preload_in_background(modules, log=True):
    """Imports `modules` (names) one after another in a daemon thread. Returns a threading.Event set
    once all are done (a failed import is logged and skipped); `event.errors` maps names to errors.

    A patch that needs one of them can wait on the event, or simply import it: Python's import lock
    is per module, so an import of a module the thread is still importing waits for that import to
    finish instead of running it a second time. The thread holds the GIL while it runs module code,
    so this overlaps imports only with the main thread's waits (file reads, CUDA, subprocesses).

    Not used by default: `import triton` costs 0.07 s after the harness's prelude (RTX 3050 Ti box,
    bench_warmup.py imports); what makes apply() long are trl, peft (which pulls bitsandbytes and
    torch._dynamo) and transformers' trainer modules, and Soup imports those itself in
    `_setup_transformers` and `train` (soup_cli/trainer/sft.py:1613, :734, :1594)."""
    ready = threading.Event()
    ready.errors = {}
    ready.seconds = {}

    def run():
        for name in modules:
            start = time.perf_counter()
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001
                ready.errors[name] = repr(exc)
                if log:
                    print(f"[solution] preload {name} failed: {exc!r}", flush=True)
            ready.seconds[name] = round(time.perf_counter() - start, 3)
        ready.set()

    threading.Thread(target=run, name="solution-preload", daemon=True).start()
    return ready


def patch_soup_warmup(start_in_setup=False):
    """Soup hooks for the warm-up: with `start_in_setup`, start it when
    SFTTrainerWrapper._setup_transformers begins (instead of from apply()); in every case, when
    SFTTrainerWrapper.train begins, wait for it (at most 60 s) so the first step does not compile a
    kernel the thread is still compiling. Waiting costs nothing once the thread is done."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_warmup", False):
        return
    original_setup = cls._setup_transformers
    original_train = cls.train

    def _setup_transformers(self, *args, **kwargs):
        if start_in_setup:
            start_background_warmup()
        return original_setup(self, *args, **kwargs)

    def train(self, *args, **kwargs):
        if _STATE["thread"] is not None and not _DONE.is_set():
            start = time.perf_counter()
            finished = wait_for_warmup(60.0)
            print(f"[solution] waited {time.perf_counter() - start:.2f} s for the warmup"
                  + ("" if finished else " (not finished)"), flush=True)
        return original_train(self, *args, **kwargs)

    cls._setup_transformers = _setup_transformers
    cls.train = train
    cls._soup_warmup = True


if __name__ == "__main__":
    if sys.argv[1:2] != ["--helper"]:
        sys.exit("usage: python -m solution.warmup --helper <arch> <half dtype> <warm-up names...>")
    # the importable module, not this __main__ copy: its registry is the one the warm-ups fill
    from solution.warmup import helper_main

    sys.exit(helper_main(sys.argv[1:]))
