"""Triton compilation of the run's kernels in a background thread of the training process, while
Soup loads the model.

Every measured run starts with empty compiler caches, so the first training step compiles every
kernel it meets (Kaggle T4, commit d649c7c: the first two steps took about 4.5 s against 0.9 s for
two ordinary ones; from the end of apply() to the first step there are about 6 s: the model load,
3.1 s, and the trainer setup, 2.9 s). Here they are compiled ahead, so that the first step finds
them in Triton's in-memory cache: the variants a real step compiles (rms_norm_rows_fwd and _bwd two
each, nf4_dequant_kernel, rope_qk_kernel forward and backward (half for fused_proj.py, fp32 for
rope.py's own path), the example SwiGLU forward and backward, the kernels other modules register:
fused_proj's rms_norm_rows_rebuild, head_stream's chunk and finish kernels; with head_stream off,
fused_ce_rows_kernel two) and a rarer third one of
fused_ce_rows_kernel.

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
`compile_kernel` does them ahead too. Nothing runs on the GPU, so the warm-up cannot change
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
called, so `apply()` calls it after the patches; an earlier call (`start_early`, with
SOUP_SOLUTION_WARMUP_START=apply_begin) starts on this file's registrations and the final call
queues the ones made since.

The thread, and what it costs the main thread
---------------------------------------------
Everything runs in this process. The compile, the
ptxas and C compiler subprocesses and the cubin loads drop and retake the GIL many times, so the
thread lowers the switch interval to 1 ms while it runs (see `_run`). Its largest single cost next
to the model load was triton's own hash, 415 MB of libtriton.so read and digested (`_triton_key`):
`start_early`, first thing in apply(), runs it in a thread of its own, next to the imports. At
exit, `_shutdown` stops the warm-up between two registrations. `watch_compiles()` logs any compile
(or on-disk cache read) outside the warm-up; the log line "waited N s for the warmup" at `train`
begin shows whether the warm-up fits.

Measuring switches (environment): SOUP_SOLUTION_WARMUP_START=apply_begin|apply_end (default),
SOUP_SOLUTION_WARMUP_WORKERS, SOUP_SOLUTION_WARMUP_SWITCH_MS, SOUP_SOLUTION_WARMUP_SKIP=name,name.

The launchers (SOUP_SOLUTION_LAUNCHER_OPT, SOUP_SOLUTION_LAUNCHER_NICE; off by default)
------------------------------------------------------------------------------------
Each kernel's launcher is a small C module that triton generates for the kernel's signature and builds
with the system C compiler, `cc <src> -O3 -shared -fPIC ...` (triton/runtime/build.py `_build`), on
an empty cache once per launcher, plus once for its cuda_utils module. On the T4 (startup_profile.py,
night-trinit session 1) the warm-up threads spent 5.3 of their 11.5 thread-seconds during Soup's model
load waiting for these compiler processes, about 0.3 s of CPU each, on a 4-vCPU machine where the
main thread's model load then got 24% less CPU work done per second. Optimization barely matters for
a launcher (it parses the arguments with the Python C API and calls cuLaunchKernelEx), and gcc spends
most of -O3's time optimizing: on the RTX 3050 Ti box a launcher takes 0.30 s at -O3, 0.21 s at -O1,
0.13 s at -O0 (preprocessing alone 0.06 s). SOUP_SOLUTION_LAUNCHER_OPT=<0|1|2|s> builds them with
that level through triton's own hook for it (`triton.knobs.build.impl`), with the same compiler,
sources, flags and libraries otherwise; SOUP_SOLUTION_LAUNCHER_NICE=<n> runs the compiler under
`nice -n <n>`, so the main thread gets a CPU first when the machine is full. The kernels themselves
(the cubins from ptxas) do not change.
"""

import atexit
import importlib
import os
import re
import queue
import re
import sys
import threading
import time

SWITCH_INTERVAL = 0.001  # seconds: the GIL switch interval while the warm-up runs (see _run)
EXIT_TIMEOUT = 60.0  # seconds the interpreter's exit waits for the warm-up thread (see _shutdown)
# threads that take the registrations in turn (see _run). Whole warm-up on the RTX 3050 Ti with
# the main thread waiting / spinning a counter: 1 worker 4.8 / 6.6 s, 2 workers 3.2 / 3.7 s,
# 3 workers 2.5 / 3.4 s, 4 workers 2.5 / 3.1 s (bench_warmup.py workers); the main thread loses
# about 1 s of its time in every case. One core is left to the main thread.
# 2 workers by default: T15c's variants session on the T4 (soup-t15, 3f5b14c, means of 2 rounds)
# put the first step at +9.73 s from apply begin with 2 against +10.46 s with 3 (4 vCPUs: the third
# worker competes with the main thread's model load)
WORKERS = max(1, min(2, (os.cpu_count() or 2) - 1))
VOCAB = 151936  # Qwen2.5's vocabulary: V and stride_row of the fused head (divisible by 16)
INTERMEDIATE = 8960  # Qwen2.5-1.5B's MLP width: 16 * 560
TOKENS = 826  # a typical packed batch; only its class matters (tokens * 8960 is always "D")

WARMUPS = []  # [(name, fn)]: run in this order by the background thread
COMPILES = []  # every compile seen by `watch_compiles`, as dicts (see there)

_LOCK = threading.Lock()


def _new_state():
    return {"thread": None, "idents": set(), "begin": None, "seconds": None, "results": {},
            "errors": {}, "queue": queue.Queue(), "queued": set(), "closed": False}


_STATE = _new_state()
_DONE = threading.Event()
_STOP = threading.Event()  # set at exit: the warm-up takes no new registration (see _shutdown)
_EXIT = {"registered": False}
_HOOKS = {"installed": False}


def register(fn=None, *, name=None):
    """Adds `fn` (no arguments) to WARMUPS; usable as `@register` or `@register(name=...)`. A second
    registration under the same name replaces the first (a reloaded module)."""

    def add(f):
        key = name or f"{f.__module__}.{f.__qualname__}"
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
    the T4."""
    import torch

    try:
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # an older torch without the keyword
        bf16 = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if bf16 else torch.float16


def compile_kernel(kernel, *args, **kwargs):
    """Compiles `kernel` (a @triton.jit function) for these arguments without launching it, then
    builds its launcher and loads the cubin on the current device. Pass a torch dtype for each
    tensor, and the ints, floats, constexprs and num_warps / num_stages as the training launch
    passes them (or values of the same class). Returns triton's CompiledKernel."""
    compiled = kernel.warmup(*args, grid=(1,), **kwargs)
    if compiled is not None and hasattr(compiled, "result"):  # triton's async compile mode
        compiled = compiled.result()
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


@register(name="cublas")
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


@register(name="step_ops")
def warm_step_ops():
    # not Triton either: the end of a training step (fp16 loss scaling, gradient clipping at 1.0,
    # AdamW with weight decay over the fp32 LoRA parameters, all foreach, as the run's
    # `adamw_torch` / max_grad_norm 1.0 / fp16 set it up) loads its CUDA kernels on first use: in the
    # first step on Kaggle AdamW's step took 0.13 s and clip_grad_norm_ 0.10 s (startup.py's profile,
    # ba8c1b3). The same ops here on two small fp32 parameters with an optimizer and a scaler of their
    # own, plus the first backward (the autograd engine's device thread starts with it): nothing of
    # the run is touched and no random number is drawn.
    import torch

    params = [torch.zeros(16, 64, device="cuda", requires_grad=True),
              torch.zeros(64, 16, device="cuda", requires_grad=True)]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    loss = params[0].sum() + params[1].sum()
    bool(torch.isnan(loss) | torch.isinf(loss))  # the trainer's check of every step's loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


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

    # fused_proj.py's attention launches it in half; rope.py's own path (fused_proj off, or its
    # plan declining) in fp32. Kaggle, 3344e16: 3588 launches = 28 layers x 64 steps x 2 + the
    # warm-up's own, so the fp32 variants never ran with fused_proj on
    if _switched_on("SOUP_SOLUTION_FUSED_PROJ"):
        dtypes = [training_half_dtype()]
    else:
        dtypes = [torch.float32]
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
    # with head_stream.py on, the head runs head_stream's kernels and this one never launches
    # (Kaggle, 3344e16: 3 launches in the run, the warm-up's own)
    if _switched_on("SOUP_SOLUTION_HEAD_STREAM"):
        return SKIPPED
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


_KEY_LOCK = threading.Lock()


def _triton_key():
    """The hash of triton's own files (`triton_key`, triton/runtime/cache.py:272-298), part of every
    on-disk cache key: computed once per process. It reads and digests 415 MB (libtriton.so),
    mostly outside the GIL (file reads, sha256): Kaggle, 2.25 s of hashing and 0.97 s of reads,
    which, next to the model load, slowed the load's copies to the GPU (startup.py's profile of
    2ed0117: the loader's `.to` took 3.45 s against 0.8 s without it). `start_early` therefore runs
    it from the start of apply(), next to the imports. The lock keeps a second caller from hashing
    again while the first one is at it."""
    with _KEY_LOCK:
        try:
            from triton.runtime.cache import triton_key

            triton_key()
        except Exception:  # noqa: BLE001 - another triton layout; the first compile does it then
            pass


def _prepare_triton():
    """What the first compile of the process does once, done before the workers start so that they
    do not all do it at the same time: the CUDA driver object and its cuda_utils module (built with
    the C compiler, triton/backends/nvidia/driver.py:61-64) and `triton_key`."""
    from triton.runtime.driver import driver

    driver.active.get_current_device()
    _triton_key()


def _setting(name, default, kind):
    """A positive number from the environment (for measuring the warm-up's settings), else
    `default`."""
    try:
        value = kind(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _shutdown():
    """At the interpreter's exit: the warm-up takes no new registration, and the exit waits for the
    thread to finish the registration it is in (at most EXIT_TIMEOUT s). Without it, a process that
    ends soon after apply() (Soup's tests end 0.6 s after it) could finalize while the thread
    compiles, and the compiler's C++ thread pool, destroyed under it, aborts the process
    ("terminate called without an active exception", exit code 134; T13 saw it in 3 of 6 runs of
    the tests). Waiting for the current registration only keeps the tests' exit short, where
    waiting for the whole warm-up would add its seconds. In a training run the warm-up is long
    done at exit: nothing to wait for."""
    _STOP.set()
    thread = _STATE["thread"]
    if thread is not None and thread is not threading.current_thread():
        thread.join(EXIT_TIMEOUT)


def _register_shutdown():
    if not _EXIT["registered"]:
        _EXIT["registered"] = True
        atexit.register(_shutdown)


def _worker(device, stream, log):
    """Takes registrations from the queue until it is closed and empty (or the process exits). One
    registration per kernel, so two workers never fill the same kernel's cache."""
    import torch

    _STATE["idents"].add(threading.get_ident())
    torch.cuda.set_device(device)
    torch.cuda.set_stream(stream)  # a registration that launches stays off the main thread's stream
    jobs = _STATE["queue"]
    while not _STOP.is_set():
        try:
            name, fn = jobs.get(timeout=0.05)
        except queue.Empty:
            if _STATE["closed"]:
                return
            continue
        start = time.perf_counter()
        try:
            skipped = fn() == SKIPPED
            _STATE["results"][name] = (SKIPPED if skipped
                                       else round(time.perf_counter() - start, 3))
        except Exception as exc:  # noqa: BLE001 - a warm-up must never break training
            _STATE["errors"][name] = repr(exc)
            if log:
                print(f"[solution] warmup {name} failed: {exc!r}", flush=True)


def _run(device, log, workers, switch):
    begin = time.perf_counter()
    previous = ours = None
    try:
        import torch

        _STATE["idents"].add(threading.get_ident())
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        # The compile drops and retakes the GIL many times; while the main thread runs Python, every
        # retake waits for the switch interval (5 ms by default). RTX 3050 Ti, one worker, medians of
        # 3 (bench_warmup.py gil, before rmsnorm / nf4 / rope): 3.1 s with the main thread waiting;
        # with it spinning a Python counter, 15.3 s at 5 ms (counter at 100% of its rate alone),
        # 5.5 s at 1 ms (89%), 4.0 s at 0.2 ms (79%). 1 ms keeps the warm-up short and costs the main
        # thread at most about 11% while it runs (SOUP_SOLUTION_WARMUP_SWITCH_MS sets it).
        previous = sys.getswitchinterval()
        sys.setswitchinterval(min(previous, switch))
        ours = sys.getswitchinterval()  # read back: CPython stores it in microseconds
        _prepare_triton()
        # several workers overlap the parts outside the GIL (ptxas and C compiler runs, cubin
        # loads); triton compiles from several threads by design (its AsyncCompileMode takes any
        # Executor, triton/runtime/_async_compile.py)
        helpers = [threading.Thread(target=_worker, args=(device, stream, log),
                                    name=f"solution-warmup-{i}", daemon=True)
                   for i in range(1, workers)]
        for helper in helpers:
            helper.start()
        _worker(device, stream, log)
        for helper in helpers:
            helper.join()
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
                  + (f" ({len(compiled) - loaded} kernels compiled, {loaded} read from the on-disk "
                     f"cache)" if _HOOKS["installed"] else "")
                  + f": {done or 'nothing'}"
                  + (f"; failed: {', '.join(_STATE['errors'])}" if _STATE["errors"] else ""),
                  flush=True)


def _launcher_build(level, nice):
    """triton/runtime/build.py `_build` (triton 3.6) with `-O<level>` instead of -O3 and the
    compiler under `nice -n <nice>` (see the module docstring)."""
    import shutil
    import subprocess
    import sysconfig

    from triton import knobs
    from triton.runtime.build import _library_flag

    nice_cmd = ["nice", "-n", str(nice)] if nice and shutil.which("nice") else []

    def build(name, src, srcdir, library_dirs, include_dirs, libraries):
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        so = os.path.join(srcdir, "{name}{suffix}".format(name=name, suffix=suffix))
        cc = os.environ.get("CC")
        if cc is None:
            clang = shutil.which("clang")
            gcc = shutil.which("gcc")
            cc = gcc if gcc is not None else clang
            if cc is None:
                raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable "
                                   "or set triton.knobs.build.impl.")
        scheme = sysconfig.get_default_scheme()
        if scheme == "posix_local":
            scheme = "posix_prefix"
        py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
        dirs = include_dirs + [srcdir, py_include_dir, *knobs.build.backend_dirs]
        cc_cmd = nice_cmd + [cc, src, f"-O{level}", "-shared", "-fPIC", "-Wno-psabi", "-o", so]
        cc_cmd += [_library_flag(lib) for lib in libraries]
        cc_cmd += [f"-L{d}" for d in library_dirs]
        cc_cmd += [f"-I{d}" for d in dirs if d is not None]
        subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
        return so

    build._soup_launcher_build = (level, nice)
    return build


def install_launcher_build():
    """SOUP_SOLUTION_LAUNCHER_OPT / SOUP_SOLUTION_LAUNCHER_NICE (see the module docstring): installs
    the build through `triton.knobs.build.impl` unless someone set that hook already. Returns
    whether it is installed."""
    level = os.environ.get("SOUP_SOLUTION_LAUNCHER_OPT", "").strip()
    try:
        nice = int(os.environ.get("SOUP_SOLUTION_LAUNCHER_NICE", "0") or 0)
    except ValueError:
        nice = 0
    if level not in ("0", "1", "2", "3", "s"):
        level = None
    if level is None and nice <= 0:
        return False
    try:
        from triton import knobs

        current = knobs.build.impl
        if current is not None:
            return bool(getattr(current, "_soup_launcher_build", None))
        knobs.build.impl = _launcher_build(level or "3", max(0, nice))
        return True
    except Exception:  # noqa: BLE001 - another triton: its own build then
        return False


def start_early(log=True):
    """apply()'s first warm-up call: triton's own hash in a thread of its own, now, next to the
    imports (see _triton_key). With SOUP_SOLUTION_WARMUP_START=apply_begin the warm-up thread
    starts here too, on this file's registrations (start_background_warmup(final=False)); by
    default it starts at the end of apply()."""
    install_launcher_build()  # off by default (SOUP_SOLUTION_LAUNCHER_OPT / _NICE)
    threading.Thread(target=_triton_key, name="solution-triton-key", daemon=True).start()
    if os.environ.get("SOUP_SOLUTION_WARMUP_START", "apply_end") == "apply_begin":
        return start_background_warmup(log=log, final=False)
    return None


def start_background_warmup(log=True, final=True):
    """Queues every registration not queued yet and, on the first call, starts the daemon thread
    that runs them with WORKERS workers (SOUP_SOLUTION_WARMUP_WORKERS), on the caller's CUDA
    device and on a stream of its own. `final` (the call at the end of apply()): nothing more will
    be registered, so the workers end once the queue is empty; an earlier call with final=False
    lets them start on what is registered so far, and the final call adds what the patch functions
    registered since. Returns the thread (the same one on later calls), or None without CUDA.
    SOUP_SOLUTION_WARMUP_SKIP=name,name (or : or . between names) leaves registrations out, for
    measuring what each one saves."""
    with _LOCK:
        if _STATE["thread"] is None:
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
        skip = set(filter(None, re.split(r"[,:.]", os.environ.get("SOUP_SOLUTION_WARMUP_SKIP", ""))))
        for name, fn in list(WARMUPS):
            if name not in _STATE["queued"] and name not in skip:
                _STATE["queued"].add(name)
                _STATE["queue"].put((name, fn))
        if final:
            _STATE["closed"] = True
        if _STATE["thread"] is None:
            _STATE["begin"] = time.perf_counter()
            workers = int(_setting("SOUP_SOLUTION_WARMUP_WORKERS", WORKERS, int))
            switch = _setting("SOUP_SOLUTION_WARMUP_SWITCH_MS", SWITCH_INTERVAL * 1000, float) / 1000
            thread = threading.Thread(target=_run, args=(device, log, workers, switch),
                                      name="solution-warmup", daemon=True)
            _STATE["thread"] = thread
            _register_shutdown()
            thread.start()
        return _STATE["thread"]


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
    served by triton's on-disk cache). Each record:
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

