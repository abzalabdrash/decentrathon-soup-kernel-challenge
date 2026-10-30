"""Triton compilation of the run's kernels in a background thread while Soup loads the model.

Every measured run starts with empty compiler caches, so the first training step compiles every
kernel it meets (Kaggle T4, commit 1274b24: step 1 took 3.45 s, the others about 0.45 s). Here a
daemon thread compiles them ahead, during the 10 s or so the main thread spends loading and
quantizing the model, so that the first step finds them in Triton's in-memory cache.

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
In your module, at import time:

    from .warmup import compile_kernel, register, training_half_dtype

    @register
    def _warm_my_kernel():
        half = training_half_dtype()
        compile_kernel(my_kernel, half, torch.float32, 1536, 826, BLOCK=1024, num_warps=4)

once per class the training launches can take (see above). A registration that raises is logged
and skipped; the others still run. `apply()` imports the patch modules before it calls
`start_background_warmup()`, so their registrations are in place.
"""

import importlib
import threading
import time

VOCAB = 151936  # Qwen2.5's vocabulary: V and stride_row of the fused head (divisible by 16)
INTERMEDIATE = 8960  # Qwen2.5-1.5B's MLP width: 16 * 560
TOKENS = 826  # a typical packed batch; only its class matters (tokens * 8960 is always "D")

WARMUPS = []  # [(name, fn)]: run in this order by the background thread
COMPILES = []  # every compile seen by `watch_compiles`, as dicts (see there)

_LOCK = threading.Lock()
_STATE = {"thread": None, "ident": None, "begin": None, "seconds": None, "results": {},
          "errors": {}}
_DONE = threading.Event()
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


# The kernels already in solution/ (their modules are not touched; the warm-ups live here), in the
# order the first step meets them. Their launches: fused_ce.py `_head_chunks`, swiglu.py `SwiGLU`.

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
    # that is (0 too: a chunk without labelled rows)
    import torch

    from .fused_ce import BLOCK_V, IGNORE_INDEX, fused_ce_rows_kernel

    half = training_half_dtype()
    for n_grad in (387, 512):
        compile_kernel(fused_ce_rows_kernel, half, VOCAB, torch.int64, torch.float32,
                       torch.float32, torch.int32, 1.0, n_grad, VOCAB,
                       IGNORE=IGNORE_INDEX, BLOCK=BLOCK_V, num_warps=8)


@register(name="example_swiglu_bwd_kernel")
def warm_swiglu_bwd():
    from .swiglu import BLOCK, example_swiglu_bwd_kernel

    half = training_half_dtype()
    compile_kernel(example_swiglu_bwd_kernel, half, half, half, half, half,
                   TOKENS * INTERMEDIATE, BLOCK=BLOCK)


@register(name="fused_ce_rows_kernel n_grad_rows=1")
def warm_fused_ce_one():
    # a chunk that ends one labelled row past its start: rare, so it goes last
    import torch

    from .fused_ce import BLOCK_V, IGNORE_INDEX, fused_ce_rows_kernel

    compile_kernel(fused_ce_rows_kernel, training_half_dtype(), VOCAB, torch.int64, torch.float32,
                   torch.float32, torch.int32, 1.0, 1, VOCAB,
                   IGNORE=IGNORE_INDEX, BLOCK=BLOCK_V, num_warps=8)


def _run(jobs, device, log):
    _STATE["ident"] = threading.get_ident()
    begin = time.perf_counter()
    try:
        import torch

        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device)
        # a registration that does launch something stays off the main thread's stream
        with torch.cuda.stream(stream):
            for name, fn in jobs:
                start = time.perf_counter()
                try:
                    fn()
                    _STATE["results"][name] = round(time.perf_counter() - start, 3)
                except Exception as exc:  # noqa: BLE001 - a warm-up must never break training
                    _STATE["errors"][name] = repr(exc)
                    if log:
                        print(f"[solution] warmup {name} failed: {exc!r}", flush=True)
        stream.synchronize()
    except Exception as exc:  # noqa: BLE001
        _STATE["errors"]["(thread)"] = repr(exc)
        if log:
            print(f"[solution] warmup thread failed: {exc!r}", flush=True)
    finally:
        _STATE["seconds"] = round(time.perf_counter() - begin, 3)
        _DONE.set()
        if log:
            done = ", ".join(f"{name} {s:.2f} s" for name, s in _STATE["results"].items())
            print(f"[solution] warmup done in {_STATE['seconds']:.2f} s: {done or 'nothing'}"
                  + (f"; failed: {', '.join(_STATE['errors'])}" if _STATE["errors"] else ""),
                  flush=True)


def start_background_warmup(log=True):
    """Starts the daemon thread that runs every registered warm-up once, on the caller's CUDA
    device and on a stream of its own. Returns the thread (the same one on later calls), or None
    without CUDA. The registry is read now: register before calling this."""
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
    `watch_compiles` inside and outside the warm-up thread."""
    ident = _STATE["ident"]
    return {
        "started": _STATE["thread"] is not None,
        "done": _DONE.is_set(),
        "seconds": _STATE["seconds"],
        "results": dict(_STATE["results"]),
        "errors": dict(_STATE["errors"]),
        "compiles_in_warmup": sum(1 for c in COMPILES if c["thread"] == ident),
        "compiles_outside": [c for c in COMPILES if c["thread"] != ident],
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
    cached launch costs nothing extra). Each record: kernel, thread id, seconds (the compile to
    cubin; the launcher build comes on the first launch), the key's specialization, num_warps,
    num_stages. With `log`, a compile outside the warm-up thread is printed: after a complete
    warm-up the training steps should print none. Hooks already set are kept and called first."""
    from triton import knobs

    with _LOCK:
        if _HOOKS["installed"]:
            return COMPILES
        before_prev = knobs.runtime.jit_cache_hook
        after_prev = knobs.runtime.jit_post_compile_hook
        local = threading.local()

        def before(**kwargs):
            local.start = time.perf_counter()
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
                }
                COMPILES.append(record)
                if log and record["thread"] != _STATE["ident"]:
                    print(f"[solution] triton compile outside warmup: {record['kernel']} "
                          f"{record['seconds']:.2f} s ({record['spec']}; num_warps="
                          f"{record['num_warps']})", flush=True)
            except Exception:  # noqa: BLE001 - diagnostics only
                pass
            return after_prev(**kwargs) if after_prev is not None else None

        knobs.runtime.jit_cache_hook = before
        knobs.runtime.jit_post_compile_hook = after
        _HOOKS["installed"] = True
    return COMPILES


def preload_in_background(modules, log=True):
    """Imports `modules` (names) one after another in a daemon thread. Returns a threading.Event set
    once all are done (a failed import is logged and skipped); `event.errors` maps names to errors.

    A patch that needs one of them can wait on the event, or simply import it: Python's import lock
    is per module, so an import of a module the thread is still importing waits for that import to
    finish instead of running it a second time. The thread holds the GIL while it runs module code,
    so this overlaps imports only with the main thread's waits (file reads, CUDA, subprocesses)."""
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
