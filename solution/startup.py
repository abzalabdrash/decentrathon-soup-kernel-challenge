"""The time before the first training step: imports, the first CUDA work, and where the rest goes.

Imports (`prepare_imports`, first thing in apply())
---------------------------------------------------
apply() imports what the patches need: transformers' model and generation modules, trl's
SFTTrainer (and through it peft, bitsandbytes, datasets). Soup imports the same modules later in a
run without our package, so their cost moves, it does not add. Five parts of it are work that
this run does not use, and `prepare_imports` skips them (Kaggle T4, `python -X importtime` of
apply(), commits d9f60bb and ba8c1b3; SOUP_SOLUTION_IMPORT_DIET=0 switches all five off):

- 2.38 s: trl computes `importlib.metadata.packages_distributions()` when imported
  (trl/import_utils.py:25), a read of the metadata of every installed distribution, about a
  thousand on Kaggle. transformers computed the same mapping in this process a few seconds before
  (transformers/utils/import_utils.py:47, in the harness's `import transformers`), and since then
  nothing was installed (the rules forbid it; the harness compares `pip freeze` before and after)
  and only the repository root, which holds no distribution, was added to sys.path. While trl is
  imported, `packages_distributions` returns a copy of transformers' mapping. trl uses it only to
  find the version of liger_kernel and vllm (`_is_package_available(..., return_version=True)`).
- 0.84 s: two transformers modules import scikit-learn (and scipy) when it is installed:
  `generation.candidate_generator` (candidate_generator.py:30-31), for `roc_curve` in assisted
  generation with an adaptive confidence threshold (:190-196, :253, :327), and the GLUE / XNLI
  metric helpers of `transformers.data` (data/metrics/__init__.py:18-20), which `transformers.data`
  imports with its collators. Training never generates nor computes GLUE metrics. While those two
  are imported, `transformers.utils.is_sklearn_available` answers False; they keep that answer
  (as on a machine without scikit-learn: no adaptive threshold, GLUE metrics unavailable).
- 0.46 s: trl's base trainer imports wandb when it is installed (trl/trainer/base_trainer.py:
  17-23), to put the W&B run's URL into a model card (:87). Soup reports to W&B only with its
  `--wandb` flag (soup_cli/commands/train.py:1257), which the run does not use; `report_to` is
  "none". While trl's trainer is imported, `transformers.is_wandb_available` answers False; the
  base trainer keeps that answer (no W&B URL in a model card, as without wandb installed).
- 0.42 s: transformers' object-detection losses (`transformers.loss`, which modeling_utils imports
  for its loss mapping) import scipy.optimize (`linear_sum_assignment`, 0.28 s;
  loss/loss_for_object_detection.py:26-27) and, through `transformers.image_transforms` and
  `transformers.image_utils` (loss/loss_deformable_detr.py:4, image_utils.py:53-56), torchvision
  (0.14 s) when they are installed. A causal LM on text never uses them. While `transformers.loss`
  is imported, `transformers.utils.is_scipy_available` and `is_torchvision_available` answer
  False; the modules imported in that window keep that answer (as without scipy and torchvision:
  the DETR-type losses would refuse to run, and transformers' image helpers would decode images
  with Pillow). Besides those, only accelerate is imported in that window, and it has its own
  checks. torchvision has a second way in, the image-processing modules (image_processing_utils.py:
  47-48, image_processing_backends.py:73-74) that trl's SFTTrainer reaches through
  `ProcessorMixin`: `is_torchvision_available` also answers False while trl's trainer is imported
  (image processors in this process would take their Pillow backend; the run has none).
- 0.76 s and more: bitsandbytes wraps four functions of its default (CPU fallback) backend in
  `torch.compile` when imported (bitsandbytes/backends/default/ops.py:12-31, 199, 262, 368, 429:
  the compute of dequantize_blockwise, dequantize_4bit and the 32-bit optimizer update), and
  building those wrappers imports Inductor (torch._inductor.compile_fx 0.76 s on Kaggle) and,
  through torch._dynamo's debug backends, functorch and networkx (0.6 s in all on the RTX 3050 Ti
  box after torch._dynamo itself is loaded, 0.03 s without). The run never calls them: every one
  of these ops has a CUDA kernel (bitsandbytes/backends/cuda/ops.py:335, 423, 1212), and the
  optimizer is torch's AdamW. While bitsandbytes is imported, TORCHDYNAMO_DISABLE=1, torch's own
  switch that makes `torch.compile(fn)` return `fn` (torch/_dynamo/eval_frame.py, `_optimize`);
  bitsandbytes keeps the plain functions, as it does when compiling fails. The variable is put
  back right after (removed, or its earlier value).

Each replacement lives only for the import it serves and is undone right after (the original
function is put back), and only modules imported in that window keep the answer. If a module is
already imported, nothing is replaced for it.

SOUP_SOLUTION_ENTRY_POINTS (on by default, =0 for the original): `importlib.metadata.entry_points()` reads the entry points of every
installed distribution on each call (about a thousand on Kaggle: 63 ms for Soup's plugin discovery
at the start of train() in the T4 profile, soup-s1-diag0), and after the harness's prelude the run
calls it three times: bitsandbytes' backend discovery (bitsandbytes/__init__.py:70) and fsspec's
registry (fsspec/__init__.py:75) during apply()'s imports, Soup's plugin discovery
(soup_cli/plugins/__init__.py:193) when train() begins. From the start of apply() the full set is
read once and every call is answered with its `select(**params)`, which is how the original builds
its answer (`EntryPoints(...).select(**params)`, importlib/metadata/__init__.py); EntryPoints is an
immutable tuple. It is read again whenever sys.path differs from the one it was read with. Nothing
is installed during a run (the rules forbid it; the harness compares `pip freeze`), so the answers
are the ones the calls would compute; the organizers allow the same for packages_distributions().
This replacement stays for the whole run.

Diagnostics (off by default)
----------------------------
SOUP_SOLUTION_PROFILE_STARTUP=1: cProfile of Soup's model setup (SFTTrainerWrapper.
_setup_transformers) and from there to the end of the first training step, in four parts (the
model setup; the rest of `setup`, where Soup builds TRL's SFTTrainer; `train` up to the first step;
the first step),
each printed as the 40 most expensive calls by cumulative time, with the packages imported
meanwhile. cProfile makes Python code about twice as slow: the shares count, not the seconds.
"""

import contextlib
import importlib
import os
import sys
import threading
import time
import types

from .phases import mark


def _flag(name, default="0"):
    return os.environ.get(name, default) not in ("0", "false", "no", "")


# ---------------------------------------------------------------- imports

@contextlib.contextmanager
def _replaced(owner, name, value):
    """`owner.name = value` for the duration of the block, then the original back (unless someone
    else replaced it meanwhile). For a module, the one in sys.modules once `name` is read: the
    first attribute read of `transformers` puts transformers' lazy module object in its place in
    sys.modules, and `from transformers import name` reads that one."""
    original = getattr(owner, name)
    if isinstance(owner, types.ModuleType):
        owner = sys.modules.get(owner.__name__, owner)
    setattr(owner, name, value)
    try:
        yield original
    finally:
        if getattr(owner, name) is value:
            setattr(owner, name, original)


def _known_distributions():
    """transformers' mapping of import names to distributions (see the module docstring), or None."""
    module = sys.modules.get("transformers.utils.import_utils")
    known = getattr(module, "PACKAGE_DISTRIBUTION_MAPPING", None)
    return known if isinstance(known, dict) and known else None


@contextlib.contextmanager
def distributions_from_transformers():
    """While the block runs, `importlib.metadata.packages_distributions()` returns a copy of the
    mapping transformers computed at its import, instead of reading every distribution again."""
    import importlib.metadata as metadata

    known = _known_distributions()
    if known is None:
        yield
        return

    def packages_distributions():
        return {name: list(dists) for name, dists in known.items()}

    with _replaced(metadata, "packages_distributions", packages_distributions):
        yield


def _not_installed():
    return False


_ENTRY_POINTS = {"path": None, "value": None, "reads": 0, "served": 0}


def _shared_entry_points(original):
    """importlib.metadata.entry_points for SOUP_SOLUTION_ENTRY_POINTS (see the module docstring)."""

    def entry_points(**params):
        path = tuple(sys.path)
        if _ENTRY_POINTS["path"] != path or _ENTRY_POINTS["value"] is None:
            _ENTRY_POINTS["value"] = original()
            _ENTRY_POINTS["path"] = path
            _ENTRY_POINTS["reads"] += 1
        else:
            _ENTRY_POINTS["served"] += 1
        everything = _ENTRY_POINTS["value"]
        return everything.select(**params) if params else everything

    entry_points._soup_original = original
    return entry_points


def share_entry_points():
    """Installs _shared_entry_points (once). Returns whether it is installed."""
    import importlib.metadata as metadata

    current = metadata.entry_points
    if getattr(current, "_soup_shared", False):
        return True
    if getattr(current, "_soup_original", None) is not None:
        return False  # someone else's replacement (the probes of trainer_build.py): leave it
    shared = _shared_entry_points(current)
    shared._soup_shared = True
    metadata.entry_points = shared
    return True


_SKIPPED = ("sklearn", "wandb", "networkx", "torch._inductor.compile_fx", "scipy.optimize",
            "torchvision")  # checked in the log, after apply() and at the first step


@contextlib.contextmanager
def _dynamo_disabled():
    """TORCHDYNAMO_DISABLE=1 for the duration of the block, then its earlier state back."""
    previous = os.environ.get("TORCHDYNAMO_DISABLE")
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
        else:
            os.environ["TORCHDYNAMO_DISABLE"] = previous


def _import(name, seconds):
    if name in sys.modules:
        return
    start = time.perf_counter()
    importlib.import_module(name)
    seconds[name] = round(time.perf_counter() - start, 3)


def _trl_imports(seconds):
    """trl's SFTTrainer and the transformers collators it imports, with the shortcuts of the module
    docstring (the part of prepare_imports that SOUP_SOLUTION_IMPORT_THREAD moves to a thread)."""
    import transformers
    import transformers.utils

    with _replaced(transformers.utils, "is_sklearn_available", _not_installed):
        _import("transformers.data", seconds)  # trl imports its collators
    with distributions_from_transformers():
        _import("trl", seconds)
        # is_wandb_available: transformers' top-level lazy attribute, which trl's base trainer
        # imports by name; is_torchvision_available: transformers' image-processing modules,
        # which trl's SFTTrainer reaches through ProcessorMixin
        with (_replaced(transformers, "is_wandb_available", _not_installed),
              _replaced(transformers.utils, "is_torchvision_available", _not_installed)):
            _import("trl.trainer.sft_trainer", seconds)


def prepare_imports(log=True, trl=True):
    """The imports of apply()'s patches, done first with the shortcuts of the module docstring.
    Returns {module: seconds}. Any failure leaves the rest to the plain imports of the patches.
    trl=False leaves out _trl_imports (start_import_thread does them)."""
    seconds = {}
    try:
        from .load_early import start as start_early_read

        start_early_read()  # SOUP_SOLUTION_LOAD_EARLY: the checkpoint read into pinned memory during the imports
    except Exception:  # noqa: BLE001 - the loader stages as before
        pass
    if _flag("SOUP_SOLUTION_ENTRY_POINTS", "1"):
        try:
            share_entry_points()  # before bitsandbytes and fsspec read them
        except Exception:  # noqa: BLE001 - every call reads them then
            pass
    try:
        import transformers
        import transformers.utils

        with _replaced(transformers.utils, "is_sklearn_available", _not_installed):
            _import("transformers.generation.candidate_generator", seconds)
            if trl:
                _import("transformers.data", seconds)  # trl imports its collators
        with (_replaced(transformers.utils, "is_scipy_available", _not_installed),
              _replaced(transformers.utils, "is_torchvision_available", _not_installed)):
            _import("transformers.loss.loss_utils", seconds)  # modeling_utils needs it at the first patch
        with _dynamo_disabled():
            _import("bitsandbytes", seconds)
        if trl:
            _trl_imports(seconds)
    except Exception as exc:  # noqa: BLE001 - the patches import what they need themselves
        if log:
            print(f"[solution] import shortcuts skipped: {exc!r}", flush=True)
    if log and seconds:
        loaded = [name for name in _SKIPPED if name in sys.modules]
        print("[solution] imports: " + ", ".join(f"{name} {s:.2f} s" for name, s in seconds.items())
              + f"; loaded of {', '.join(_SKIPPED)}: {', '.join(loaded) or 'none'}", flush=True)
    from .startup_profile import mark as profile_mark

    profile_mark("imports done")  # SOUP_SOLUTION_STARTUP_PROFILE=1 only
    return seconds


# ---------------------------------------------------------------- imports in a thread

_THREAD = {"thread": None, "error": None, "seconds": {}, "start": None, "done": None, "retried": 0}


class _ImportRetry(BaseException):
    """A main-thread import that failed because of the import thread (an import-lock deadlock, a
    partially initialized module), carried past `except Exception` to the outermost import, which
    joins the thread and imports again. BaseException: the import code in between must not take
    it for a missing optional module."""


_GUARD = {"main": None, "depth": 0, "originals": None}


def _concurrent(exc):
    """Whether a nested import error can come from the other thread holding a module lock."""
    if type(exc).__name__ == "_DeadlockError":  # importlib._bootstrap: raised to the importer
        return True
    # an `import x` statement of a module in sys.modules accepts it partially initialized on a
    # deadlock (_lock_unlock_module), and the next `from x import y` or `x.y` fails
    return isinstance(exc, (ImportError, AttributeError)) and "partially initialized" in str(exc)


def _guarded(original):
    """`original` (builtins.__import__ or importlib.import_module) for the window of the import
    thread: on the main thread, an import that fails while the thread runs is imported again after
    the join, on the plain path (_DeadlockError is not caught by importlib when it enters a module
    lock, importlib/_bootstrap.py `_ModuleLock.acquire`, so it would reach Soup's code)."""

    def guarded(*args, **kwargs):
        if threading.get_ident() != _GUARD["main"] or _GUARD["originals"] is None:
            return original(*args, **kwargs)
        depth = _GUARD["depth"]
        _GUARD["depth"] = depth + 1
        try:
            return original(*args, **kwargs)
        except _ImportRetry as exc:
            if depth:
                raise
            failure = exc.args[0]
        except Exception as exc:  # noqa: BLE001 - redone below, or passed on as it was
            if depth:
                if _concurrent(exc) and _GUARD["originals"] is not None:
                    raise _ImportRetry(exc) from exc
                raise
            failure = exc
        finally:
            _GUARD["depth"] = depth
        # outermost import of the main thread: no module lock of ours is held any more
        _THREAD["retried"] += 1
        if not join_import_thread(timeout=20.0):
            raise failure
        return original(*args, **kwargs)

    guarded._soup_import_guard = original
    return guarded


def _install_guard():
    import builtins

    _GUARD.update(main=threading.get_ident(), depth=0,
                  originals=(builtins.__import__, importlib.import_module))
    builtins.__import__ = _guarded(builtins.__import__)
    importlib.import_module = _guarded(importlib.import_module)


def _remove_guard():
    import builtins

    originals, _GUARD["originals"] = _GUARD["originals"], None
    if originals is None:
        return
    if getattr(builtins.__import__, "_soup_import_guard", None) is originals[0]:
        builtins.__import__ = originals[0]
    if getattr(importlib.import_module, "_soup_import_guard", None) is originals[1]:
        importlib.import_module = originals[1]


def _import_thread_run(patch, shortcuts):
    try:
        if shortcuts:
            _trl_imports(_THREAD["seconds"])
        patch()
    except BaseException as exc:  # noqa: BLE001 - join_import_thread redoes it on the main thread
        _THREAD["error"] = exc
    _THREAD["done"] = time.perf_counter()


def start_import_thread(patch, shortcuts=True):
    """SOUP_SOLUTION_IMPORT_THREAD=1: _trl_imports and `patch` (patch_sft_trainer) on a thread
    while apply() and Soup's CLI go on; patch_soup_import_join makes SFTTrainerWrapper.setup, whose
    first line imports trl (soup_cli/trainer/sft.py:734, before the model load at :767), join it
    first. soup_cli.trainer.sft is imported here, before the thread starts (patch_soup_phases
    imports it later in apply() anyway), so the modules the main thread still imports meanwhile
    are our own and those of Soup's CLI. Without Soup's wrapper, both run here as before.
    Returns whether the thread started."""
    try:
        from soup_cli.trainer import sft as soup_sft

        callable(soup_sft.SFTTrainerWrapper.setup)
    except Exception:  # noqa: BLE001 - no Soup trainer to join at: the plain order
        if shortcuts:
            try:
                _trl_imports({})
            except Exception:  # noqa: BLE001 - the patch imports what it needs itself
                pass
        patch()
        return False
    _THREAD.update(error=None, seconds={}, start=time.perf_counter(), done=None, patch=patch,
                   shortcuts=shortcuts, retried=0)
    thread = threading.Thread(target=_import_thread_run, args=(patch, shortcuts),
                              name="soup-import", daemon=True)
    _THREAD["thread"] = thread
    _install_guard()  # until the join: main-thread imports that collide with the thread are redone
    thread.start()
    return True


def patch_soup_import_join():
    """SFTTrainerWrapper.setup joins start_import_thread's thread before anything else (installed at
    the end of apply(), so it wraps our other wrappers of setup)."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    original = cls.setup
    if getattr(original, "_soup_import_join", False):
        return

    def setup(self, *args, **kwargs):
        join_import_thread()
        return original(self, *args, **kwargs)

    setup._soup_import_join = True
    cls.setup = setup


def join_import_thread(timeout=None):
    """Waits for start_import_thread's thread (once); if it failed, its work on this thread.
    Returns False only when `timeout` ran out (the thread is then left as it is)."""
    thread = _THREAD["thread"]
    if thread is None:
        return True
    begin = time.perf_counter()
    thread.join(timeout)
    if thread.is_alive():
        return False
    _THREAD["thread"] = None
    _remove_guard()
    waited = time.perf_counter() - begin
    seconds, error = _THREAD["seconds"], _THREAD["error"]
    took = (_THREAD["done"] or begin) - _THREAD["start"]
    print("[solution] import thread: " + ", ".join(f"{name} {s:.2f} s" for name, s in seconds.items())
          + f"; done {took:.2f} s after its start, main thread waited {waited:.2f} s at "
          + ("a failed import, redone" if _THREAD["retried"] else "setup")
          + f"; loaded of {', '.join(_SKIPPED)}: {', '.join(n for n in _SKIPPED if n in sys.modules) or 'none'}"
          + (f"; failed ({error!r}), redone here" if error is not None else ""), flush=True)
    if error is not None:
        if _THREAD["shortcuts"]:
            try:
                _trl_imports({})
            except Exception:  # noqa: BLE001 - the patch imports what it needs itself
                pass
        _THREAD["patch"]()
    return True


# ---------------------------------------------------------------- diagnostics: profile of the setup

def _top_packages(names):
    counts = {}
    for name in names:
        parts = name.split(".")
        top = ".".join(parts[:2]) if parts[0] in ("torch", "transformers", "trl") else parts[0]
        counts[top] = counts.get(top, 0) + 1
    return ", ".join(f"{top} {n}" for top, n in sorted(counts.items(), key=lambda kv: -kv[1])[:25])


class _Profile:
    """One cProfile session printed to the log with the modules imported while it ran."""

    def __init__(self, label):
        import cProfile

        self.label = label
        self.modules = set(sys.modules)
        self.begin = time.perf_counter()
        self.profile = cProfile.Profile()
        self.profile.enable()

    def report(self, limit=40):
        import io
        import pstats

        self.profile.disable()
        seconds = time.perf_counter() - self.begin
        new = sorted(set(sys.modules) - self.modules)
        out = io.StringIO()
        stats = pstats.Stats(self.profile, stream=out)
        stats.sort_stats("cumulative").print_stats(limit)
        print(f"[solution] profile {self.label}: {seconds:.2f} s under cProfile; {len(new)} modules "
              f"imported ({_top_packages(new) or 'none'})", flush=True)
        for line in out.getvalue().splitlines():
            if line.strip():
                print(f"[solution] profile {self.label} | {line}", flush=True)


def patch_soup_startup_profile():
    """SFTTrainerWrapper hooks of SOUP_SOLUTION_PROFILE_STARTUP=1 (see the module docstring)."""
    import transformers
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_startup_profile", False):
        return
    original_setup = cls._setup_transformers
    original_train = cls.train
    # from here (the end of apply()) to the model setup: Soup's command line, config and data
    state = {"profile": _Profile("after apply() up to the model setup")}

    class FirstStep(transformers.TrainerCallback):
        def on_step_begin(self, args, train_state, control, **kwargs):
            if state["profile"] is not None and not state.get("step"):
                state["profile"].report()
                state["step"] = True
                state["profile"] = _Profile("first step")

        def on_step_end(self, args, train_state, control, **kwargs):
            if state["profile"] is not None and state.get("step"):
                import torch

                torch.cuda.synchronize()
                state["profile"].report()
                state["profile"] = None

    def _setup_transformers(self, *args, **kwargs):
        if state.get("profile") is not None and state["profile"].label.startswith("after apply"):
            state["profile"].report()
            state["profile"] = None
        profile = _Profile("model setup") if state["profile"] is None else None
        result = original_setup(self, *args, **kwargs)
        if profile is not None:
            profile.report(limit=60)
            state["profile"] = _Profile("setup after the model (trainer build)")
        return result

    def train(self, *args, **kwargs):
        if state["profile"] is not None:
            state["profile"].report()
            state["profile"] = _Profile("train up to the first step")
            trainer = getattr(self, "trainer", None)
            if trainer is not None:
                trainer.add_callback(FirstStep())
        return original_train(self, *args, **kwargs)

    cls._setup_transformers = _setup_transformers
    cls.train = train
    cls._soup_startup_profile = True


def patch_soup_first_step_mark():
    """A phase mark when the first training step begins (the harness times the steps, not the gap
    between Soup's `train` and the first step)."""
    import transformers
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_first_step_mark", False):
        return
    from .startup_profile import install_marks
    from .trainer_build import patch_soup_trainer_build

    install_marks()  # SOUP_SOLUTION_STARTUP_PROFILE=1 only: marks inside the trainer build
    # apply() calls this function unconditionally, so the trainer-build switches hook in here
    # (SOUP_SOLUTION_EARLY_ROWS, SOUP_SOLUTION_HELD_ENCODE)
    patch_soup_trainer_build()
    from .tracker_async import enabled as tracker_thread, patch_tracker

    if tracker_thread():
        patch_tracker()  # SOUP_SOLUTION_TRACKER_THREAD: Soup's per-log SQLite commits in a thread
    original_train = cls.train

    class FirstStepMark(transformers.TrainerCallback):
        def __init__(self):
            self.done = False

        def on_step_begin(self, args, train_state, control, **kwargs):
            if not self.done:
                self.done = True
                mark("first step begin")
                loaded = [name for name in _SKIPPED if name in sys.modules]
                print(f"[solution] at the first step, loaded of {', '.join(_SKIPPED)}: "
                      f"{', '.join(loaded) or 'none'}", flush=True)

    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(FirstStepMark())
        return original_train(self, *args, **kwargs)

    cls.train = train
    cls._soup_first_step_mark = True
