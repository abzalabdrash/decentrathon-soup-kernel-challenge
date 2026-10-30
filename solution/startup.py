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

The objects of all these imports live as long as the process. `freeze_imported_objects` (the end
of apply()) moves them out of the garbage collector's reach with `gc.freeze()`, CPython's API for
exactly that: the full collections that come later (accelerate's `free_memory` runs `gc.collect()`
when training starts, 0.28 s on Kaggle, and Python's own) no longer walk the ~0.5 M objects
(0.13 s per collection after apply() on the RTX 3050 Ti box, 0.00 s once frozen). Frozen objects
are still freed by reference counting; only a reference cycle among them would never be collected,
and at the end of apply() no model, tensor or dataset exists yet to be caught in one.

The trainer build (`patch_soup_dataset_hashing`)
------------------------------------------------
Soup tokenizes the training rows with `Dataset.from_list(rows).map(checked_format_row, ...)`
(soup_cli/trainer/sft.py:143-178), and datasets fingerprints every map: it pickles the transform
with its closure to hash it (datasets/fingerprint.py:253, `Hasher.hash`). The closure holds the
tokenizer, and the tokenizer keeps Python copies of its vocabulary and merges (`_vocab`, 151,643
entries; `_merges`, 151,387 pairs), which datasets' pickler, dill in pure Python, walks one object
at a time: 0.9-1.0 s on the RTX 3050 Ti box, the largest part of the 3.1 s between the model
setup and `train` on Kaggle (cProfile there: 8.0 of 9.4 s in that pickling). While Soup's `setup`
runs, the reducer datasets uses for the tokenizer's type gets the tokenizer with those two
containers already pickled by the C pickler (the same content, so the fingerprint still changes
with any change of the vocabulary or merges, and is the same in every run): 0.09 s. datasets'
pickler only ever feeds the hasher (datasets/fingerprint.py:214, builder.py:475); nothing it
writes is unpickled. The fingerprint differs from the one datasets would compute; it names cache
files only for datasets backed by files, and this one is built in memory.

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
import time
import types

from .phases import mark


def _flag(name):
    return os.environ.get(name, "0") not in ("0", "false", "no", "")


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


def freeze_imported_objects():
    """gc.freeze() (see the module docstring). Returns the number of objects frozen."""
    import gc

    gc.freeze()
    return gc.get_freeze_count()


def prepare_imports(log=True):
    """The imports of apply()'s patches, done first with the shortcuts of the module docstring.
    Returns {module: seconds}. Any failure leaves the rest to the plain imports of the patches."""
    seconds = {}
    try:
        import transformers
        import transformers.utils

        with _replaced(transformers.utils, "is_sklearn_available", _not_installed):
            _import("transformers.generation.candidate_generator", seconds)
            _import("transformers.data", seconds)  # trl imports its collators
        with (_replaced(transformers.utils, "is_scipy_available", _not_installed),
              _replaced(transformers.utils, "is_torchvision_available", _not_installed)):
            _import("transformers.loss.loss_utils", seconds)
        with _dynamo_disabled():
            _import("bitsandbytes", seconds)
        with distributions_from_transformers():
            _import("trl", seconds)
            # is_wandb_available: transformers' top-level lazy attribute, which trl's base trainer
            # imports by name; is_torchvision_available: transformers' image-processing modules,
            # which trl's SFTTrainer reaches through ProcessorMixin
            with (_replaced(transformers, "is_wandb_available", _not_installed),
                  _replaced(transformers.utils, "is_torchvision_available", _not_installed)):
                _import("trl.trainer.sft_trainer", seconds)
    except Exception as exc:  # noqa: BLE001 - the patches import what they need themselves
        if log:
            print(f"[solution] import shortcuts skipped: {exc!r}", flush=True)
    if log and seconds:
        loaded = [name for name in _SKIPPED if name in sys.modules]
        print("[solution] imports: " + ", ".join(f"{name} {s:.2f} s" for name, s in seconds.items())
              + f"; loaded of {', '.join(_SKIPPED)}: {', '.join(loaded) or 'none'}", flush=True)
    return seconds


# ---------------------------------------------------------------- the trainer build

_BULKY = ("_vocab", "_merges")  # TokenizersBackend's Python copies of the vocabulary and merges


def _fast_tokenizer_reducer(original):
    """datasets' reducer `original` given a copy of the tokenizer whose bulky containers are
    already C-pickled bytes (see the module docstring)."""
    import pickle

    def save(pickler, obj):
        try:
            state = obj.__dict__
            bulky = [key for key in _BULKY if isinstance(state.get(key), (dict, list))]
            if not bulky:
                return original(pickler, obj)
            view = object.__new__(type(obj))  # no __init__: a plain carrier of the state
            view.__dict__.update(state)
            for key in bulky:
                view.__dict__[key] = pickle.dumps(state[key], protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:  # noqa: BLE001 - datasets' own path then
            return original(pickler, obj)
        return original(pickler, view)

    save._soup_fast_tokenizer = True
    return save


def patch_soup_dataset_hashing():
    """SFTTrainerWrapper hooks: once `_setup_transformers` has loaded the tokenizer, datasets'
    pickler hashes its type with `_fast_tokenizer_reducer`; when `setup` returns (or raises), the
    previous entry is back."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_dataset_hashing", False):
        return
    original_setup = cls.setup
    original_setup_transformers = cls._setup_transformers
    installed = []  # (dispatch, type, previous entry or None)

    def install(tokenizer):
        try:
            from datasets.utils import _dill
            from transformers import PreTrainedTokenizerBase

            if not isinstance(tokenizer, PreTrainedTokenizerBase):
                return
            original = getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None)
            dispatch = _dill.Pickler.dispatch
            kind = type(tokenizer)
            previous = dispatch.get(kind)
            if original is None or getattr(previous, "_soup_fast_tokenizer", False):
                return
            if previous is not None and previous is not original:
                return  # someone else's reducer for this type: leave it
            dispatch[kind] = _fast_tokenizer_reducer(original)
            installed.append((dispatch, kind, previous))
        except Exception:  # noqa: BLE001 - datasets' own hashing then
            pass

    def uninstall():
        while installed:
            dispatch, kind, previous = installed.pop()
            if getattr(dispatch.get(kind), "_soup_fast_tokenizer", False):
                if previous is None:
                    dispatch.pop(kind, None)
                else:
                    dispatch[kind] = previous

    def _setup_transformers(self, *args, **kwargs):
        result = original_setup_transformers(self, *args, **kwargs)
        install(getattr(self, "tokenizer", None))
        return result

    def setup(self, *args, **kwargs):
        try:
            return original_setup(self, *args, **kwargs)
        finally:
            uninstall()

    cls._setup_transformers = _setup_transformers
    cls.setup = setup
    cls._soup_dataset_hashing = True


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
