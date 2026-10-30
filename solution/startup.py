"""The time before the first training step: imports, the first CUDA work, and where the rest goes.

Imports (`prepare_imports`, first thing in apply())
---------------------------------------------------
apply() imports what the patches need: transformers' model and generation modules, trl's
SFTTrainer (and through it peft, bitsandbytes, datasets). Soup imports the same modules later in a
run without our package, so their cost moves, it does not add. Three parts of it are work that
this run does not use, and `prepare_imports` skips them (Kaggle T4, `python -X importtime` of
apply(), commit d9f60bb; SOUP_SOLUTION_IMPORT_DIET=0 switches all three off):

- 2.38 s: trl computes `importlib.metadata.packages_distributions()` when imported
  (trl/import_utils.py:25), a read of the metadata of every installed distribution, about a
  thousand on Kaggle. transformers computed the same mapping in this process a few seconds before
  (transformers/utils/import_utils.py:47, in the harness's `import transformers`), and since then
  nothing was installed (the rules forbid it; the harness compares `pip freeze` before and after)
  and only the repository root, which holds no distribution, was added to sys.path. While trl is
  imported, `packages_distributions` returns a copy of transformers' mapping. trl uses it only to
  find the version of liger_kernel and vllm (`_is_package_available(..., return_version=True)`).
- 0.84 s: `transformers.generation.candidate_generator` imports scikit-learn when it is installed
  (candidate_generator.py:30-31), for `roc_curve` in assisted generation with an adaptive
  confidence threshold (:190-196, :253, :327). Training never generates. While that module is
  imported, `transformers.utils.is_sklearn_available` answers False; the module keeps that answer
  (it skips the adaptive threshold, as on a machine without scikit-learn).
- 0.46 s: trl's base trainer imports wandb when it is installed (trl/trainer/base_trainer.py:
  17-23), to put the W&B run's URL into a model card (:87). Soup reports to W&B only with its
  `--wandb` flag (soup_cli/commands/train.py:1257), which the run does not use; `report_to` is
  "none". While trl's trainer is imported, `transformers.is_wandb_available` answers False; the
  base trainer keeps that answer (no W&B URL in a model card, as without wandb installed).

Each replacement lives only for the import it serves and is undone right after (the original
function is put back), and only modules imported in that window keep the answer. If a module is
already imported, nothing is replaced for it.

Diagnostics (off by default)
----------------------------
SOUP_SOLUTION_PROFILE_STARTUP=1: cProfile from the end of Soup's model setup
(SFTTrainerWrapper._setup_transformers) to the end of the first training step, in three parts (the
rest of `setup`, where Soup builds TRL's SFTTrainer; `train` up to the first step; the first step),
each printed as the 40 most expensive calls by cumulative time, with the packages imported
meanwhile. cProfile makes Python code about twice as slow: the shares count, not the seconds.
"""

import contextlib
import importlib
import os
import sys
import time

from .phases import mark


def _flag(name):
    return os.environ.get(name, "0") not in ("0", "false", "no", "")


# ---------------------------------------------------------------- imports

@contextlib.contextmanager
def _replaced(owner, name, value):
    """`owner.name = value` for the duration of the block, then the original back (unless someone
    else replaced it meanwhile)."""
    original = getattr(owner, name)
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


def _import(name, seconds):
    if name in sys.modules:
        return
    start = time.perf_counter()
    importlib.import_module(name)
    seconds[name] = round(time.perf_counter() - start, 3)


def prepare_imports(log=True):
    """The imports of apply()'s patches, done first with the shortcuts of the module docstring.
    Returns {module: seconds}. Any failure leaves the rest to the plain imports of the patches."""
    seconds = {}
    try:
        import transformers
        import transformers.utils

        with _replaced(transformers.utils, "is_sklearn_available", _not_installed):
            _import("transformers.generation.candidate_generator", seconds)
        with distributions_from_transformers():
            _import("trl", seconds)
            # transformers' top-level lazy attribute, which trl's base trainer imports by name
            with _replaced(transformers, "is_wandb_available", _not_installed):
                _import("trl.trainer.sft_trainer", seconds)
    except Exception as exc:  # noqa: BLE001 - the patches import what they need themselves
        if log:
            print(f"[solution] import shortcuts skipped: {exc!r}", flush=True)
    if log and seconds:
        print("[solution] imports: " + ", ".join(f"{name} {s:.2f} s" for name, s in seconds.items()),
              flush=True)
    return seconds


# ---------------------------------------------------------------- diagnostics: profile of the setup

def _top_packages(names):
    counts = {}
    for name in names:
        top = name.split(".")[0]
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
    state = {"profile": None}

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
        result = original_setup(self, *args, **kwargs)
        if state["profile"] is None:
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

    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and hasattr(trainer, "add_callback"):
            trainer.add_callback(FirstStepMark())
        return original_train(self, *args, **kwargs)

    cls.train = train
    cls._soup_first_step_mark = True
