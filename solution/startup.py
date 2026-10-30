"""The time before the first training step: imports, the first CUDA work, and where the rest goes.

Diagnostics (off by default)
----------------------------
SOUP_SOLUTION_PROFILE_STARTUP=1: cProfile from the end of Soup's model setup
(SFTTrainerWrapper._setup_transformers) to the first training step, in two parts (the rest of
`setup`, where Soup builds TRL's SFTTrainer; then `train` up to the first step), each printed as the
40 most expensive calls by cumulative time, with the packages imported meanwhile. cProfile makes
Python code about twice as slow: the shares count, not the seconds.
"""

import os
import sys
import time

from .phases import mark


def _flag(name):
    return os.environ.get(name, "0") not in ("0", "false", "no", "")


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
            if state["profile"] is not None:
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
