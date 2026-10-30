"""Wall-clock marks of the phases of a run, printed to the training log.

The score is the wall time of the whole `soup train` command, and only part of it is the training
loop: imports, model load, trainer setup and the adapter save are the rest. These marks say where
our run spends that time (the harness records only the loop and `apply()`).
"""

import atexit
import os
import time

_START = None


def _process_start():
    try:
        import psutil

        return psutil.Process(os.getpid()).create_time()
    except Exception:  # noqa: BLE001 - psutil is optional
        return None


def mark(name):
    """Print `[solution] phase <name> +<seconds since process start> s`."""
    global _START
    if _START is None:
        _START = _process_start() or time.time()
    memory = ""
    if os.environ.get("SOUP_SOLUTION_DIAG", "0") not in ("0", "false", "no", ""):
        try:
            from .diag import memory_line

            memory = memory_line()
        except Exception:  # noqa: BLE001 - diagnostics only
            memory = ""
    print(f"[solution] phase {name} +{time.time() - _START:.2f} s" + (f"  {memory}" if memory else ""),
          flush=True)
    if os.environ.get("SOUP_SOLUTION_STARTUP_PROFILE", "0") not in ("0", "false", "no", ""):
        from .startup_profile import on_mark

        on_mark(name)  # off by default: CPU split per mark and a sampling profile, see startup_profile.py


def patch_soup_phases():
    """Marks after Soup builds the model, when training starts and ends, and at exit."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_phases", False):
        return
    original_setup = cls._setup_transformers
    original_train = cls.train

    def _setup_transformers(self, *args, **kwargs):
        mark("setup_transformers begin")
        result = original_setup(self, *args, **kwargs)
        mark("setup_transformers end")
        return result

    def train(self, *args, **kwargs):
        mark("train begin")
        result = original_train(self, *args, **kwargs)
        mark("train end")
        return result

    cls._setup_transformers = _setup_transformers
    cls.train = train
    cls._soup_phases = True
    atexit.register(mark, "exit")
