"""Model setup of Soup's training run (SFTTrainerWrapper._setup_transformers): measurements for now.

SOUP_SOLUTION_LOAD_MARKS=1: phase marks around the tokenizer load, `from_pretrained`, its tensor loop
(`convert_and_load_state_dict_in_model`), `prepare_model_for_kbit_training` and `get_peft_model`.
SOUP_SOLUTION_LOAD_PROFILE=1: a cProfile of the whole `_setup_transformers`, printed to the log (the
top functions by cumulative and by own time).
Both off by default; nothing here changes what the setup does.
"""

import functools
import os

_DONE = {"marks": False, "profile": False}


def _flag(name):
    return os.environ.get(name, "0").strip().lower() not in ("0", "false", "no", "")


def _marked(original, label):
    from .phases import mark

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        mark(f"{label} begin")
        try:
            return original(*args, **kwargs)
        finally:
            mark(f"{label} end")

    wrapper._soup_load_marks = True
    return wrapper


def _wrap_function(owner, name, label):
    current = getattr(owner, name)
    if getattr(current, "_soup_load_marks", False):
        return
    setattr(owner, name, _marked(current, label))


def _wrap_classmethod(cls, name, label):
    raw = cls.__dict__.get(name)
    if not isinstance(raw, classmethod) or getattr(raw.__func__, "_soup_load_marks", False):
        return
    setattr(cls, name, classmethod(_marked(raw.__func__, label)))


def patch_load_marks():
    """Phase marks inside the setup (SOUP_SOLUTION_LOAD_MARKS=1)."""
    if not _flag("SOUP_SOLUTION_LOAD_MARKS") or _DONE["marks"]:
        return
    import peft
    from transformers import AutoTokenizer, modeling_utils
    from transformers.models.auto import auto_factory

    _wrap_classmethod(AutoTokenizer, "from_pretrained", "tokenizer")
    _wrap_classmethod(auto_factory._BaseAutoModelClass, "from_pretrained", "from_pretrained")
    _wrap_function(modeling_utils, "convert_and_load_state_dict_in_model", "load loop")
    _wrap_function(peft, "prepare_model_for_kbit_training", "kbit preparation")
    _wrap_function(peft, "get_peft_model", "get_peft_model")
    _DONE["marks"] = True


def patch_setup_profile():
    """A cProfile of SFTTrainerWrapper._setup_transformers in the log (SOUP_SOLUTION_LOAD_PROFILE=1)."""
    if not _flag("SOUP_SOLUTION_LOAD_PROFILE") or _DONE["profile"]:
        return
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    original = cls._setup_transformers

    @functools.wraps(original)
    def _setup_transformers(self, *args, **kwargs):
        import cProfile
        import io
        import pstats

        profile = cProfile.Profile()
        profile.enable()
        try:
            return original(self, *args, **kwargs)
        finally:
            profile.disable()
            for key, count in (("cumulative", 60), ("tottime", 30)):
                out = io.StringIO()
                pstats.Stats(profile, stream=out).sort_stats(key).print_stats(count)
                print(f"[solution] setup profile by {key}:\n{out.getvalue()}", flush=True)

    cls._setup_transformers = _setup_transformers
    _DONE["profile"] = True
