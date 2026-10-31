"""warmup.install_launcher_build: triton's launcher stubs built through triton's own build hook at
another optimization level / niceness (SOUP_SOLUTION_LAUNCHER_OPT / _NICE), CPU only."""

import os
import shutil
import tempfile

import pytest

triton = pytest.importorskip("triton")

from solution import warmup  # noqa: E402

SOURCE = r"""
#define PY_SSIZE_T_CLEAN
#include <Python.h>
static PyObject *answer(PyObject *self, PyObject *args) { return PyLong_FromLong(42); }
static PyMethodDef methods[] = {{"answer", answer, METH_NOARGS, NULL}, {NULL, NULL, 0, NULL}};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "__soup_test_launcher", NULL, -1, methods};
PyMODINIT_FUNC PyInit___soup_test_launcher(void) { return PyModule_Create(&module); }
"""


@pytest.fixture
def clean_hook(monkeypatch):
    from triton import knobs

    saved = knobs.build.impl
    knobs.build.impl = None
    yield knobs
    knobs.build.impl = saved


def test_on_by_default_and_off_switch(clean_hook, monkeypatch):
    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHER_OPT", raising=False)
    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHER_NICE", raising=False)
    assert warmup.install_launcher_build() is True  # -O0 launchers by default
    assert clean_hook.build.impl._soup_launcher_build == ("0", 0)
    clean_hook.build.impl = None
    monkeypatch.setenv("SOUP_SOLUTION_LAUNCHER_OPT", "off")
    assert warmup.install_launcher_build() is False
    assert clean_hook.build.impl is None


def test_someone_elses_hook_kept(clean_hook, monkeypatch):
    def theirs(*args):
        raise AssertionError("not called")

    clean_hook.build.impl = theirs
    monkeypatch.setenv("SOUP_SOLUTION_LAUNCHER_OPT", "1")
    assert warmup.install_launcher_build() is False
    assert clean_hook.build.impl is theirs


@pytest.mark.skipif(shutil.which("gcc") is None and shutil.which("clang") is None, reason="no C compiler")
def test_builds_a_module(clean_hook, monkeypatch):
    monkeypatch.setenv("SOUP_SOLUTION_LAUNCHER_OPT", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LAUNCHER_NICE", "5")
    monkeypatch.setenv("TRITON_CACHE_DIR", tempfile.mkdtemp())
    assert warmup.install_launcher_build() is True
    assert clean_hook.build.impl._soup_launcher_build == ("0", 5)
    commands = []
    import subprocess

    original = subprocess.check_call

    def spy(cmd, *args, **kwargs):
        commands.append(list(cmd))
        return original(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "check_call", spy)
    from triton.runtime.build import compile_module_from_src

    module = compile_module_from_src(src=SOURCE, name="__soup_test_launcher")
    assert module.answer() == 42
    assert commands and "-O0" in commands[0] and "-O3" not in commands[0]
    if shutil.which("nice"):
        assert commands[0][:3] == ["nice", "-n", "5"]
    assert os.path.basename(commands[0][commands[0].index("-o") + 1]).startswith("__soup_test_launcher")


def test_warmup_order_longest(monkeypatch):
    """SOUP_SOLUTION_WARMUP_ORDER=longest queues the registrations by COSTS, longest first; the
    default keeps the registration order."""
    import queue as queue_module

    names = ["cublas", "fused_proj_masked_pair_sum", "head_stream_kernels", "unknown_kernel",
             "rms_norm_rows_fwd"]

    def queued(order):
        monkeypatch.setattr(warmup, "WARMUPS", [(name, lambda: None) for name in names])
        state = warmup._new_state()
        state["thread"] = object()  # pretend the thread runs: nothing is started
        monkeypatch.setattr(warmup, "_STATE", state)
        if order:
            monkeypatch.setenv("SOUP_SOLUTION_WARMUP_ORDER", order)
        else:
            monkeypatch.delenv("SOUP_SOLUTION_WARMUP_ORDER", raising=False)
        warmup.start_background_warmup(log=False, final=True)
        out = []
        while True:
            try:
                out.append(state["queue"].get_nowait()[0])
            except queue_module.Empty:
                return out

    assert queued(None) == names
    assert queued("longest") == ["cublas", "rms_norm_rows_fwd", "head_stream_kernels", "unknown_kernel",
                                 "fused_proj_masked_pair_sum"]


def test_light_triton_key(monkeypatch):
    """SOUP_SOLUTION_TRITON_KEY=light: triton's key with libtriton.so named by its file attributes;
    the other parts are triton's own; not installed once triton's own key was computed."""
    from triton.runtime import cache

    original = cache.triton_key
    while getattr(original, "_soup_light", False):
        pytest.skip("the light key is installed in this process already")
    light = warmup._light_triton_key()
    key = light()
    assert key == light() and key.startswith(triton.__version__)
    full = original()
    ours, theirs = key[len(triton.__version__):].split("-"), full[len(triton.__version__):].split("-")
    assert len(ours) == len(theirs)
    differ = [i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    assert len(differ) == 1 and ours[differ[0]].startswith("libtriton:")  # only the binary's part
    monkeypatch.setenv("SOUP_SOLUTION_TRITON_KEY", "light")
    assert warmup.install_light_triton_key() is False  # triton's own key is computed (cached) here
    assert cache.triton_key is original
    import functools

    monkeypatch.setattr(cache, "triton_key", functools.lru_cache()(original.__wrapped__))  # not computed yet
    assert warmup.install_light_triton_key() is True and getattr(cache.triton_key, "_soup_light", False)
    monkeypatch.setattr(cache, "triton_key", functools.lru_cache()(original.__wrapped__))
    monkeypatch.setenv("SOUP_SOLUTION_TRITON_KEY", "full")  # light is the default: anything else keeps triton's
    assert warmup.install_light_triton_key() is False and not getattr(cache.triton_key, "_soup_light", False)
