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


def test_off_by_default(clean_hook, monkeypatch):
    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHER_OPT", raising=False)
    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHER_NICE", raising=False)
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
