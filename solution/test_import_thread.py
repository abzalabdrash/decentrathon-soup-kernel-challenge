"""startup.start_import_thread (SOUP_SOLUTION_IMPORT_THREAD=1): trl's imports and the trl patch run
on a thread and are in place once Soup's setup() starts; a failed thread is redone on the main
thread; without Soup's trainer wrapper everything runs at once, as before."""

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from solution import startup

REPO = Path(__file__).resolve().parents[1]

THREAD_CHILD = r"""
import importlib, importlib.metadata as metadata, sys
import torch
import transformers
import transformers.utils
original_pd = metadata.packages_distributions
original_sk = transformers.utils.is_sklearn_available
sys.path.insert(0, sys.argv[1])
from solution.startup import (prepare_imports, start_import_thread, patch_soup_import_join,
                              _not_installed, _THREAD)
from solution.trainer_patch import patch_sft_trainer
seconds = prepare_imports(log=False, trl=False)
print("main_imported=" + ",".join(sorted(seconds)))
print("started=" + str(start_import_thread(patch_sft_trainer)))
patch_soup_import_join()
from soup_cli.trainer import sft as soup_sft
calls = []
def fake_setup(self, dataset):
    import trl
    calls.append(getattr(trl.SFTTrainer, "_soup_fused_metrics", False))
# the join wrapper calls the setup it wrapped; replace that one by a stand-in for Soup's body
cls = soup_sft.SFTTrainerWrapper
wrapper = cls.setup
(cell,) = wrapper.__closure__  # `original`
cell.cell_contents = fake_setup
wrapper(object(), {})
print("patched_at_setup=" + str(calls == [True]))
print("thread_imported=" + ",".join(sorted(_THREAD["seconds"])))
print("restored=" + str(metadata.packages_distributions is original_pd
                        and transformers.utils.is_sklearn_available is original_sk
                        and transformers.is_wandb_available is importlib.import_module(
                            "transformers.integrations.integration_utils").is_wandb_available))
base_trainer = sys.modules["trl.trainer.base_trainer"]
print("wandb_answer_kept=" + str(base_trainer.is_wandb_available is _not_installed))
print("skipped_loaded=" + ",".join(n for n in ("sklearn", "wandb") if n in sys.modules))
import builtins
print("guard_removed=" + str(not hasattr(builtins.__import__, "_soup_import_guard")
                             and not hasattr(importlib.import_module, "_soup_import_guard")))
"""


def test_thread_path_in_a_fresh_process():
    pytest.importorskip("trl")
    pytest.importorskip("soup_cli")
    out = subprocess.run([sys.executable, "-c", THREAD_CHILD, str(REPO)], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line
                 and not line.startswith("["))
    assert lines["started"] == "True"
    assert "trl.trainer.sft_trainer" not in lines["main_imported"]
    assert "transformers.data" not in lines["main_imported"]
    assert "trl.trainer.sft_trainer" in lines["thread_imported"]
    assert lines["patched_at_setup"] == "True"
    assert lines["restored"] == "True"
    assert lines["wandb_answer_kept"] == "True"
    assert lines["skipped_loaded"] == ""
    assert lines["guard_removed"] == "True"
    assert "[solution] import thread: " in out.stdout
    assert "main thread waited" in out.stdout


class _Wrapper:
    def setup(self, dataset):
        return ("setup", dataset)


@pytest.fixture
def fake_soup(monkeypatch):
    import types

    trainer = types.ModuleType("soup_cli.trainer")
    sft = types.ModuleType("soup_cli.trainer.sft")
    wrapper = type("SFTTrainerWrapper", (_Wrapper,), {})
    sft.SFTTrainerWrapper = wrapper
    trainer.sft = sft
    root = types.ModuleType("soup_cli")
    root.trainer = trainer
    monkeypatch.setitem(sys.modules, "soup_cli", root)
    monkeypatch.setitem(sys.modules, "soup_cli.trainer", trainer)
    monkeypatch.setitem(sys.modules, "soup_cli.trainer.sft", sft)
    monkeypatch.setattr(startup, "_THREAD", dict(startup._THREAD, thread=None))
    return wrapper


def test_patch_runs_on_the_thread_and_setup_joins(fake_soup, monkeypatch):
    gate, where = threading.Event(), []
    monkeypatch.setattr(startup, "_trl_imports", lambda seconds: (gate.wait(5), seconds.update(x=0.0)))
    assert startup.start_import_thread(lambda: where.append(threading.current_thread().name))
    startup.patch_soup_import_join()
    startup.patch_soup_import_join()  # once
    assert fake_soup.setup._soup_import_join
    gate.set()
    assert fake_soup().setup("rows") == ("setup", "rows")
    assert where == ["soup-import"]
    assert startup._THREAD["thread"] is None
    assert fake_soup().setup("again") == ("setup", "again")  # the join is done once
    assert where == ["soup-import"]


def test_failed_thread_is_redone_on_the_main_thread(fake_soup, monkeypatch, capsys):
    calls = []

    def imports(seconds):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            raise RuntimeError("deadlock detected")

    monkeypatch.setattr(startup, "_trl_imports", imports)
    where = []
    startup.start_import_thread(lambda: where.append(threading.current_thread().name))
    startup.patch_soup_import_join()
    fake_soup().setup(None)
    assert calls == ["soup-import", threading.current_thread().name]
    assert where == [threading.current_thread().name]
    assert "redone here" in capsys.readouterr().out


def test_without_soup_everything_runs_at_once(monkeypatch):
    monkeypatch.setitem(sys.modules, "soup_cli", None)  # the import raises ImportError
    monkeypatch.setattr(startup, "_THREAD", dict(startup._THREAD, thread=None))
    calls = []
    monkeypatch.setattr(startup, "_trl_imports", lambda seconds: calls.append("imports"))
    assert startup.start_import_thread(lambda: calls.append("patch")) is False
    assert calls == ["imports", "patch"]
    assert startup._THREAD["thread"] is None


class _DeadlockError(RuntimeError):
    """Stands for importlib._bootstrap._DeadlockError (matched by name)."""


@pytest.fixture
def window(monkeypatch):
    """A running import thread (released by the returned event) with the import guard's state on,
    without touching builtins: the tests call _guarded around stand-in imports."""
    gate = threading.Event()
    thread = threading.Thread(target=gate.wait, args=(5,), daemon=True)
    monkeypatch.setattr(startup, "_THREAD", dict(startup._THREAD, thread=thread, error=None,
                                                 seconds={}, start=0.0, done=None, retried=0))
    monkeypatch.setattr(startup, "_GUARD", {"main": threading.get_ident(), "depth": 0,
                                            "originals": ("unused", "unused")})
    monkeypatch.setattr(startup, "_remove_guard", lambda: startup._GUARD.update(originals=None))
    thread.start()
    yield gate
    gate.set()


def test_nested_deadlock_passes_except_exception_and_is_redone_after_the_join(window):
    calls = []

    def inner_import():
        calls.append("inner")
        if len(calls) == 1:
            window.set()
            raise _DeadlockError("deadlock detected by _ModuleLock('x')")
        return "module"

    inner = startup._guarded(inner_import)

    def outer_import():
        try:
            return inner()
        except Exception:  # what an optional import would do: take the module as missing
            return "missing"

    assert startup._guarded(outer_import)() == "module"
    assert calls == ["inner", "inner"]
    assert startup._THREAD["retried"] == 1
    assert startup._THREAD["thread"] is None


def test_outermost_failure_is_retried_once_then_raised(window):
    calls = []

    def missing():
        calls.append(1)
        window.set()
        raise ModuleNotFoundError("No module named 'nothing'")

    with pytest.raises(ModuleNotFoundError):
        startup._guarded(missing)()
    assert len(calls) == 2  # the retry, on the plain path, fails as the import would have


def test_nested_plain_error_stays_with_the_importer(window):
    def inner_import():
        raise ModuleNotFoundError("No module named 'optional'")

    inner = startup._guarded(inner_import)

    def outer_import():
        try:
            return inner()
        except ImportError:
            return "fallback"

    assert startup._guarded(outer_import)() == "fallback"
    assert startup._THREAD["retried"] == 0


def test_other_threads_pass_through(window):
    seen = []
    guarded = startup._guarded(lambda: seen.append(startup._GUARD["depth"]) or "ok")
    worker = threading.Thread(target=lambda: seen.append(guarded()))
    worker.start()
    worker.join()
    assert seen == [0, "ok"]


def test_guard_installed_and_removed(fake_soup, monkeypatch):
    import builtins
    import importlib

    before = (builtins.__import__, importlib.import_module)
    gate = threading.Event()
    monkeypatch.setattr(startup, "_trl_imports", lambda seconds: gate.wait(5))
    startup.start_import_thread(lambda: None)
    try:
        assert builtins.__import__._soup_import_guard is before[0]
        assert importlib.import_module._soup_import_guard is before[1]
        import json  # noqa: F401 - an import through the guard on the main thread
    finally:
        gate.set()
        startup.join_import_thread()
    assert (builtins.__import__, importlib.import_module) == before


DEADLOCK_CHILD = r"""
import sys, threading
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])
import builtins
from solution import startup
original = builtins.__import__
startup._install_guard()
thread = threading.Thread(target=lambda: __import__("cyc_a"), name="soup-import", daemon=True)
startup._THREAD.update(thread=thread, error=None, seconds={}, start=0.0, done=None, retried=0)
thread.start()
import cyc_b  # cyc_b imports cyc_a while the thread, inside cyc_a, waits for cyc_b's lock
print("b_done=" + str(cyc_b.DONE))
print("a_done=" + str(sys.modules["cyc_a"].DONE))
print("retried=" + str(startup._THREAD["retried"]))
print("guard_removed=" + str(builtins.__import__ is original))
"""

CYC_A = """
import threading, time
STATE = __import__("cyc_state")
STATE.a_started.set()
STATE.b_started.wait(5)
time.sleep(0.2)
import cyc_b
DONE = True
"""

CYC_B = """
import importlib, threading, time
STATE = __import__("cyc_state")
STATE.a_started.wait(5)
STATE.b_started.set()
time.sleep(0.5)  # the thread is now waiting for this module's lock
{take_a}
DONE = True
"""

# importlib.import_module raises _DeadlockError; the statement takes cyc_a partially initialized
# and the `from` import fails on it
TAKE_A = {"import_module": "A_DONE = importlib.import_module('cyc_a').DONE",
          "statement": "from cyc_a import DONE as A_DONE"}

CYC_STATE = """
import threading
a_started, b_started = threading.Event(), threading.Event()
"""


@pytest.mark.parametrize("take_a", sorted(TAKE_A))
def test_real_import_deadlock_on_the_main_thread_is_redone(tmp_path, take_a):
    (tmp_path / "cyc_a.py").write_text(CYC_A)
    (tmp_path / "cyc_b.py").write_text(CYC_B.format(take_a=TAKE_A[take_a]))
    (tmp_path / "cyc_state.py").write_text(CYC_STATE)
    out = subprocess.run([sys.executable, "-c", DEADLOCK_CHILD, str(REPO), str(tmp_path)],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-3000:]
    lines = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line
                 and not line.startswith("["))
    assert lines == {"b_done": "True", "a_done": "True", "retried": "1", "guard_removed": "True"}
