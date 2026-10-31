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
