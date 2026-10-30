"""startup.py: the import shortcuts leave nothing replaced behind and change no answer the run uses,
the tokenizer's fast fingerprint is deterministic and follows its content, the hooks clean up."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

IMPORTS_CHILD = r"""
import importlib.metadata as metadata, os, sys
import torch
import transformers
import transformers.utils
original_pd = metadata.packages_distributions
original_sk = transformers.utils.is_sklearn_available
sys.path.insert(0, sys.argv[1])
from solution.startup import prepare_imports
seconds = prepare_imports(log=False)
import importlib
trl_import_utils = importlib.import_module("trl.import_utils")
from bitsandbytes.backends.default import ops
print("imported=" + ",".join(sorted(seconds)))
print("restored=" + str(metadata.packages_distributions is original_pd
                        and transformers.utils.is_sklearn_available is original_sk
                        and transformers.is_wandb_available is importlib.import_module(
                            "transformers.integrations.integration_utils").is_wandb_available
                        and "TORCHDYNAMO_DISABLE" not in os.environ))
print("trl_mapping_exact=" + str(trl_import_utils.PACKAGE_DISTRIBUTION_MAPPING
                                 == metadata.packages_distributions()))
metrics = importlib.import_module("transformers.data.metrics")
print("sklearn_answer=" + str(metrics.is_sklearn_available()))
from solution.startup import _not_installed
base_trainer = sys.modules["trl.trainer.base_trainer"]
print("wandb_answer_kept=" + str(base_trainer.is_wandb_available is _not_installed))
from solution.startup import _not_installed
detection = importlib.import_module("transformers.loss.loss_for_object_detection")
image_utils = importlib.import_module("transformers.image_utils")
print("detection_answers=" + str(detection.is_scipy_available is _not_installed
                                  and image_utils.is_torchvision_available is _not_installed))
print("bnb_plain=" + str(type(ops._dequantize_4bit_compute).__name__ == "function"
                         and not hasattr(ops._dequantize_4bit_compute, "_torchdynamo_orig_callable")))
print("heavy=" + ",".join(m for m in ("sklearn", "wandb", "networkx", "torch._inductor.compile_fx")
                          if m in sys.modules))
"""


def test_prepare_imports_in_a_fresh_process():
    env = {k: v for k, v in os.environ.items() if k != "TORCHDYNAMO_DISABLE"}
    out = subprocess.run([sys.executable, "-c", IMPORTS_CHILD, str(REPO)], capture_output=True,
                         text=True, timeout=600, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    res = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
    assert {"trl", "trl.trainer.sft_trainer", "bitsandbytes", "transformers.data",
            "transformers.loss.loss_utils"} <= set(res["imported"].split(","))
    assert res["restored"] == "True"
    assert res["trl_mapping_exact"] == "True"  # trl got the mapping a scan gives now
    assert res["bnb_plain"] == "True"
    assert res["sklearn_answer"] == "False"  # what the module saw at its import
    assert res["wandb_answer_kept"] == "True"  # trl's base trainer got the replacement
    assert res["detection_answers"] == "True"  # the loss modules and image_utils kept the answer
    assert res["heavy"] == ""


def test_dynamo_switch_restores_an_earlier_value(monkeypatch):
    from solution.startup import _dynamo_disabled

    monkeypatch.setenv("TORCHDYNAMO_DISABLE", "0")
    with _dynamo_disabled():
        assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert os.environ["TORCHDYNAMO_DISABLE"] == "0"
    monkeypatch.delenv("TORCHDYNAMO_DISABLE")
    with _dynamo_disabled():
        pass
    assert "TORCHDYNAMO_DISABLE" not in os.environ


def test_replaced_acts_on_the_module_in_sys_modules(monkeypatch):
    """A module that puts another object in its place in sys.modules on its first attribute read
    (transformers does): the replacement and the restore act on that object."""
    import types

    from solution.startup import _replaced

    first, second = types.ModuleType("soup_test_lazy"), types.ModuleType("soup_test_lazy")
    second.flag = "original"
    monkeypatch.setitem(sys.modules, "soup_test_lazy", first)

    def swap(name):
        sys.modules["soup_test_lazy"] = second
        return getattr(second, name)

    first.__getattr__ = swap
    with _replaced(first, "flag", "ours"):
        ns = {}
        exec("from soup_test_lazy import flag", ns)
        assert ns["flag"] == "ours"
    assert second.flag == "original"


def test_replaced_puts_back_unless_changed_meanwhile():
    from solution.startup import _replaced

    class Owner:
        value = "original"

    with _replaced(Owner, "value", "ours") as original:
        assert original == "original" and Owner.value == "ours"
    assert Owner.value == "original"
    with _replaced(Owner, "value", "ours"):
        Owner.value = "someone else's"
    assert Owner.value == "someone else's"


def test_distributions_copy_is_independent(monkeypatch):
    import importlib.metadata as metadata

    from solution import startup

    known = {"pkg": ["dist-a"], "other": ["dist-b", "dist-c"]}
    monkeypatch.setattr(startup, "_known_distributions", lambda: known)
    original = metadata.packages_distributions
    with startup.distributions_from_transformers():
        first = metadata.packages_distributions()
        first["pkg"].append("changed")
        assert metadata.packages_distributions() == {"pkg": ["dist-a"],
                                                     "other": ["dist-b", "dist-c"]}
    assert metadata.packages_distributions is original
    monkeypatch.setattr(startup, "_known_distributions", lambda: None)
    with startup.distributions_from_transformers():
        assert metadata.packages_distributions is original


def test_freeze_in_a_fresh_process():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import gc; "
            "from solution.startup import freeze_imported_objects; n = freeze_imported_objects(); "
            "print(n > 1000, gc.get_freeze_count() == n)")
    out = subprocess.run([sys.executable, "-c", code, str(REPO)], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0 and out.stdout.split() == ["True", "True"], out.stderr[-2000:]


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("datasets")
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                          local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen2.5 tokenizer not in the local cache: {exc!r}")


def test_fast_tokenizer_fingerprint(tokenizer):
    import transformers
    from datasets.fingerprint import Hasher
    from datasets.utils import _dill

    from solution.startup import _fast_tokenizer_reducer

    kind = type(tokenizer)
    assert kind not in _dill.Pickler.dispatch or not getattr(
        _dill.Pickler.dispatch[kind], "_soup_fast_tokenizer", False)
    slow = Hasher.hash(tokenizer)
    _dill.Pickler.dispatch[kind] = _fast_tokenizer_reducer(
        _dill._save_transformersPreTrainedTokenizerBase)
    try:
        start = time.perf_counter()
        fast = Hasher.hash(tokenizer)
        seconds = time.perf_counter() - start
        assert Hasher.hash(tokenizer) == fast  # deterministic
        fresh = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                           local_files_only=True)
        assert Hasher.hash(fresh) == fast  # the same for another load of the same files
        fresh._merges = fresh._merges[:-1]  # the bulky containers are part of it
        assert Hasher.hash(fresh) != fast
        fresh = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
                                                           local_files_only=True)
        fresh.add_tokens(["<soup_test_token>"])  # and so is the rest of the tokenizer
        assert Hasher.hash(fresh) != fast
    finally:
        _dill.Pickler.dispatch.pop(kind, None)
    assert Hasher.hash(tokenizer) == slow  # datasets' own reducer again
    print(f"fast fingerprint {seconds:.3f} s")
    assert seconds < 0.5


def test_dataset_hashing_hook_scope(tokenizer, monkeypatch):
    """Installed once Soup's `_setup_transformers` has the tokenizer, used by the maps of `setup`,
    gone when `setup` returns, even when it raises."""
    from datasets import Dataset
    from datasets.utils import _dill
    from soup_cli.trainer import sft as soup_sft

    from solution.startup import patch_soup_dataset_hashing

    cls = soup_sft.SFTTrainerWrapper
    kind = type(tokenizer)
    seen = {}

    def fake_setup_transformers(self, *args, **kwargs):
        self.tokenizer = tokenizer

    def fake_setup(self, fail=False):
        self._setup_transformers()
        entry = _dill.Pickler.dispatch.get(kind)
        seen["during"] = getattr(entry, "_soup_fast_tokenizer", False)
        tok = self.tokenizer

        def format_row(example):
            return {"n": len(tok(example["text"])["input_ids"])}

        start = time.perf_counter()
        seen["rows"] = Dataset.from_list([{"text": "hello world"}]).map(format_row)["n"]
        seen["seconds"] = time.perf_counter() - start
        if fail:
            raise RuntimeError("setup failed")

    monkeypatch.setattr(cls, "_soup_dataset_hashing", False, raising=False)
    monkeypatch.setattr(cls, "_setup_transformers", fake_setup_transformers)
    monkeypatch.setattr(cls, "setup", fake_setup)
    patch_soup_dataset_hashing()
    wrapper = object.__new__(cls)
    wrapper.setup()
    assert seen["during"] and seen["rows"] == [2] and seen["seconds"] < 0.5, seen
    assert not getattr(_dill.Pickler.dispatch.get(kind), "_soup_fast_tokenizer", False)
    with pytest.raises(RuntimeError):
        wrapper.setup(fail=True)
    assert not getattr(_dill.Pickler.dispatch.get(kind), "_soup_fast_tokenizer", False)
