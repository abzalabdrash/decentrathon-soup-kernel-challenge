"""alloc_watch.py: the pool mapped back up to its peak after a release, the callback's log, and the
patch only on a real CUDA trainer."""

import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

MIB = 2 ** 20
HARNESS_ALLOC_CONF = "expandable_segments:True"  # harness/common.py ALLOC_CONF
EXPANDABLE = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == HARNESS_ALLOC_CONF


def _spike_and_release(mib):
    """A transient of `mib` MiB above the current pool, then empty_cache: torch's reserved peak
    stays, the pool goes back down (the setup pattern that failed check 9)."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_reserved()
    spike = torch.empty(mib * MIB, dtype=torch.uint8, device="cuda")
    peak = torch.cuda.memory_reserved()
    assert peak >= base + mib * MIB
    del spike
    torch.cuda.empty_cache()
    assert torch.cuda.memory_reserved() <= base + 2 * MIB
    assert torch.cuda.max_memory_reserved() == peak
    return peak


def test_refill_maps_the_pool_back_to_its_peak():
    from solution.alloc_watch import refill_to_peak

    peak = _spike_and_release(256)
    allocated = torch.cuda.memory_allocated()
    added = refill_to_peak()
    reserved = torch.cuda.memory_reserved()
    assert added > 200 * MIB
    assert peak - 2 * MIB <= reserved <= peak + 2 * MIB
    assert torch.cuda.max_memory_reserved() <= peak + 2 * MIB  # no new peak of its own
    assert torch.cuda.memory_allocated() == allocated  # the chunks are freed, the pool keeps them
    if EXPANDABLE:
        # the freed chunks merge in the expandable segment and serve a larger request without
        # mapping anything (with plain segments each chunk is a segment of its own)
        x = torch.empty(200 * MIB, dtype=torch.uint8, device="cuda")
        assert torch.cuda.memory_reserved() == reserved
        del x
    assert refill_to_peak() == 0  # nothing to do at the peak
    torch.cuda.empty_cache()


@pytest.mark.skipif(EXPANDABLE, reason="this is the run under the harness's setting")
def test_under_the_harness_allocator_setting():
    """The refill and callback tests again in a process with the harness's allocator setting (it
    must be set before CUDA starts, and this process has started it)."""
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=HARNESS_ALLOC_CONF)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           f"{Path(__file__).name}::test_refill_maps_the_pool_back_to_its_peak",
                           f"{Path(__file__).name}::test_callback_logs_and_guards"],
                          cwd=Path(__file__).parent, env=env, capture_output=True, text=True,
                          timeout=600, check=False)
    assert proc.returncode == 0 and "3 passed" in proc.stdout, proc.stdout[-3000:] + proc.stderr[-2000:]


@pytest.mark.parametrize("guard", [True, False])
def test_callback_logs_and_guards(guard):
    from solution.alloc_watch import make_callback

    lines = []
    callback = make_callback(guard=guard, log=lines.append)
    args = state = control = None
    peak = _spike_and_release(128)
    callback.on_train_begin(args, state, control)
    if guard:
        assert torch.cuda.memory_reserved() >= peak - 2 * MIB
        assert "pool mapped back up" in lines[0]
    else:
        assert lines == []
    for _ in range(3):
        work = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda")
        del work
        callback.on_step_end(args, state, control)
    torch.cuda.empty_cache()  # a release in the middle of training
    callback.on_step_end(args, state, control)
    callback.on_train_end(args, state, control)
    summary = lines[-1]
    assert summary.startswith("[solution] allocator: reserved MiB at train begin")
    assert f"guard {'on' if guard else 'off'}" in summary
    assert "after 1 of 4 steps" in summary  # the release is seen either way
    if guard:  # ... and mapped back right away
        assert sum("pool mapped back up" in line for line in lines) == 2
        assert torch.cuda.memory_reserved() >= torch.cuda.max_memory_reserved() - 2 * MIB
    else:
        assert torch.cuda.memory_reserved() < torch.cuda.max_memory_reserved() - 32 * MIB
    torch.cuda.empty_cache()


def test_patch_only_for_a_real_cuda_trainer(monkeypatch):
    from soup_cli.trainer import sft
    from transformers import Trainer

    from solution import alloc_watch

    added = []

    def fake_train(self):
        return "trained"

    monkeypatch.setattr(sft.SFTTrainerWrapper, "train", fake_train)
    monkeypatch.delattr(sft.SFTTrainerWrapper, "_soup_alloc_watch", raising=False)
    alloc_watch.patch_soup_alloc_watch()
    try:
        assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=object())) == "trained"

        class RealTrainer(Trainer):
            def add_callback(self, callback):
                added.append(callback)

        trainer = RealTrainer.__new__(RealTrainer)
        trainer.model = torch.nn.Linear(2, 2)
        assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=trainer)) == "trained"
        assert added == []  # model on the CPU
        trainer.model = trainer.model.cuda()
        assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=trainer)) == "trained"
        assert len(added) == 1 and type(added[0]).__name__ == "AllocWatch"
        alloc_watch.patch_soup_alloc_watch()  # idempotent
        assert sft.SFTTrainerWrapper.train.__qualname__.startswith("patch_soup_alloc_watch")
    finally:
        for name in ("_soup_alloc_watch", "_soup_alloc_watch_previous"):
            if name in vars(sft.SFTTrainerWrapper):
                delattr(sft.SFTTrainerWrapper, name)
