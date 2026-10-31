"""The release of the pinned host memory at the end of training (exit_release.py): registered buffers
go back to pageable memory with their values (copies from them stay exact), the offload's slabs go
back to the driver, the release thread runs once and is joinable."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


class _Owner:
    pass


def test_unpin_all_keeps_values():
    from solution.weight_stream import _PINNED, pinned_host_buffer, unpin_all

    owner = _Owner()
    buffer, finalizer = pinned_host_buffer(3 << 20, owner)
    assert buffer.is_pinned() and finalizer in _PINNED
    buffer.copy_(torch.arange(3 << 20, dtype=torch.int64).to(torch.uint8))
    expected = buffer.clone()
    assert unpin_all() >= 3 << 20
    assert not buffer.is_pinned() and not finalizer.alive and not _PINNED
    gpu = torch.empty(3 << 20, dtype=torch.uint8, device="cuda")
    gpu.copy_(buffer, non_blocking=True)
    assert torch.equal(gpu.cpu(), expected)
    finalizer()  # spent: a second call (an unpin at exit) does nothing


def test_release_host_memory_frees_spares_and_idle_stack():
    from solution import offload

    torch.cuda.synchronize()
    with offload._SPARE_LOCK:
        offload._SPARE.append(torch.empty(offload.SLAB_BYTES >> 4, dtype=torch.uint8, pin_memory=True))
    stats = torch.cuda.host_memory_stats()
    before = stats.get("reserved_bytes.current", stats.get("reserved_bytes", {}).get("current", 0))
    released = offload.release_host_memory()
    assert released == offload.SLAB_BYTES >> 4
    assert not offload._SPARE
    stats = torch.cuda.host_memory_stats()
    after = stats.get("reserved_bytes.current", stats.get("reserved_bytes", {}).get("current", 0))
    if before:  # the host allocator gave the slab back to the driver
        assert after <= before - (offload.SLAB_BYTES >> 4)


def test_release_thread_runs_once(monkeypatch):
    from solution import exit_release

    monkeypatch.setattr(exit_release, "_STATE", dict(exit_release._STATE, thread=None, seconds=None))
    calls = []
    monkeypatch.setattr(exit_release, "_release", lambda device: calls.append(device))
    monkeypatch.setattr(exit_release.atexit, "register", lambda *a, **k: None)  # no report line at exit
    thread = exit_release.start_release("cuda")
    assert exit_release.start_release("cuda") is thread
    thread.join()
    assert calls == ["cuda"] and not thread.daemon


class _FakeTrainer:
    """Stands in for transformers' Trainer: save_model is a class method, the checkpoint's call passes
    _internal_call=True as Trainer._save_checkpoint does."""

    def __init__(self, events):
        self.events, self.callbacks = events, []
        self.model = torch.nn.Linear(2, 2)

    def save_model(self, output_dir=None, _internal_call=False):
        self.events.append(("save", output_dir, _internal_call))

    def add_callback(self, callback):
        self.callbacks.append(callback)


def _patched_wrapper(monkeypatch, mode, body, real=True):
    """A fake SFTTrainerWrapper whose train() runs `body(trainer)`, patched by patch_soup_exit_release
    with SOUP_SOLUTION_EXIT_RELEASE=mode; start_release only records its call."""
    from soup_cli.trainer import sft as soup_sft

    from solution import exit_release, offload

    events = []

    class Wrapper:
        def __init__(self):
            self.trainer = _FakeTrainer(events)

        def train(self):
            return body(self.trainer)

    monkeypatch.setenv("SOUP_SOLUTION_EXIT_RELEASE", mode)
    monkeypatch.setattr(soup_sft, "SFTTrainerWrapper", Wrapper)
    monkeypatch.setattr(offload, "_real_cuda_trainer", lambda trainer: real)
    monkeypatch.setattr(exit_release, "start_release", lambda device=None: events.append(("release",)))
    exit_release.patch_soup_exit_release()
    return Wrapper(), events


def _soup_tail(trainer):
    # what the run does after the last step: the checkpoint, Soup's final save, Soup's tokenizer save
    trainer.save_model("out/checkpoint-64", _internal_call=True)
    trainer.save_model("out")
    trainer.events.append(("tokenizer",))
    return "done"


def test_late_release_starts_after_the_final_save(monkeypatch):
    wrapper, events = _patched_wrapper(monkeypatch, "late", _soup_tail)
    assert wrapper.train() == "done"
    # not before or during the checkpoint, right after Soup's final save returned, once (the second
    # start at the train end is start_release's no-op in the real code)
    assert events[:4] == [("save", "out/checkpoint-64", True), ("save", "out", False), ("release",),
                          ("tokenizer",)]
    assert events[4:] == [("release",)]
    assert "save_model" not in wrapper.trainer.__dict__ and not wrapper.trainer.callbacks


def test_late_release_without_a_final_save_starts_at_train_end(monkeypatch):
    def body(trainer):
        trainer.save_model("out/checkpoint-64", _internal_call=True)
        trainer.events.append(("tokenizer",))

    wrapper, events = _patched_wrapper(monkeypatch, "late", body)
    wrapper.train()
    assert events == [("save", "out/checkpoint-64", True), ("tokenizer",), ("release",)]


def test_late_release_restores_save_model_when_train_raises(monkeypatch):
    def body(trainer):
        trainer.save_model("out/checkpoint-8", _internal_call=True)
        raise RuntimeError("boom")

    wrapper, events = _patched_wrapper(monkeypatch, "late", body)
    own = lambda *a, **k: None  # noqa: E731 - an instance attribute someone else set
    wrapper.trainer.save_model = own
    with pytest.raises(RuntimeError, match="boom"):
        wrapper.train()
    assert ("release",) not in events and wrapper.trainer.__dict__["save_model"] is own


def test_mode_1_keeps_the_last_step_callback(monkeypatch):
    wrapper, events = _patched_wrapper(monkeypatch, "1", _soup_tail)
    wrapper.train()
    assert len(wrapper.trainer.callbacks) == 1 and ("release",) not in events  # the callback starts it
    assert "save_model" not in wrapper.trainer.__dict__


def test_late_leaves_fake_trainers_alone(monkeypatch):
    wrapper, events = _patched_wrapper(monkeypatch, "late", _soup_tail, real=False)
    wrapper.train()
    assert ("release",) not in events and not wrapper.trainer.callbacks
