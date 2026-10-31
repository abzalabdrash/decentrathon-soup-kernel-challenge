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
