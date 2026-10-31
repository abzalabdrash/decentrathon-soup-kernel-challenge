"""CPU checks for pinning failure and optional stream diagnostics.

These exercise failure paths without reserving a real pinned slab or starting CUDA.
The existing stream tests cover actual copies and gradients on a GPU.
"""

import collections
import contextlib
import types
import weakref

import pytest
import torch


@pytest.mark.parametrize("error", [RuntimeError, MemoryError])
def test_host_stack_keeps_tensor_on_pinning_failure(monkeypatch, error):
    from solution import offload

    monkeypatch.setattr(offload, "_SPARE", [])
    monkeypatch.setattr(offload, "_PREPINNING", [0])
    calls = []

    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise error("page-locked allocation failed")

    monkeypatch.setattr(torch, "empty", fail)
    stack = offload._HostStack()
    assert stack.push(4096) == (None, None)
    assert stack.push(4096) == (None, None)  # do not retry every layer
    assert len(calls) == 1
    assert stack.nbytes == 0 and stack.slabs == [] and stack.slots == []


def test_prepin_stops_cleanly_on_pinning_failure(monkeypatch):
    from solution import offload

    monkeypatch.setattr(offload, "_SPARE", [])
    monkeypatch.setattr(offload, "_PREPINNING", [0])

    def fail(*args, **kwargs):
        raise RuntimeError("page-locked allocation failed")

    monkeypatch.setattr(torch, "empty", fail)
    thread = offload.prepin(512)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert offload._PREPINNING[0] == 0 and offload._SPARE == []


def test_registered_buffer_is_freed_before_pin_allocator_fallback(monkeypatch):
    from solution import weight_stream

    empty = torch.empty
    original = []

    def allocate(*args, **kwargs):
        if kwargs.pop("pin_memory", False):
            assert original[0]() is None  # no second full host copy at the fallback peak
        result = empty(*args, **kwargs)
        if not original:
            original.append(weakref.ref(result))
        return result

    runtime = types.SimpleNamespace(cudaHostRegister=lambda *args: 1,
                                    cudaError=types.SimpleNamespace(success=0))
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "cudart", lambda: runtime)
    buffer, finalizer = weight_stream.pinned_host_buffer(32, object())
    assert buffer.numel() == 32 and finalizer is None


def test_head_pin_failure_leaves_parameter_unchanged(monkeypatch):
    from solution import head_stream, weight_stream

    weight = torch.arange(32, dtype=torch.float16).view(16, 2)
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    before = param.data_ptr()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: object())

    def fail(*args, **kwargs):
        raise RuntimeError("page-locked allocation failed")

    monkeypatch.setattr(weight_stream, "pinned_host_buffer", fail)
    with pytest.raises(head_stream.HeadSkip, match="cannot pin the head weight"):
        head_stream.HeadStreamer(param, source=weight, device="cuda:0")
    assert param.data_ptr() == before and torch.equal(param, weight)
    assert head_stream.streamer_for(param) is None


class _Event:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def record(self, stream):
        self.log.append(("record", self.name, stream.name))


class _Stream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_event(self, event):
        self.log.append(("wait", event.name, self.name))


def _streams(monkeypatch):
    log = []
    compute, side = _Stream(log, "compute"), _Stream(log, "copy")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: compute)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    return log, side, [_Event(log, "free")], [_Event(log, "ready")]


def _no_timing_event():
    raise AssertionError("disabled diagnostics allocated a timing event")


def test_weight_timing_off_keeps_dependency_events(monkeypatch):
    from solution import weight_stream

    log, side, free, ready = _streams(monkeypatch)
    streamer = weight_stream.WeightStreamer.__new__(weight_stream.WeightStreamer)
    dst, source = torch.zeros(8, dtype=torch.uint8), torch.arange(8, dtype=torch.uint8)
    streamer.device, streamer._copy_stream = torch.device("cpu"), side
    streamer._free, streamer._ready = free, ready
    streamer._copies = [[(dst, source)]]
    streamer._groups = [types.SimpleNamespace(first=0, count=1)]
    streamer._owner, streamer.group_loads, streamer.loads = [-1], [0], 0
    streamer.timing = streamer.matrix_timing = False
    streamer._spans, streamer._event = collections.deque(), _no_timing_event
    streamer._load(0)
    streamer._wait(0, 1)
    assert torch.equal(dst, source) and streamer.loads == 1 and streamer._owner == [0]
    assert log == [("record", "free", "compute"), ("wait", "free", "copy"),
                   ("record", "ready", "copy"), ("wait", "ready", "compute")]
    assert not streamer._spans


def test_head_timing_off_keeps_dependency_events(monkeypatch):
    from solution import head_stream

    log, side, free, ready = _streams(monkeypatch)
    streamer = head_stream.HeadStreamer.__new__(head_stream.HeadStreamer)
    streamer.device, streamer._copy_stream = torch.device("cpu"), side
    streamer._free, streamer._ready = free, ready
    streamer.slots, streamer.rows, streamer.vocab, streamer.hidden = 1, 4, 4, 2
    streamer.host = torch.arange(8, dtype=torch.float16).view(4, 2)
    streamer._ring = torch.zeros(1, 4, 2, dtype=torch.float16)
    streamer._owner, streamer.loads, streamer.late_loads = [-1], 0, 0
    streamer.timing = False
    streamer._spans, streamer._event = collections.deque(), _no_timing_event
    streamer._load(0)
    assert torch.equal(streamer._acquire(0), streamer.host)
    assert log == [("record", "free", "compute"), ("wait", "free", "copy"),
                   ("record", "ready", "copy"), ("wait", "ready", "compute")]
    assert not streamer._spans


def test_stream_timing_switch_defaults_to_existing_path(monkeypatch):
    from solution.weight_stream import stream_timing_enabled

    monkeypatch.delenv("SOUP_SOLUTION_STREAM_TIMING", raising=False)
    assert stream_timing_enabled()
    monkeypatch.setenv("SOUP_SOLUTION_STREAM_TIMING", "0")
    assert not stream_timing_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_head_timing_switch_preserves_outputs_bitwise():
    from solution.head_stream import HeadStreamer

    generator = torch.Generator().manual_seed(37)
    param = torch.nn.Parameter(torch.randn(48, 16, generator=generator).half().cuda(),
                               requires_grad=False)
    streamer = HeadStreamer(param, rows=16)
    hidden = torch.randn(5, 16, generator=generator).half().cuda()
    targets = torch.tensor([1, 17, 47, -100, -100], device="cuda")
    try:
        streamer.timing = True
        before = streamer.head(hidden, targets, 3, 1.0 / 3)
        streamer.timing = False
        after = streamer.head(hidden, targets, 3, 1.0 / 3)
        assert all(torch.equal(a, b) for a, b in zip(before, after))
    finally:
        streamer.restore()
