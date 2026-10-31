"""Pinned host memory running out (A1 jury audit; tests from codex/review-1, 611cade).

The run holds ~3.6 GB of page-locked host memory (offload slabs, the layers' 4-bit weights, the
fp16 head) and pins more offload slabs on longer steps. If the driver refuses, the run must go on
with the tensors on the GPU instead of failing (check 1). CPU only: the pinning is stubbed to fail.
"""

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




def test_load_embedding_keeps_values_when_pinning_fails(monkeypatch):
    """The loader's path (load_stream.py -> head_stream.load_embedding): the pinner thread made no
    buffer and pinning here fails too. Before the HeadSkip conversion the RuntimeError went up
    through transformers' set_param_for_module and failed the model load."""
    from solution import head_stream, weight_stream

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(16, 2)
            self.head = torch.nn.Linear(2, 16, bias=False)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    def fail(*args, **kwargs):
        raise RuntimeError("page-locked allocation failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", lambda *args, **kwargs: object())
    monkeypatch.setattr(weight_stream, "pinned_host_buffer", fail)
    value = torch.arange(32, dtype=torch.float16).view(16, 2)
    logged = []
    out, nbytes = head_stream.load_embedding(Model(), value, "cpu", log=logged.append)
    assert nbytes == 0 and torch.equal(out, value)
    assert any("cannot pin the head weight" in line for line in logged)
