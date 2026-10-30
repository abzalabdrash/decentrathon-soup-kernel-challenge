"""fused_proj._DequantPipe (SOUP_SOLUTION_DEQUANT_PIPE): the next weight dequantized on a side stream
gives the same step bit for bit (dequantization and GEMMs unchanged, only the scheduling), the
guesses are hit from the second step on, and the slot rule keeps prefetches inside a layer."""

import pytest
import torch

from solution.test_fused_proj import _lora_params, _tiny_peft_model, _tiny_step, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENGTHS = [200, 151, 300, 177]


@pytest.fixture
def pipe():
    from solution import fused_proj

    saved = fused_proj._PIPE
    fused_proj._PIPE = fused_proj._DequantPipe()
    yield fused_proj
    fused_proj._PIPE = saved


def test_pipe_gives_the_same_step(pipe):
    model = _tiny_peft_model()
    ids = torch.randint(0, 512, (1, sum(LENGTHS)), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(2))
    pipe._PIPE.on = False
    torch.manual_seed(3)
    ref = _tiny_step(model, ids, LENGTHS)
    pipe._PIPE.on = True
    for _ in range(3):
        torch.manual_seed(3)
        got = _tiny_step(model, ids, LENGTHS)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
        for a, b in zip(got[2], ref[2]):
            assert torch.allclose(a, b, rtol=1e-3, atol=1e-6)  # attention's dq atomics only
    stats = pipe._PIPE.stats
    assert stats["prefetches"] > 0 and stats["hits"] > 0
    print(f"\npipe: {stats}")


def test_slot_rule_blocks_other_layers(pipe, monkeypatch):
    from solution import weight_stream

    model = _tiny_peft_model()
    ids = torch.randint(0, 512, (1, sum(LENGTHS)), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(2))
    layers = model.base_model.model.model.layers
    owner = {}
    for i, layer in enumerate(layers):
        for m in layer.modules():
            if hasattr(m, "weight") and hasattr(m.weight, "quant_state"):
                owner[m.weight.data.data_ptr()] = i
    # every layer in a "slot" of its own: prefetches must never cross from one layer to the next
    monkeypatch.setattr(weight_stream, "slot_of", lambda t: ("fake", owner.get(t.data_ptr())))
    pipe._PIPE.on = True
    crossed = []
    original = pipe._dequant_into

    def watch(bases, dtype, w):
        layers_of = {owner.get(b.weight.data.data_ptr()) for b in bases}
        crossed.append(len(layers_of))
        return original(bases, dtype, w)

    monkeypatch.setattr(pipe, "_dequant_into", watch)
    for _ in range(2):
        _tiny_step(model, ids, LENGTHS)
    assert crossed and max(crossed) == 1  # no single dequantization spans two layers
    assert pipe._PIPE.stats["prefetches"] > 0
