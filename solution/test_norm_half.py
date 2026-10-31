"""SOUP_SOLUTION_NORM_HALF (rmsnorm.py): the norm's second, half output is bit for bit the cast its
consumer made, `half_of` falls back to the cast when the output changed or does not match, and a
tiny 4-bit Qwen2 with LoRA through the fused path gives the same bits with and without it."""

import pytest
import torch

from solution import rmsnorm
from solution.test_fused_proj import _lora_params, _tiny_peft_model, _tiny_step, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def half_on():
    saved = rmsnorm._HALF["on"]
    rmsnorm._HALF["on"] = True
    yield
    rmsnorm._HALF["on"] = saved


@pytest.mark.parametrize("rows", [826, 31])
def test_half_output_is_the_cast(half_on, rows):
    g = torch.Generator("cuda").manual_seed(0)
    x = torch.randn(rows, 1536, device="cuda", generator=g) * 3
    branch = torch.randn(rows, 1536, device="cuda", generator=g).half()
    w = torch.rand(1536, device="cuda", generator=g) + 0.5
    with torch.autocast("cuda", dtype=torch.float16):
        y = rmsnorm.rmsnorm(x, w, 1e-6, half=True)
        total, y2 = rmsnorm.add_rmsnorm(x, branch, w, 1e-6, half=True)
        plain = rmsnorm.rmsnorm(x, w, 1e-6)  # not asked for (the final norm): no half copy
    assert not hasattr(plain, "_soup_half")
    for out in (y, y2):
        y16, version = out._soup_half
        assert y16.dtype == torch.float16 and version == out._version
        assert torch.equal(y16.view(torch.int16), out.to(torch.float16).view(torch.int16))
        assert rmsnorm.half_of(out, torch.float16) is y16
    rmsnorm._HALF["on"] = False
    with torch.autocast("cuda", dtype=torch.float16):
        y_off = rmsnorm.rmsnorm(x, w, 1e-6, half=True)
        total_off, y2_off = rmsnorm.add_rmsnorm(x, branch, w, 1e-6, half=True)
    assert not hasattr(y_off, "_soup_half") and not hasattr(y2_off, "_soup_half")
    assert torch.equal(y, y_off) and torch.equal(y2, y2_off) and torch.equal(total, total_off)


def test_half_of_falls_back(half_on):
    x = torch.randn(64, 1536, device="cuda")
    w = torch.ones(1536, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = rmsnorm.rmsnorm(x, w, 1e-6, half=True)
    y16 = y._soup_half[0]
    assert rmsnorm.half_of(y, torch.bfloat16).dtype == torch.bfloat16  # another dtype: the cast
    y.mul_(2)  # changed after the norm: its half copy is stale
    got = rmsnorm.half_of(y, torch.float16)
    assert got is not y16 and torch.equal(got, y.half())
    without = rmsnorm.rmsnorm(x, w, 1e-6, half=True)  # no autocast: no half copy
    assert not hasattr(without, "_soup_half")


def test_tiny_peft_model_same_bits(half_on):
    from solution import fused_proj, unpad

    splits = unpad._STATE["splits_key"]
    unpad._STATE["splits_key"] = 1  # dq in a fixed order (several key splits add it with atomics)
    try:
        model = _tiny_peft_model()
        lengths = [200, 151, 300, 177]
        ids = torch.randint(0, 512, (1, sum(lengths)), device="cuda",
                            generator=torch.Generator("cuda").manual_seed(2))
        fused_proj._STATE["enabled"] = True
        rmsnorm._HALF["on"] = False
        ref_hidden, ref_loss, ref_grads = _tiny_step(model, ids, lengths)
        rmsnorm._HALF["on"] = True
        taken = []
        original = rmsnorm.half_of

        def spy(x, half, cast=True):
            out = original(x, half, cast)
            taken.append(out is getattr(x, "_soup_half", (None,))[0])
            return out

        fused_proj.half_of = spy
        try:
            hidden, loss, grads = _tiny_step(model, ids, lengths)
        finally:
            fused_proj.half_of = original
        assert taken and all(taken)  # every fused consumer took the norm's half copy
        assert torch.equal(hidden, ref_hidden) and torch.equal(loss, ref_loss)
        assert len(grads) == len(ref_grads) == len(_lora_params(model))
        for g, ref in zip(grads, ref_grads):
            assert torch.equal(g, ref)
    finally:
        unpad._STATE["splits_key"] = splits
