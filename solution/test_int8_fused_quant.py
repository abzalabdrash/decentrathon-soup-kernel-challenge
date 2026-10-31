"""SOUP_SOLUTION_INT8_FUSED_QUANT: the int8 MLP backward's per-token quantization of the incoming gradient
and step 7's two scalings as own kernels give torch's int8 values, scales and fp32 results bit for bit."""

import pytest
import torch

from solution.test_fused_proj import _config, _quantize_mlp, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def int8_on(monkeypatch):
    from solution import int8_bwd

    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    int8_bwd.compile_now()
    return int8_bwd


@pytest.mark.parametrize("tokens", [64, 832, 1440])
def test_fused_quant_kernels_bitwise(int8_on, tokens):
    gen = torch.Generator("cuda").manual_seed(tokens)
    x = (torch.randn(tokens, 1536, device="cuda", generator=gen) * 3).half()
    x[-16:] = 0
    q_ref, s_ref = int8_on._quant_rows(x)
    q, s = int8_on._quant_rows_fused(x)
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref)
    acc = torch.randint(-2**24, 2**24, (tokens, 1536), device="cuda", dtype=torch.int32, generator=gen)
    a = torch.rand(tokens, device="cuda", generator=gen) * 50
    b = torch.rand(1536, device="cuda", generator=gen) * 1e-3
    ref = acc.float().mul_((a / 127.0)[:, None]).mul_(b[None, :])
    assert torch.equal(int8_on._scale_rows_cols(acc, a, b), ref)


def test_mlp_backward_same_with_fused_quant(int8_on, monkeypatch):
    from types import SimpleNamespace

    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=34).cuda()
    plan = SimpleNamespace(gate=mlp.gate_proj, up=mlp.up_proj, down=mlp.down_proj, half=torch.float16)
    gen = torch.Generator("cuda").manual_seed(6)
    gate_up = (torch.randn(2 * 8960, 832, device="cuda", generator=gen) * 2).half()
    grad16 = (torch.randn(832, 1536, device="cuda", generator=gen) * 1e-2).half()
    buffer = torch.empty(8960 * 1536, dtype=torch.float16, device="cuda")
    monkeypatch.setitem(int8_on._FUSED_QUANT, "on", False)
    ref = int8_on.mlp_backward(plan, gate_up.clone(), grad16, buffer)
    monkeypatch.setitem(int8_on._FUSED_QUANT, "on", True)
    assert torch.equal(int8_on.mlp_backward(plan, gate_up.clone(), grad16, buffer), ref)


def test_gate_up_scales_cached_once(int8_on):
    """The shared column scales of [W_gate^T | W_up^T] are made once per layer, with the values the
    per-step torch ops gave."""
    from types import SimpleNamespace

    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=35).cuda()
    plan = SimpleNamespace(gate=mlp.gate_proj, up=mlp.up_proj, down=mlp.down_proj, half=torch.float16)
    buffer = torch.empty(8960 * 1536, dtype=torch.float16, device="cuda")
    inv_g, scale_g = int8_on._column_scales(plan.gate, plan.half, buffer)
    inv_u, scale_u = int8_on._column_scales(plan.up, plan.half, buffer)
    scale_ref = torch.maximum(scale_g, scale_u)
    inv_ref = int8_on._inverse(scale_ref * 127.0)
    first = int8_on._gate_up_scales(plan, buffer)
    assert int8_on._gate_up_scales(plan, buffer) is first
    assert torch.equal(first[0], inv_ref) and torch.equal(first[1], scale_ref)
