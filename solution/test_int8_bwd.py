"""int8_bwd.py (night task N2): the int8 kernels against plain torch, and the fused MLP's backward on
int8 tensor cores against its fp16 backward at the 1.5B shapes (quantization noise only)."""

import pytest
import torch

from solution.test_fused_proj import _config, _linear4bit, _quantize_mlp, _rel, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def int8_on():
    from solution import int8_bwd

    saved = int8_bwd._STATE["on"]
    int8_bwd._STATE["on"] = True
    int8_bwd.compile_now()
    yield int8_bwd
    int8_bwd._STATE["on"] = saved


def _round(x):
    return torch.floor(x + 0.5).clamp_(-127, 127)


def test_nf4_int8_transposed_decode(int8_on):
    from solution import nf4

    layer = _linear4bit(1536, 8960, seed=4)  # (I, D) like gate
    w16 = nf4.dequantize_nf4(layer.weight, layer.weight.quant_state, torch.float16)
    colmax = w16.abs().amax(0).float()
    inv = int8_on._inverse(colmax)
    out = torch.empty((1536, 2 * 8960), dtype=torch.int8, device="cuda")
    int8_on._decode_t(layer, inv, out, 2 * 8960, 8960)
    ref = _round(w16.float() * inv[None, :]).to(torch.int8).t()
    got = out[:, 8960:]
    diff = (got.int() - ref.int()).abs()
    # the kernel scales the fp32 decode, the reference the fp16-rounded weight: a value within the
    # fp16 rounding of a .5 boundary lands one step apart (0.45% of them on the T4)
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-2


def test_row_quant(int8_on):
    g = torch.randn(832, 1536, device="cuda").half()
    g[-16:] = 0  # padding rows
    q, s = int8_on._quant_rows(g)
    amax = g.float().abs().amax(1)
    assert torch.allclose(s, amax / 127.0, rtol=1e-6, atol=0)
    ref = _round(g.float() * torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))[:, None])
    assert torch.equal(q.float(), ref)
    assert torch.all(q[-16:] == 0) and torch.all(s[-16:] == 0)


def test_mlp_backward_int8_close_to_fp16(int8_on, monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()
    gen = torch.Generator("cuda").manual_seed(7)
    x = torch.randn(1, 832, 1536, device="cuda", generator=gen)
    grad = (torch.randn(1, 832, 1536, device="cuda", generator=gen) * 1e-2).half()
    results = []
    for on in (False, True):
        int8_on._STATE["on"] = on
        xg = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(xg)
        out.backward(grad.to(out.dtype))
        results.append((out.detach(), xg.grad))
    (out16, dx16), (out8, dx8) = results
    assert torch.equal(out16, out8)  # the forward does not change
    err = _rel(dx8, dx16)
    print(f"\nint8 MLP backward: relative error of dL/dx against fp16 {err:.4f}")
    assert err < 0.05


def test_takeover_after_start_calls(int8_on, monkeypatch):
    """The first START_CALLS eligible backwards stay fp16 while a thread compiles the kernels; the
    next one waits for it, so the switch point does not depend on timing."""
    from types import SimpleNamespace

    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=21).cuda()
    plan = SimpleNamespace(gate=mlp.gate_proj, up=mlp.up_proj, down=mlp.down_proj,
                           half=torch.float16)
    monkeypatch.setattr(int8_on, "START_CALLS", 3)
    monkeypatch.setitem(int8_on._STATE, "calls", 0)
    monkeypatch.setitem(int8_on._STATE, "thread", None)
    ready = int8_on._READY.is_set()
    int8_on._READY.clear()
    try:
        assert not int8_on.usable(plan, 20)  # tokens % 8: never counted
        assert [int8_on.usable(plan, 832) for _ in range(5)] == [False] * 3 + [True] * 2
        assert int8_on._STATE["calls"] == 5 and not int8_on._STATE["thread"].is_alive()
    finally:
        if ready:
            int8_on._READY.set()


def test_decode_pair_matches_two_decodes(int8_on, monkeypatch):
    """nf4_int8_t2_kernel (W_gate^T | W_up^T in one launch, from fused_proj's cached operands) writes
    what two nf4_int8_t_kernel launches write, bit for bit."""
    gate, up = _linear4bit(1536, 8960, seed=11), _linear4bit(1536, 8960, seed=12)
    w16 = torch.cat([int8_on.nf4.dequantize_nf4(m.weight, m.weight.quant_state, torch.float16)
                     for m in (gate, up)])
    inv = int8_on._inverse(w16.abs().amax(0).float())
    pair = torch.empty((1536, 2 * 8960), dtype=torch.int8, device="cuda")
    int8_on._decode_t_pair(gate, up, inv, pair, 2 * 8960, torch.float16)
    monkeypatch.setitem(int8_on._PAIR, "on", False)
    two = torch.empty_like(pair)
    int8_on._decode_t(gate, inv, two, 2 * 8960, 0)
    int8_on._decode_t(up, inv, two, 2 * 8960, 8960)
    assert torch.equal(pair, two)
