"""int8_fwd.py (task I1): the int8 kernels against plain torch, and the fused MLP's forward on int8
tensor cores against its fp16 forward at the 1.5B shapes (quantization noise only)."""

import pytest
import torch

from solution.test_fused_proj import _config, _linear4bit, _quantize_mlp, _rel, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def int8_on():
    from solution import int8_fwd

    saved = int8_fwd._STATE["on"]
    int8_fwd._STATE["on"] = True
    int8_fwd.compile_now()
    yield int8_fwd
    int8_fwd._STATE["on"] = saved


def _round(x):
    return torch.floor(x + 0.5).clamp_(-127, 127)


def _ops(layer):
    from solution.fused_proj import _nf4_args

    return _nf4_args(layer, torch.float16)[1]


def _decoded(layer):
    """The fp32 NF4 decode of a layer in torch: level * block absmax (with double quant), as the
    kernels compute it before any rounding (bitsandbytes' own dequantize rounds to fp16)."""
    import bitsandbytes.functional as F

    state = layer.weight.quant_state
    absmax = F.dequantize_blockwise(state.absmax, state.state2) + state.offset
    packed = layer.weight.data.view(-1)
    codes = torch.stack([packed >> 4, packed & 15], dim=1).view(-1).long()
    vals = state.code[codes].view(-1, 64) * absmax[:, None]
    return vals.view(*state.shape)


def _reference_rows(w32):
    rowmax = w32.abs().amax(1)
    inv = torch.where(rowmax > 0, 127.0 / rowmax, torch.zeros_like(rowmax))
    return _round(w32 * inv[:, None]), rowmax / 127.0


@pytest.mark.parametrize("chunks", [1, 2])
def test_decode_gate_up(int8_on, monkeypatch, chunks):
    gate, up = _linear4bit(1536, 8960, seed=4), _linear4bit(1536, 8960, seed=5)
    monkeypatch.setattr(int8_on, "CHUNKS", chunks)
    chunk = 8960 // chunks
    out = torch.empty((2 * 8960, 1536), dtype=torch.int8, device="cuda")
    scales = torch.empty(2 * 8960, dtype=torch.float32, device="cuda")
    int8_on._decode_gate_up(_ops(gate), _ops(up), out, scales, 8960, 1536, chunk)
    order = torch.cat([torch.cat([torch.arange(k * chunk, (k + 1) * chunk),
                                  8960 + torch.arange(k * chunk, (k + 1) * chunk)])
                       for k in range(chunks)]).cuda()
    ref_q, ref_s = _reference_rows(torch.cat([_decoded(gate), _decoded(up)]))
    # the row scale comes from the block absmax: the decoded row maximum, every block has a +-1 code
    assert torch.allclose(scales, ref_s[order], rtol=1e-6, atol=0)
    diff = (out.float() - ref_q[order]).abs()
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-4


def test_decode_down(int8_on):
    down = _linear4bit(8960, 1536, seed=6)
    out = torch.empty((1536, 8960), dtype=torch.int8, device="cuda")
    scales = torch.empty(1536, dtype=torch.float32, device="cuda")
    int8_on._decode_down(_ops(down), out, scales, 1536, 8960)
    ref_q, ref_s = _reference_rows(_decoded(down))
    assert torch.allclose(scales, ref_s, rtol=1e-6, atol=0)
    diff = (out.float() - ref_q).abs()
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-4


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_quantize_tokens(int8_on, dtype):
    x = torch.randn(830, 1536, device="cuda").to(dtype)
    x[5] = 0
    x_q, s_x = int8_on.quantize_tokens(x, 830, 832)
    amax = x.float().abs().amax(1)
    assert torch.allclose(s_x[:830], amax / 127.0, rtol=1e-6, atol=0)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    diff = (x_q[:830].float() - _round(x.float() * inv[:, None])).abs()
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-4
    assert torch.all(x_q[830:] == 0) and torch.all(s_x[830:] == 0) and torch.all(x_q[5] == 0)


@pytest.mark.parametrize("chunks", [1, 2])
def test_mlp_forward_kernels_match_torch(int8_on, monkeypatch, chunks):
    """gate_up_forward / down_forward against the same int8 math in torch (float64 products of the
    same int8 operands): exact up to the fp32 epilogue and the half roundings."""
    monkeypatch.setattr(int8_on, "CHUNKS", chunks)
    gate, up, down = (_linear4bit(1536, 8960, seed=7), _linear4bit(1536, 8960, seed=8),
                      _linear4bit(8960, 1536, seed=9))
    x = torch.randn(830, 1536, device="cuda")
    buffer = torch.empty(8960 * 1536, dtype=torch.float16, device="cuda")
    x_q, s_x = int8_on.quantize_tokens(x, 830, 832)
    gate_up, pmax = int8_on.gate_up_forward(x_q, s_x, _ops(gate), _ops(up), buffer, torch.float16,
                                            8960)
    wq, ws = _reference_rows(torch.cat([_decoded(gate), _decoded(up)]))
    acc = wq.double() @ x_q.double().t()
    ref_gu = (acc.float() * ws[:, None] * s_x[None, :]).half()
    assert _rel(gate_up, ref_gu) < 1e-3
    g, u = gate_up[:8960].float(), gate_up[8960:].float()
    h = g * torch.sigmoid(g) * u
    assert torch.allclose(pmax.amax(0), h.abs().amax(0), rtol=1e-5, atol=1e-7)
    out = int8_on.down_forward(gate_up, pmax, _ops(down), buffer, torch.float16, 1536)
    amax = h.abs().amax(0)
    hq = _round(h.t() * torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))[:, None])
    dq, ds = _reference_rows(_decoded(down))
    ref = ((hq.double() @ dq.double().t()).float() * (amax / 127.0)[:, None] * ds[None, :]).half()
    assert _rel(out, ref) < 2e-3
    assert torch.all(out[830:] == 0) and torch.all(gate_up[:, 830:] == 0)


def _run(mlp, x, grad, int8, on):
    int8._STATE["on"] = on
    xg = x.clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = mlp(xg)
    out.backward(grad.to(out.dtype))
    return out.detach().float(), xg.grad


@pytest.mark.parametrize("n", [830, 1440])
@pytest.mark.parametrize("mode,bound", [("all", 0.04), ("gu", 0.02)])
def test_mlp_forward_int8_close_to_fp16(int8_on, monkeypatch, n, mode, bound):
    """Gaussian inputs and weights: per-token x per-row int8 noise (on the real model's activations
    see bench_int8_fwd.py --errors); the int8 SwiGLU output of mode "all" is the heavy-tailed part."""
    from solution import int8_bwd
    from transformers.models.qwen2 import modeling_qwen2 as mq

    monkeypatch.setitem(int8_bwd._STATE, "on", False)
    monkeypatch.setattr(int8_on, "MODE", mode)
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()
    gen = torch.Generator("cuda").manual_seed(n)
    x = torch.randn(1, n, 1536, device="cuda", generator=gen)
    grad = (torch.randn(1, n, 1536, device="cuda", generator=gen) * 1e-2).half()
    out16, dx16 = _run(mlp, x, grad, int8_on, False)
    calls = int8_on._STATE["calls"]
    out8, dx8 = _run(mlp, x, grad, int8_on, True)
    assert int8_on._STATE["calls"] == calls + 1  # the int8 path ran
    err_out, err_dx = _rel(out8, out16), _rel(dx8, dx16)
    print(f"\nint8 MLP forward ({mode}) at {n} tokens: relative error of the output {err_out:.4f}, "
          f"of dL/dx {err_dx:.4f}")
    assert err_out < bound and err_dx < bound


def test_no_grad_forward_stays_fp16(int8_on):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=22).cuda()
    x = torch.randn(1, 64, 1536, device="cuda")
    calls = int8_on._STATE["calls"]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out8 = mlp(x)
    int8_on._STATE["on"] = False
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out16 = mlp(x)
    assert int8_on._STATE["calls"] == calls and torch.equal(out8, out16)


def test_takeover_after_start_calls(int8_on, monkeypatch):
    """The first START_CALLS eligible forwards stay fp16 while a thread compiles the kernels; the
    next one waits for it, so the switch point does not depend on timing."""
    monkeypatch.setattr(int8_on, "START_CALLS", 3)
    monkeypatch.setitem(int8_on._STATE, "calls", 0)
    monkeypatch.setitem(int8_on._STATE, "thread", None)
    ready = int8_on._READY.is_set()
    int8_on._READY.clear()
    try:
        assert not int8_on.usable(20)  # tokens % 8: never counted
        assert [int8_on.usable(832) for _ in range(5)] == [False] * 3 + [True] * 2
        assert int8_on._STATE["calls"] == 5 and not int8_on._STATE["thread"].is_alive()
    finally:
        if ready:
            int8_on._READY.set()


def test_off_by_default_under_pytest():
    import os
    import subprocess
    import sys

    code = "from solution import int8_fwd; print(int8_fwd._STATE['on'])"
    env = {k: v for k, v in os.environ.items() if k != "SOUP_SOLUTION_INT8_FWD"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=env)
    assert out.stdout.strip() == "False"


def test_fp16_layers_counted_and_kept(int8_on, monkeypatch):
    """A layer of FP16_LAYERS (the decoder layer's index on its MLP) counts toward START_CALLS and
    runs the fp16 forward bit for bit; an untagged MLP counts as listed."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=23).cuda().train()
    x = torch.randn(1, 832, 1536, device="cuda")
    monkeypatch.setattr(int8_on, "FP16_LAYERS", frozenset({3}))
    outs = []
    for idx, on in ((3, True), (None, True), (5, True), (5, False)):
        if idx is None:
            mlp.__dict__.pop("_soup_layer_idx", None)
        else:
            mlp.__dict__["_soup_layer_idx"] = idx
        int8_on._STATE["on"] = on
        calls = int8_on._STATE["calls"]
        with torch.autocast("cuda", dtype=torch.float16):
            outs.append(mlp(x.clone().requires_grad_(True)).detach())
        assert int8_on._STATE["calls"] == calls + (1 if on else 0)
    listed, untagged, int8, fp16 = outs
    assert torch.equal(listed, fp16) and torch.equal(untagged, fp16) and not torch.equal(int8, fp16)
    assert _layers_parse()


def _layers_parse():
    from solution.int8_fwd import _layers

    return (_layers("0-4") == {0, 1, 2, 3, 4} and _layers("1.2.4") == {1, 2, 4}
            and _layers("") == frozenset() and _layers("2,26") == {2, 26})
