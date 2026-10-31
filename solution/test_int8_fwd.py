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


@pytest.mark.parametrize("chunks", [1, 2, 5])
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


def test_decode_gate_up_topk_columns(int8_on):
    """The extra programs of the gate|up decode: the fp32 decode of the top-k columns, in the int8
    rows' order; the int8 rows and scales are those of the plain decode."""
    gate, up = _linear4bit(1536, 8960, seed=4), _linear4bit(1536, 8960, seed=5)
    idx = torch.tensor([0, 7, 63, 64, 100, 511, 512, 777, 1000, 1023, 1024, 1200, 1300, 1400, 1501,
                        1535], dtype=torch.int32, device="cuda")
    out = torch.empty((2 * 8960, 1536), dtype=torch.int8, device="cuda")
    scales = torch.empty(2 * 8960, dtype=torch.float32, device="cuda")
    w_k = torch.empty((2 * 8960, 16), dtype=torch.float32, device="cuda")
    int8_on._decode_gate_up(_ops(gate), _ops(up), out, scales, 8960, 1536, 8960, idx, w_k)
    ref = torch.cat([_decoded(gate), _decoded(up)])
    assert torch.allclose(w_k, ref[:, idx.long()], rtol=1e-6, atol=0)  # fp32 ulps of the absmax
    out0, scales0 = torch.empty_like(out), torch.empty_like(scales)
    int8_on._decode_gate_up(_ops(gate), _ops(up), out0, scales0, 8960, 1536, 8960)
    assert torch.equal(out, out0) and torch.equal(scales, scales0)


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
    x_q, s_x, _ = int8_on.quantize_tokens(x, 830, 832)
    amax = x.float().abs().amax(1)
    assert torch.allclose(s_x[:830], amax / 127.0, rtol=1e-6, atol=0)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    diff = (x_q[:830].float() - _round(x.float() * inv[:, None])).abs()
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-4
    assert torch.all(x_q[830:] == 0) and torch.all(s_x[830:] == 0) and torch.all(x_q[5] == 0)
    # with top-k channels: those go to x_k in fp32 and out of the int8 rows and their scale
    idx = torch.tensor([3, 100, 1535] + list(range(200, 213)), device="cuda")
    pos = torch.full((1536,), -1, dtype=torch.int32, device="cuda")
    pos[idx] = torch.arange(16, dtype=torch.int32, device="cuda")
    x_q, s_x, x_k = int8_on.quantize_tokens(x, 830, 832, pos)
    assert torch.equal(x_k[:830], x.float()[:, idx]) and torch.all(x_k[830:] == 0)
    rest = x.float().clone()
    rest[:, idx] = 0
    amax = rest.abs().amax(1)
    assert torch.allclose(s_x[:830], amax / 127.0, rtol=1e-6, atol=0)
    assert torch.all(x_q[:, idx] == 0)


@pytest.mark.parametrize("chunks", [1, 2, 5])
def test_mlp_forward_kernels_match_torch(int8_on, monkeypatch, chunks):
    """gate_up_forward / down_forward against the same int8 math in torch (float64 products of the
    same int8 operands): exact up to the fp32 epilogue and the half roundings."""
    monkeypatch.setattr(int8_on, "CHUNKS", chunks)
    gate, up, down = (_linear4bit(1536, 8960, seed=7), _linear4bit(1536, 8960, seed=8),
                      _linear4bit(8960, 1536, seed=9))
    x = torch.randn(830, 1536, device="cuda")
    buffer = torch.empty(8960 * 1536, dtype=torch.float16, device="cuda")
    x_q, s_x, _ = int8_on.quantize_tokens(x, 830, 832)
    gate_up, pmax = int8_on.gate_up_forward(x_q, s_x, _ops(gate), _ops(up), buffer, torch.float16,
                                            8960)
    wq, ws = _reference_rows(torch.cat([_decoded(gate), _decoded(up)]))
    acc = wq.double() @ x_q.double().t()
    ref_gu = (acc.float() * ws[:, None] * s_x[None, :]).half()
    assert _rel(gate_up, ref_gu) < 1e-3
    g, u = gate_up[:8960].float(), gate_up[8960:].float()
    h = g * torch.sigmoid(g) * u
    assert torch.allclose(pmax.amax(0), h.abs().amax(0), rtol=1e-5, atol=1e-7)
    # with top-k channels: the int8 product of the other channels + those in fp32
    idx = torch.arange(16, dtype=torch.int32, device="cuda") * 96 + 5
    pos = torch.full((1536,), -1, dtype=torch.int32, device="cuda")
    pos[idx.long()] = torch.arange(16, dtype=torch.int32, device="cuda")
    xq2, sx2, x_k = int8_on.quantize_tokens(x, 830, 832, pos)
    gu2, _ = int8_on.gate_up_forward(xq2, sx2, _ops(gate), _ops(up), buffer, torch.float16, 8960,
                                     idx=idx, x_k=x_k)
    w32 = torch.cat([_decoded(gate), _decoded(up)])
    ref2 = ((wq.double() @ xq2.double().t()).float() * ws[:, None] * sx2[None, :]
            + w32[:, idx.long()] @ x_k.t()).half()
    assert _rel(gu2, ref2) < 1e-3
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


def test_off_by_default():
    """Off in every process unless SOUP_SOLUTION_INT8_FWD=1 (the conftest pins it off per test)."""
    import os
    import subprocess
    import sys

    code = "from solution import int8_fwd; print(int8_fwd._STATE['on'])"
    env = {k: v for k, v in os.environ.items() if k != "SOUP_SOLUTION_INT8_FWD"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=env)
    on = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                        env={**env, "SOUP_SOLUTION_INT8_FWD": "1"})
    assert out.stdout.strip() == "False" and on.stdout.strip() == "True"


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


def test_drain_after_first_product(int8_on, monkeypatch):
    """With the early offload, the int8 forward lets the copies in flight finish right before its
    (gate|up) is allocated (after the decode and the first int8 product), once; the fp16 forward
    before anything of its own."""
    from solution import offload
    from transformers.models.qwen2 import modeling_qwen2 as mq

    calls = []
    monkeypatch.setattr(offload, "EARLY", True)
    monkeypatch.setattr(offload, "drain_inflight", lambda: calls.append(int8_on._STATE["calls"]))
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=24).cuda().train()
    x = torch.randn(1, 832, 1536, device="cuda", requires_grad=True)
    seen = []
    real = int8_on.gate_up_forward

    def spy(*args, before_alloc=None, **kwargs):
        seen.append(before_alloc is not None)
        return real(*args, before_alloc=before_alloc, **kwargs)

    monkeypatch.setattr(int8_on, "gate_up_forward", spy)
    with torch.autocast("cuda", dtype=torch.float16):
        mlp(x)
    assert seen == [True] and len(calls) == 1
    int8_on._STATE["on"] = False
    with torch.autocast("cuda", dtype=torch.float16):
        mlp(x)
    assert seen == [True] and len(calls) == 2


def test_channels_from_the_fp16_forwards(int8_on, monkeypatch):
    """Before the switch the training forwards (input with requires_grad) collect max |x| per input
    channel; the first int8 forward fixes the top-k channels from it, for the rest of the run."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=25).cuda().train()
    monkeypatch.setattr(int8_on, "START_CALLS", 2)
    monkeypatch.setitem(int8_on._STATE, "calls", 0)
    xs = [torch.randn(1, 832, 1536, device="cuda") for _ in range(3)]
    xs[0][0, :, 7] *= 50  # channel 7 large in the first forward, channel 9 in the second
    xs[1][0, :, 9] *= 40
    for x in xs:
        with torch.autocast("cuda", dtype=torch.float16):
            mlp(x.requires_grad_(True)).sum().backward()
    pos, idx = mlp.__dict__["_soup_int8_topk"]
    assert int8_on._STATE["calls"] == 3 and "_soup_int8_peak" not in mlp.__dict__
    peak = torch.maximum(xs[0].detach().abs().amax((0, 1)), xs[1].detach().abs().amax((0, 1)))
    assert set(idx.tolist()) == set(peak.topk(int8_on.TOPK).indices.tolist())
    assert {7, 9} <= set(idx.tolist()) and int(pos[7]) >= 0 and int(pos[9]) >= 0


def test_epilogue_writes_the_int8_offload_form(int8_on):
    """With wire_for, the gate|up epilogue writes offload_int8.py's int8 form (the int16 view of the
    gate half, the per-token per-GROUP scales) of exactly the half values it writes otherwise, and the
    same SwiGLU output: what offload_int8.encode makes from the half (gate|up), bit for bit."""
    from solution import offload_int8

    gate, up = _linear4bit(1536, 8960, seed=26), _linear4bit(1536, 8960, seed=27)
    x = torch.randn(830, 1536, device="cuda")
    buffer = torch.empty(8960 * 1536, dtype=torch.float16, device="cuda")
    idx = torch.arange(16, dtype=torch.int32, device="cuda") * 90 + 3
    pos = torch.full((1536,), -1, dtype=torch.int32, device="cuda")
    pos[idx.long()] = torch.arange(16, dtype=torch.int32, device="cuda")
    x_q, s_x, x_k = int8_on.quantize_tokens(x, 830, 832, pos)
    gate_up, hidden = int8_on.gate_up_forward(x_q, s_x, _ops(gate), _ops(up), buffer,
                                              torch.float16, 8960, hidden=True, idx=idx, x_k=x_k)
    ref_wire, _ = offload_int8.encode(gate_up)
    ref_q, ref_s = offload_int8.views(ref_wire, *gate_up.shape)

    def wire_for(gu):
        group = offload_int8.GROUP
        return gu[:8960].view(torch.int16), torch.empty((2 * 8960 // group, 832),
                                                        dtype=torch.float32, device="cuda")

    made = int8_on.gate_up_forward(x_q, s_x, _ops(gate), _ops(up), buffer, torch.float16, 8960,
                                   hidden=True, idx=idx, x_k=x_k, wire_for=wire_for)
    assert len(made) == 3
    _, hidden2, (q, s) = made
    assert torch.equal(hidden2, hidden) and torch.equal(q, ref_q) and torch.equal(s, ref_s)


def test_int8_forward_with_the_int8_offload(int8_on, monkeypatch):
    """The training path (offload_saved_tensors, early offload) of three fused MLPs: with the int8
    forward and offload_int8 on, every layer's (gate|up) crosses as the int8 form that the int8
    epilogue wrote itself (reserve_int8, then offload_early(wire=...)); the forward is bit for bit
    the one with the half offload, and the input gradient moves from the half offload's as much as it
    does on the fp16 forward with offload_int8's own quantization (the same int8 form), for both MLP
    backwards."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import int8_bwd, offload, offload_int8

    monkeypatch.setattr(offload, "EARLY", True)
    monkeypatch.setitem(offload_int8._STATE, "on", True)
    offload_int8.compile_now()
    int8_bwd.compile_now()
    mlps = [_quantize_mlp(mq.Qwen2MLP(_config()), seed=30 + 3 * i).cuda().train() for i in range(3)]
    x0 = torch.randn(1, 826, 1536, device="cuda")
    grad = torch.randn_like(x0) * 1e-2

    def go(fwd_int8, wire, bwd_int8):
        monkeypatch.setitem(int8_on._STATE, "on", fwd_int8)
        monkeypatch.setitem(offload_int8._STATE, "on", wire)
        monkeypatch.setitem(int8_bwd._STATE, "on", bwd_int8)
        calls = int8_on._STATE["calls"]
        x = x0.clone().requires_grad_(True)
        with offload.offload_saved_tensors(weights=[p for m in mlps for p in m.parameters()]) as off,                 torch.autocast("cuda", dtype=torch.float16):
            h = x
            for mlp in mlps:
                h = h + mlp(h)
            h.backward(grad)
        torch.cuda.synchronize()
        assert int8_on._STATE["calls"] == calls + (3 if fwd_int8 else 0)
        assert off.stats["int8"] == (3 if wire else 0) and off.stats["early"] == 3
        return h.detach(), x.grad

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    for bwd_int8 in (False, True):
        ref_out, ref_gx = go(True, False, bwd_int8)
        out, gx = go(True, True, bwd_int8)
        assert torch.equal(out, ref_out)
        _, half_gx = go(False, False, bwd_int8)
        _, p1_gx = go(False, True, bwd_int8)
        err, p1_err = rel(gx, ref_gx), rel(p1_gx, half_gx)
        print(f"\nint8 backward {bwd_int8}: dL/dx error of the int8 offload against the half one: "
              f"int8 forward {err:.4f}, fp16 forward {p1_err:.4f}")
        assert err < 1.2 * p1_err + 1e-3
