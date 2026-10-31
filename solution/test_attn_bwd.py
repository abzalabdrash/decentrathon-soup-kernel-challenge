"""The attention backward's math path (attn_bwd.py, SOUP_SOLUTION_ATTN_BWD=math): gradients against an
fp32 reference and against aten's memory-efficient backward (unpad.py's path), the alignment tail's
zero rows, the wiring through unpad's _VarlenAttention inside `varlen` and its fallbacks (bitwise the
original backward), the scratch in fused_proj's shared buffer, the guards and the warm-up."""

import pytest
import torch

from solution import attn_bwd, unpad

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

H, G, D = 12, 2, 128
SCALE = D ** -0.5
PATTERNS = {
    "4 rows of ~200 and the tail": [208, 204, 196, 188],
    "uneven, no tail": [250, 175, 125, 90],
    "one long, short ones": [256, 17, 1, 70],
    "one row": [160],
}
# first fit, longest first, into bins of the longest row rounded up to 16
BINS = {"4 rows of ~200 and the tail": 4, "uneven, no tail": 3, "one long, short ones": 2, "one row": 1}


@pytest.fixture
def math_on():
    saved = dict(attn_bwd._STATE)
    attn_bwd._STATE.update(mode="math", min_tokens=0, max_len=256, scratch="own", error=None)
    yield attn_bwd._STATE
    attn_bwd._STATE.clear()
    attn_bwd._STATE.update(saved)


def _batch(lengths, seed=0, tail=True):
    n_real = sum(lengths)
    pad = (-n_real) % 16 if tail else 0
    bounds = [0]
    for length in lengths:
        bounds.append(bounds[-1] + length)
    if pad:
        bounds.append(bounds[-1] + pad)
    n = bounds[-1]
    g = torch.Generator("cuda").manual_seed(seed)
    q = torch.randn(1, n, H, D, device="cuda", generator=g).half()
    k = torch.randn(1, n, G, D, device="cuda", generator=g).half()
    v = torch.randn(1, n, G, D, device="cuda", generator=g).half()
    dout = torch.randn(1, n, H, D, device="cuda", generator=g).half()
    dout[:, n_real:] = 0  # training: the tail's grad_out is zero
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    return q, k, v, dout, cu, tuple(bounds), len(lengths)


def _original(q, k, v, dout, cu, bounds):
    max_len = max(b - a for a, b in zip(bounds[:-1], bounds[1:]))
    out, lse, seed, offset = unpad.varlen_forward(q, k, v, cu, max_len, SCALE)
    gq, gk, gv, _ = unpad._efficient_backward(dout, q, unpad._expand_heads(k, H // G),
                                              unpad._expand_heads(v, H // G), out, cu, max_len, lse,
                                              seed, offset, SCALE)
    return gq, gk.unflatten(2, (G, H // G)).sum(3), gv.unflatten(2, (G, H // G)).sum(3)


def _reference(q, k, v, dout, bounds, n_seq):
    grads = [torch.zeros(t.shape, device="cuda") for t in (q, k, v)]
    for b in range(n_seq):
        a, e = bounds[b], bounds[b + 1]
        qb, kb, vb = (t[0, a:e].float().transpose(0, 1).requires_grad_() for t in (q, k, v))
        s = qb @ kb.repeat_interleave(H // G, 0).transpose(1, 2) * SCALE
        mask = torch.ones(e - a, e - a, dtype=torch.bool, device="cuda").tril()
        o = s.masked_fill(~mask, float("-inf")).softmax(-1) @ vb.repeat_interleave(H // G, 0)
        o.backward(dout[0, a:e].float().transpose(0, 1))
        for grad, t in zip(grads, (qb, kb, vb)):
            grad[0, a:e] = t.grad.transpose(0, 1)
    return grads


def _max_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("pattern", list(PATTERNS))
@pytest.mark.parametrize("tail", [True, False])
@pytest.mark.parametrize("blocks", [1, 2, 4])
def test_math_backward_matches_reference_and_original(pattern, tail, blocks):
    q, k, v, dout, cu, bounds, n_seq = _batch(PATTERNS[pattern], tail=tail)
    plan = attn_bwd.Plan(bounds, n_seq, q.device, blocks=blocks)
    assert plan.bins == BINS[pattern]
    got = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)
    want = _reference(q, k, v, dout, bounds, n_seq)
    orig = _original(q, k, v, dout, cu, bounds)
    for name, g, w, o in zip("qkv", got, want, orig):
        assert g.shape == o.shape and g.dtype == o.dtype and g.is_contiguous(), name
        assert torch.isfinite(g).all(), name
        # the original path's own error against fp32 is ~5e-4 of the largest gradient here
        assert _max_rel(g, w) < 2.5e-3, (name, _max_rel(g, w), _max_rel(o, w))
        assert _max_rel(g, o) < 2.5e-3, name
        assert (g[:, bounds[n_seq]:] == 0).all() and (o[:, bounds[n_seq]:] == 0).all(), name


def test_scratch_in_the_shared_buffer_and_chunks(math_on):
    """S and P go after what fused_proj's buffer holds (the held weights stay untouched), in chunks
    of as many (bin, kv group) entries as fit there; the gradients are those of one chunk."""
    from solution import fused_proj

    q, k, v, dout, cu, bounds, n_seq = _batch(PATTERNS["4 rows of ~200 and the tail"])
    plan = attn_bwd.Plan(bounds, n_seq, q.device)
    own = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)  # one chunk (16 MiB cap)
    key = (q.device, torch.float16)
    saved_buf, saved_held = fused_proj._BUFFERS.get(key), dict(fused_proj._HELD)
    per_entry = 6 * plan.lp // plan.blocks * plan.lp
    try:
        buf = torch.randn(13_125_000, device="cuda").half()
        fused_proj._BUFFERS[key] = buf
        held = (buf[:5_000_000].view(-1, 1000), buf[5_000_000:5_500_000].view(-1, 500))
        fused_proj._HELD.update(key="test", views=held)
        before = buf[:5_500_000].clone()
        tail = attn_bwd._free_tail(q.device, torch.float16)
        assert tail is not None and tail.storage_offset() >= 5_500_000
        attn_bwd._STATE.update(scratch="nf4", shared=0)
        shared = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)  # 8 entries fit: one chunk
        assert attn_bwd._STATE["shared"] == 1
        assert torch.equal(buf[:5_500_000], before)
        for a, b in zip(shared, own):
            assert torch.equal(a, b)
        # room for 3 entries: chunks of 3, 3, 2
        fused_proj._HELD.update(views=(buf[:buf.numel() - 3 * 3 * per_entry - 100],))
        chunked = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)
        assert attn_bwd._STATE["shared"] == 2
        for a, b in zip(chunked, own):
            assert _max_rel(a, b) < 1e-3
        # not even one entry: allocated scratch
        fused_proj._HELD.update(views=(buf[:buf.numel() - 1000],))
        attn_bwd.math_backward(dout, q, k, v, plan, SCALE)
        assert attn_bwd._STATE["shared"] == 2
    finally:
        if saved_buf is None:
            fused_proj._BUFFERS.pop(key, None)
        else:
            fused_proj._BUFFERS[key] = saved_buf
        fused_proj._HELD.update(saved_held)


def test_own_scratch_cap_chunks(math_on):
    q, k, v, dout, cu, bounds, n_seq = _batch(PATTERNS["uneven, no tail"], tail=False)
    plan = attn_bwd.Plan(bounds, n_seq, q.device)
    whole = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)
    attn_bwd._STATE.update(scratch_mb=1)  # under one entry: one entry per chunk
    single = attn_bwd.math_backward(dout, q, k, v, plan, SCALE)
    for a, b in zip(single, whole):
        assert _max_rel(a, b) < 1e-3


def _through_varlen(lengths, q, k, v, dout):
    """unpad.attention inside `varlen` (as a training step calls it) and its backward."""
    leaves = [t.detach().transpose(1, 2).requires_grad_() for t in (q, k, v)]  # (1, H, N, D) views
    spans = []
    start = 0
    for length in lengths:
        spans.append((start, start + length))
        start += length
    n = q.shape[1]
    with unpad.varlen(spans):
        bounds = unpad.sequences(spans, n)
        unpad._cu_seqlens(bounds, q.device)
        out = unpad.attention(*leaves, bounds, SCALE)
    out.backward(dout)
    return out, [t.grad for t in leaves]


def test_varlen_attention_uses_math_path_and_falls_back(math_on):
    lengths = PATTERNS["4 rows of ~200 and the tail"]
    q, k, v, dout, cu, bounds, n_seq = _batch(lengths)
    attn_bwd._STATE.update(mode="sdpa")
    out0, grads0 = _through_varlen(lengths, q, k, v, dout)
    attn_bwd._STATE.update(mode="math", calls=0, fallbacks=0)
    out1, grads1 = _through_varlen(lengths, q, k, v, dout)
    assert attn_bwd._STATE["calls"] == 1 and attn_bwd._STATE["fallbacks"] == 0
    assert torch.equal(out0, out1)  # the forward is untouched
    for a, b in zip(grads1, grads0):
        assert _max_rel(a, b) < 2.5e-3
    # outside the filter (a row longer than max_len, too few tokens): the original, bitwise
    for change in ({"max_len": 200}, {"min_tokens": 1000}):
        attn_bwd._STATE.update(mode="math", calls=0, fallbacks=0, max_len=256, min_tokens=0)
        attn_bwd._STATE.update(change)
        _, grads2 = _through_varlen(lengths, q, k, v, dout)
        assert attn_bwd._STATE["calls"] == 0  # the forward's `taken` gave no plan
        for a, b in zip(grads2, grads0):
            # aten's dq is not bit-reproducible run to run with several key splits (T16): compare
            # dk, dv bitwise and dq to rounding
            assert _max_rel(a, b) < 1e-3
        assert torch.equal(grads2[1], grads0[1]) and torch.equal(grads2[2], grads0[2])


def test_outside_varlen_or_off_takes_the_original(math_on):
    """No `varlen` context (no boundaries recorded in the forward), or the switch off: no math call."""
    lengths = [200, 190]
    q, k, v, dout, cu, bounds, n_seq = _batch(lengths, tail=False)
    leaves = [t.detach().transpose(1, 2).requires_grad_() for t in (q, k, v)]
    attn_bwd._STATE.update(calls=0, fallbacks=0)
    out = unpad.attention(*leaves, bounds, SCALE)  # cu_seqlens built outside varlen: not cached
    out.backward(dout)
    assert attn_bwd._STATE["calls"] == 0
    attn_bwd._STATE.update(mode="sdpa")
    assert attn_bwd.taken(cu) is None


def test_guards(math_on):
    q, k, v, dout, cu, bounds, n_seq = _batch([200, 190, 180, 170])
    plan = attn_bwd.Plan(bounds, n_seq, q.device)
    assert attn_bwd._fits(dout, q, k, v, plan)
    assert not attn_bwd._fits(dout.float(), q.float(), k.float(), v.float(), plan)
    assert not attn_bwd._fits(dout, q.transpose(1, 2).contiguous().transpose(1, 2), k, v, plan)
    assert not attn_bwd._fits(dout[:, :-16], q[:, :-16], k[:, :-16], v[:, :-16], plan)
    assert not attn_bwd._fits(dout, q[..., :96].contiguous(), k[..., :96].contiguous(),
                              v[..., :96].contiguous(), plan)
    attn_bwd._STATE.update(min_tokens=1000)
    assert not attn_bwd._fits(dout, q, k, v, plan)
    attn_bwd._STATE.update(min_tokens=0, max_len=199)
    assert not attn_bwd._fits(dout, q, k, v, plan)
    assert attn_bwd.backward(dout, q, k, v, cu, plan, SCALE) is None
    assert attn_bwd.backward(dout, q, k, v, cu, None, SCALE) is None


def test_warmup_compiles(math_on):
    attn_bwd._STATE.update(min_tokens=600, max_len=256)
    attn_bwd._warm()


def test_layout_ratio_and_the_ratio_filter(math_on):
    assert attn_bwd.layout([250, 175, 125, 90], 2)[:2] == (256, 3)
    assert attn_bwd.layout([250, 175, 125, 90], 2)[2] == ((0, 0), (1, 0), (2, 0), (2, 125))
    assert attn_bwd.work_ratio([256], 1) == 2.0 and attn_bwd.work_ratio([256], 2) == 1.5
    lengths = PATTERNS["4 rows of ~200 and the tail"]
    q, k, v, dout, cu, bounds, n_seq = _batch(lengths)
    ratio = attn_bwd.work_ratio(lengths, attn_bwd._STATE["blocks"])
    attn_bwd._STATE.update(max_ratio=ratio - 0.01, calls=0)
    _through_varlen(lengths, q, k, v, dout)
    assert attn_bwd._STATE["calls"] == 0
    attn_bwd._STATE.update(max_ratio=ratio + 0.01)
    _through_varlen(lengths, q, k, v, dout)
    assert attn_bwd._STATE["calls"] == 1
