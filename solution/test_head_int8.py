"""The int8 streamed head (head_int8.py): the int8 copy against the fp16 host weight, the head against
the same int8 math in fp64 (loss, entropy, correctness, dL/dh), the fp16 head it replaces, the
deterministic switch point, rows without labels, tiny heads and non-finite hidden rows."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

V, D, ROWS = 8 * 4096 + 384, 1536, 4096  # eight full chunks and the real model's 384-row tail


@pytest.fixture(scope="module")
def streamed():
    from solution.head_stream import HeadStreamer

    torch.manual_seed(0)
    # rows of different magnitudes, as an embedding's (the per-row scales matter)
    weight = (torch.randn(V, D, device="cuda") * 0.02 * torch.rand(V, 1, device="cuda").add_(0.3)).half()
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    streamer = HeadStreamer(param, rows=ROWS, slots=2)
    int8 = streamer.start_int8(defer=False)
    int8.join()
    assert int8.ready, int8.error
    yield weight, streamer, int8
    streamer.restore()


def _int8_weight(int8):
    """(Q (V, D) int8, s_v (V,) fp32) gathered from the host chunks."""
    qs, ss = [], []
    for c in range(int8.n_chunks):
        _, vc, _, _ = int8.chunk(c)
        raw = int8.host_chunk(c)
        qs.append(raw[:vc * D].view(torch.int8).view(vc, D))
        ss.append(raw[vc * D:].view(torch.float32))
    return torch.cat(qs).cuda(), torch.cat(ss).cuda()


def _sorted_inputs(n, frac_labelled, scale, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = (torch.randn(n, D, device="cuda", generator=gen) * scale).half()
    hidden[:, 7] *= 12  # an outlier channel, as the model's final hidden states have
    targets = torch.randint(0, V, (n,), device="cuda", generator=gen)
    targets[torch.rand(n, device="cuda", generator=gen) >= frac_labelled] = -100
    order = torch.argsort((targets != -100).to(torch.int8), descending=True, stable=True)
    t = targets[order].contiguous()
    return hidden[order].contiguous(), t, int((t != -100).sum())


def _run(streamer, int8, h16, t, n_lab, grad_scale):
    int8.calls = max(int8.calls, 100)  # past the switch point
    before = streamer.int8_heads
    streamer.prefetch(h16.shape[0])
    out = streamer.head(h16, t, n_lab, grad_scale)
    assert streamer.int8_heads == before + 1
    return out


def _reference(h16, t, n_lab, grad_scale, q, sv, col=None):
    """The int8 head's math in fp64 from the head's own per-token quantization of h / col (quant_rows,
    checked on its own below): int32-exact products, fp64 softmax; dL/dh divided by col."""
    from solution.head_int8 import quant_rows

    inv = torch.ones(D, device="cuda") if col is None else 1.0 / col
    hq = torch.empty(h16.shape, dtype=torch.int8, device="cuda")
    sh = quant_rows(h16, hq, inv).double()
    logits = (hq.double() @ q.double().t()) * sh[:, None] * sv.double()[None, :]
    lse = torch.logsumexp(logits, 1)
    p = torch.softmax(logits, 1)
    entropy = lse - (p * logits).sum(1)
    lab = t != -100
    tt = t.clamp_min(0)
    loss = torch.where(lab, lse - logits.gather(1, tt[:, None])[:, 0], torch.zeros_like(lse))
    correct = (lab & (logits.argmax(1) == tt)).int()
    w_q = q.double() * sv.double()[:, None]
    g = p[:n_lab].clone()
    g[torch.arange(n_lab), tt[:n_lab]] -= 1
    dh = (g @ w_q) * grad_scale
    if col is not None:
        dh = dh / col.double()
    return loss, entropy, correct, dh, logits


def test_quant_rows_against_torch():
    from solution.head_int8 import quant_rows

    torch.manual_seed(1)
    x = (torch.randn(300, D, device="cuda") * torch.rand(300, 1, device="cuda") * 5).half()
    x[7] = 0
    col = torch.rand(D, device="cuda") + 0.5
    q = torch.empty(300, D, dtype=torch.int8, device="cuda")
    s = quant_rows(x, q, col)
    xf = x.float() * col
    amax = xf.abs().amax(1)
    assert torch.allclose(s, amax / 127, rtol=1e-6, atol=0)
    expect = torch.floor(xf * torch.where(amax > 0, 127 / amax, 0)[:, None] + 0.5).clamp(-127, 127)
    assert (q.float() - expect).abs().max().item() <= 1  # the kernel's division may round a half the other way
    assert (q.float() == expect).float().mean().item() > 0.999
    assert s[7] == 0 and not q[7].any()


def test_int8_copy_matches_the_weight(streamed):
    weight, _, int8 = streamed
    q, sv = _int8_weight(int8)
    w = weight.float()
    amax = w.abs().amax(1)
    assert torch.allclose(sv, amax / 127.0, rtol=1e-6, atol=0)
    expect = torch.floor(w / sv[:, None] + 0.5).clamp(-127, 127)
    # w / s_v in fp32 against w * (127 / amax): the rounding may differ at exact halves only
    assert (q.float() - expect).abs().max().item() <= 1
    assert (q.float() == expect).float().mean().item() > 0.9999


@pytest.mark.parametrize("n,frac,scale", [(826, 0.6, 1.0), (354, 0.3, 3.0), (1440, 1.0, 1.0), (61, 0.5, 8.0)])
def test_int8_head_matches_int8_math(streamed, n, frac, scale):
    _, streamer, int8 = streamed
    q, sv = _int8_weight(int8)
    h16, t, n_lab = _sorted_inputs(n, frac, scale, seed=n)
    grad_scale = 1.0 / max(n_lab, 1)
    loss, entropy, correct, dh = _run(streamer, int8, h16, t, n_lab, grad_scale)
    ref_loss, ref_entropy, ref_correct, ref_dh, logits = _reference(h16, t, n_lab, grad_scale, q, sv)
    assert torch.allclose(loss.double(), ref_loss, rtol=1e-4, atol=1e-4)
    assert torch.allclose(entropy.double(), ref_entropy, rtol=1e-4, atol=1e-4)
    # the argmax may differ only where the two best logits tie to fp32 rounding
    top2 = logits.topk(2, dim=1).values
    tie = (top2[:, 0] - top2[:, 1]) < 1e-5 * top2[:, 0].abs().clamp_min(1)
    assert torch.equal(correct[~tie], ref_correct[~tie])
    # dL/dh: P quantized per row and chunk (and W_q[target] rounded to fp16)
    err = (dh[:n_lab].double() - ref_dh).norm() / ref_dh.norm()
    assert err.item() < 1e-2, err.item()


def test_smoothing_moves_the_outlier_channel(streamed):
    """h / c and W * c: the same products, and with c large on the outlier channel the int8 loss
    lands closer to the fp16 head's."""
    from solution.head_int8 import HeadInt8

    weight, streamer, plain = streamed
    c = torch.ones(D, device="cuda")
    c[7] = 4.0  # _sorted_inputs scales channel 7 by 12
    c[100:110] = 0.5
    smooth = HeadInt8(streamer, smooth=c).start()
    smooth.join()
    assert smooth.ready, smooth.error
    h16, t, n_lab = _sorted_inputs(826, 0.6, 1.0, seed=41)
    grad_scale = 1.0 / n_lab
    try:
        streamer._int8 = smooth
        out = _run(streamer, smooth, h16, t, n_lab, grad_scale)
    finally:
        streamer._int8 = plain
    q, sv = _int8_weight(smooth)
    ref_loss, ref_entropy, _, ref_dh, _ = _reference(h16, t, n_lab, grad_scale, q, sv, col=c)
    assert torch.allclose(out[0].double(), ref_loss, rtol=1e-4, atol=1e-4)
    assert torch.allclose(out[1].double(), ref_entropy, rtol=1e-4, atol=1e-4)
    assert ((out[3][:n_lab].double() - ref_dh).norm() / ref_dh.norm()).item() < 1e-2
    plain_out = _run(streamer, plain, h16, t, n_lab, grad_scale)
    int8_off = plain.calls
    plain.calls = -1000
    streamer.prefetch(826)
    fp16 = streamer.head(h16, t, n_lab, grad_scale)
    plain.calls = int8_off
    lab = t != -100
    err_plain = (plain_out[0][lab] - fp16[0][lab]).abs().mean()
    err_smooth = (out[0][lab] - fp16[0][lab]).abs().mean()
    assert err_smooth < err_plain
    smooth.stop()


def test_int8_head_close_to_the_fp16_head(streamed):
    weight, streamer, int8 = streamed
    h16, t, n_lab = _sorted_inputs(826, 0.6, 1.0, seed=5)
    grad_scale = 1.0 / n_lab
    int8.calls = -1000  # before the switch point: the fp16 ring
    before = streamer.int8_heads
    streamer.prefetch(826)
    fp16 = streamer.head(h16, t, n_lab, grad_scale)
    assert streamer.int8_heads == before
    out = _run(streamer, int8, h16, t, n_lab, grad_scale)
    lab = t != -100
    rel = (out[0][lab].sum() / fp16[0][lab].sum() - 1).abs().item()
    assert rel < 5e-3, rel
    cos = torch.nn.functional.cosine_similarity(out[3][:n_lab].flatten(), fp16[3][:n_lab].flatten(), dim=0)
    assert cos.item() > 0.99


def test_switch_point_is_fixed(streamed, monkeypatch):
    from solution import head_int8

    _, streamer, int8 = streamed
    monkeypatch.setattr(head_int8, "START_HEADS", 2)
    int8.calls = 0
    h16, t, n_lab = _sorted_inputs(300, 0.5, 1.0, seed=3)
    taken = []
    for _ in range(4):
        before = streamer.int8_heads
        streamer.prefetch(300)
        streamer.head(h16, t, n_lab, 1.0 / n_lab)
        taken.append(streamer.int8_heads - before)
    assert taken == [0, 0, 1, 1]


@pytest.mark.parametrize("n,frac", [(97, 0.0), (5, 1.0), (16, 0.5), (17, 1.0)])
def test_int8_head_small_and_unlabelled(streamed, n, frac):
    _, streamer, int8 = streamed
    q, sv = _int8_weight(int8)
    h16, t, n_lab = _sorted_inputs(n, frac, 1.0, seed=11 + n)
    grad_scale = 1.0 / max(n_lab, 1)
    loss, entropy, correct, dh = _run(streamer, int8, h16, t, n_lab, grad_scale)
    ref_loss, ref_entropy, _, ref_dh, _ = _reference(h16, t, n_lab, grad_scale, q, sv)
    assert torch.allclose(loss.double(), ref_loss, rtol=1e-4, atol=1e-4)
    assert torch.allclose(entropy.double(), ref_entropy, rtol=1e-4, atol=1e-4)
    if n_lab:
        assert ((dh[:n_lab].double() - ref_dh).norm() / ref_dh.norm()).item() < 1e-2
    else:
        assert dh.shape == (1, D) and not dh.any()


def test_nonfinite_hidden_stays_nonfinite(streamed):
    _, streamer, int8 = streamed
    h16, t, n_lab = _sorted_inputs(200, 1.0, 1.0, seed=9)
    h16[3, 100] = float("inf")
    h16[5, 10] = float("nan")
    loss, entropy, _, dh = _run(streamer, int8, h16, t, n_lab, 1.0 / n_lab)
    for row in (3, 5):
        assert not torch.isfinite(loss[row]) and not torch.isfinite(dh[row]).all()
    assert torch.isfinite(loss[[0, 1, 2, 4, 6]]).all() and torch.isfinite(dh[[0, 1, 2, 4, 6]]).all()


def test_build_deferred_to_the_first_head(streamed, monkeypatch):
    """SOUP_SOLUTION_HEAD_INT8_AT=head: the thread starts at the first take(); head START_HEADS + 1
    waits for it, so the switch point stays fixed."""
    from solution import head_int8
    from solution.head_int8 import HeadInt8

    _, streamer, plain = streamed
    monkeypatch.setattr(head_int8, "START_HEADS", 1)
    late = HeadInt8(streamer).start(defer=True)
    assert late._thread is None and not late.ready
    h16, t, n_lab = _sorted_inputs(300, 0.5, 1.0, seed=8)
    taken = []
    try:
        streamer._int8 = late
        for _ in range(3):
            before = streamer.int8_heads
            streamer.prefetch(300)
            streamer.head(h16, t, n_lab, 1.0 / n_lab)
            taken.append(streamer.int8_heads - before)
    finally:
        streamer._int8 = plain
    assert taken == [0, 1, 1] and late.ready
    late.stop()


def test_forward_row_blocks_change_nothing(streamed, monkeypatch):
    from solution import head_int8

    _, streamer, int8 = streamed
    h16, t, n_lab = _sorted_inputs(700, 0.6, 1.0, seed=31)
    a = _run(streamer, int8, h16, t, n_lab, 1.0 / n_lab)
    monkeypatch.setattr(head_int8, "FWD_ROWS", 96)  # 7 full blocks and a 28-row tail
    b = _run(streamer, int8, h16, t, n_lab, 1.0 / n_lab)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_eval_logits_stay_fp16(streamed):
    """eval and generation (the lm_head override) read the fp16 host copy, not the int8 one."""
    weight, streamer, _ = streamed
    x = torch.randn(3, 5, D, device="cuda")
    out = streamer.linear(x)  # the virtual dtype (fp32) outside autocast, as the base
    assert torch.allclose(out, x @ weight.float().t(), rtol=1e-4, atol=1e-4)
