"""The jury's longest batches (A1 jury audit): four rows at max_length 512 (2,048 tokens, a multiple of
16, so no alignment tail) and 1,999 tokens (a one-token tail to 2,000), the shapes the public rows
never reach (1,427 tokens at most). Varlen attention against per-row SDPA, the packed model forward
with and without the tail, the type ring of streamed weights against the resident model, and the
streamed head on 2,048 rows. The 2,048 / 1,999 cases of the type ring and of the head follow the
codex/review-1 stress tests (611cade, 23c1ddf); the helpers and fixtures are the existing tests'.
"""

import pytest
import torch

from solution import unpad
from solution.test_head_stream import D, _chunked_logits, _chunked_statistics, _exact_grad, _targets
from solution.test_head_stream import streamed  # noqa: F401 - fixture
from solution.test_unpad import (HEAD_DIM, HEADS, SCALE, _masked_reference, _model_step, _per_row_sdpa,
                                 _qkv, _rel, _run, _spans, _tiny_model)
from solution.test_unpad import attention_patch  # noqa: F401 - fixture
from solution.test_weight_ring import _ring_step, night  # noqa: F401 - night is a fixture
from solution.test_weight_stream import _same, _soup_like
from solution.test_weight_stream import layer_shaped_qwen_dir  # noqa: F401 - fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LONG = {"2048 tokens, four max-length rows": [512, 512, 512, 512],
        "1999 tokens, a one-token tail": [512, 511, 509, 467],
        "2047 tokens, one short row": [512, 512, 512, 511]}


@pytest.fixture(autouse=True)
def _unpad_defaults():
    saved = dict(unpad._STATE)
    unpad._STATE.update(pad=16, varlen=True, carry=True)
    yield
    unpad._STATE.update(saved)


@pytest.mark.parametrize("pattern", list(LONG))
def test_varlen_attention_long(pattern):
    lengths = LONG[pattern]
    n = sum(lengths)
    n += unpad._pad_count(n)
    bounds = unpad.sequences(_spans(lengths), n)
    assert bounds[-1] == n and max(b - a for a, b in zip(bounds[:-1], bounds[1:])) == 512
    qkv, q, k, v = _qkv(n, seed=len(pattern))
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(1)).half()
    out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    ref_out, ref_dqkv = _run(_per_row_sdpa, qkv, q, k, v, bounds, grad)
    assert torch.equal(out, ref_out)  # the forward is bitwise per-row SDPA's
    assert torch.isfinite(dqkv).all() and _rel(dqkv, ref_dqkv) < 2e-3
    exact_out, exact_dqkv = _run(_masked_reference, qkv, q, k, v, bounds, grad.float())
    assert _rel(out, exact_out) < 2e-3 and _rel(dqkv, exact_dqkv) < 1e-2


@pytest.mark.parametrize("pattern", list(LONG))
def test_varlen_model_forward_long(attention_patch, pattern):  # noqa: F811 - fixture
    lengths = LONG[pattern]
    model = _tiny_model()
    n = sum(lengths)
    unpad._STATE["pad"] = 0
    seen = []
    ref_hidden, ref_grads = _model_step(model, lengths, seen)
    unpad._STATE["pad"] = 16
    seen.clear()
    hidden, grads = _model_step(model, lengths, seen)
    assert seen == [-(-n // 16) * 16] * 2
    assert hidden.shape == ref_hidden.shape == (1, n, 256)
    assert torch.isfinite(hidden).all() and _rel(hidden, ref_hidden) < 5e-3
    for g, ref in zip(grads, ref_grads):
        assert torch.isfinite(g).all() and _rel(g, ref) < 2e-2


@pytest.mark.parametrize("pattern", list(LONG)[:2])
def test_type_ring_long(layer_shaped_qwen_dir, night, monkeypatch, pattern):  # noqa: F811 - fixtures
    """Two streamed steps of the longest batches equal the resident model's step bit for bit (the
    night fixture fixes the attention backward's key splits), with no copy issued late."""
    from solution import fused_proj
    from solution.weight_stream import stream_model_weights

    monkeypatch.setitem(fused_proj._MLP_MEMORY, "recompute", False)
    lengths = tuple(LONG[pattern])
    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    resident = _ring_step(model, lengths=lengths)
    streamer = stream_model_weights(model, slots=2, ring="type")
    assert streamer is not None and streamer.ring == "type"
    try:
        for _ in range(2):
            assert _same(resident, _ring_step(model, lengths=lengths))
        assert streamer.late_loads == 0
    finally:
        streamer.restore()


@pytest.mark.parametrize("pattern", ["mixed", "all_labelled"])
def test_streamed_head_2048(streamed, pattern):  # noqa: F811 - fixture
    """The streamed head on 2,048 rows: per-token loss, entropy and correctness as fp32 statistics of
    the same fp16 chunk logits, no gradient on unlabelled rows, and dL/dh of the labelled rows within
    fp16 rounding of the exact gradient of those logits."""
    from solution.fused_ce import fused_lm_head_loss

    weight, param, streamer = streamed
    rows = 4096
    streamer.configure(rows=rows, slots=2)
    n = 2048
    torch.manual_seed(n)
    hidden = (torch.randn(n, D, device="cuda") * 2.0).requires_grad_(True)
    targets = _targets(n, pattern, seed=n)
    labelled = targets != -100
    loss, token_loss, entropy, correct = fused_lm_head_loss(hidden, param, targets, None)
    (loss * 65536.0).backward()
    grad = hidden.grad / 65536.0
    logits, sorted_targets, order = _chunked_logits(hidden, weight, targets, rows)
    ref_loss, ref_entropy, ref_correct = _chunked_statistics(logits, sorted_targets, order)
    torch.testing.assert_close(token_loss, ref_loss, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-5, atol=5e-5)
    assert torch.equal(correct, ref_correct)
    denom = int(labelled.sum())
    assert abs(loss.item() - float(ref_loss.sum()) / denom) <= 1e-4 * max(abs(loss.item()), 1.0)
    assert torch.all(token_loss[~labelled] == 0) and torch.all(grad[~labelled] == 0)
    k = min(48, denom)
    first = order[:k]
    exact = _exact_grad(logits[:k], weight, sorted_targets[:k], denom)
    assert (grad[first].double() - exact).abs().max() <= 1e-2 * exact.abs().max()
