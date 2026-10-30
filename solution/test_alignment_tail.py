"""The alignment tail of the packed sequence (unpad.varlen_model_forward pads the packed tokens with a
few dummy tokens to a multiple of 16) influences nothing real: not the real tokens' hidden states or
gradients, not their attention, not the loss, the TRL metrics, Soup's rewind values or the counted
tokens, not the LoRA weight gradients.

How it is pinned, on a 4-bit Qwen2 with peft LoRA on q and v through the patched training step:
- bitwise, at identical shapes: with the tail present, changing what it holds (another token id, and
  NaN in its attention inputs) leaves every real result bit for bit the same. Any influence of the
  tail would show here. (Between tail on and off the GEMMs run on another row count, so cuBLAS may
  pick another kernel and summation order, which is the point of the tail: there the results agree
  to rounding, also checked.)
- exactly: the gradient reaching the tail's rows is 0 at every layer, so they add exactly nothing
  to the LoRA weight gradients (sums over all rows);
- the counted tokens (TRL's num_tokens = attention_mask.sum(), Soup's rewind row tokens from the
  labels) and the per-token metrics live on the (B, T) batch, which has no tail.
Rows are at most 64 tokens long so that the attention backward is reproducible run to run (one key
block per sequence; longer rows sum dq over key blocks in any order, per-row SDPA as well)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENGTHS = (64, 57, 50, 41)  # 212 real tokens: the tail adds 12 (more than 128 tokens: the fused path)


@pytest.fixture(scope="module", autouse=True)
def patched():
    import solution
    from solution import warmup

    solution.apply()
    # the background warm-up draws random numbers on the GPU (cuBLAS warm-up): let it finish, as
    # the trainer does before its first step, so that seeded runs draw the same dropout masks
    assert warmup.wait_for_warmup(600)


@pytest.fixture(autouse=True)
def unpad_state():
    from solution import fused_ce, unpad

    saved, enabled = dict(unpad._STATE), fused_ce.UNPAD["enabled"]
    fused_ce.UNPAD["enabled"] = True
    unpad._STATE.update(pad=16, varlen=True, carry=True, tail_id=None)
    yield unpad._STATE
    unpad._STATE.clear()
    unpad._STATE.update(saved)
    fused_ce.UNPAD["enabled"] = enabled


def _model(dropout):
    from solution.test_fused_proj import _tiny_peft_model

    return _tiny_peft_model(dropout=dropout)


def _batch(vocab=512):
    g = torch.Generator("cuda").manual_seed(3)
    b, t = len(LENGTHS), max(LENGTHS)
    ids = torch.randint(0, vocab, (b, t), device="cuda", generator=g)
    mask = (torch.arange(t, device="cuda")[None, :]
            < torch.tensor(LENGTHS, device="cuda")[:, None]).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :5] = -100  # a prompt part without loss
    return ids, mask, labels


def _step(model, seed=11):
    """One patched training step: loss, the head's per-token values for the trainer, the decoder
    layers' outputs and the gradients reaching them (each over all rows, tail included), the LoRA
    gradients."""
    from solution import fused_ce

    ids, mask, labels = _batch()
    model.zero_grad(set_to_none=True)
    outputs, grads = [], []

    def keep(module, args, output):
        for t in (output if isinstance(output, tuple) else (output,)):
            if torch.is_tensor(t) and t.requires_grad:
                outputs.append(t.detach().clone())
                t.register_hook(lambda g: grads.append(g.detach().clone()))

    layers = model.base_model.model.model.layers
    hooks = [layer.register_forward_hook(keep) for layer in layers]
    try:
        torch.cuda.manual_seed(seed)  # the same LoRA dropout masks in runs of the same shapes
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=mask, labels=labels,
                        num_items_in_batch=int((labels[:, 1:] != -100).sum()))
        step = fused_ce.LAST_STEP["step"]
        (out.loss * 1024.0).backward()  # the GradScaler's role: half gradients out of subnormals
    finally:
        for hook in hooks:
            hook.remove()
    lora = [p.grad.detach().clone() for n, p in model.named_parameters() if "lora_" in n]
    return {"loss": out.loss.detach(), "step": step, "outputs": outputs, "grads": grads[::-1],
            "lora": lora, "mask": mask, "labels": labels}


def _real(t, n):
    return t[:, :n]


def _rewind(step, labels):
    from soup_cli.trainer.rewind_hf import row_losses_from_token_losses

    return row_losses_from_token_losses(step.token_loss, labels)


def _metrics(run):
    """What the trainer logs from a step (trainer_patch.py's formulas) and Soup's rewind values."""
    step, mask, labels = run["step"], run["mask"], run["labels"]
    entropy = (step.token_entropy * mask).sum() / mask.sum()
    accuracy = step.correct.sum() / step.labelled.sum()
    return {"num_tokens": int(mask.sum()), "entropy": float(entropy), "accuracy": float(accuracy),
            "rewind": _rewind(step, labels)}


def test_tail_content_changes_no_real_result(unpad_state, monkeypatch):
    """Another token id in the tail: every real hidden state, gradient, metric, rewind value and LoRA
    gradient is bitwise the same (with LoRA dropout on, as on the task)."""
    from solution import fused_proj

    calls = []
    fused_attention = fused_proj._fused_attention
    monkeypatch.setattr(fused_proj, "_fused_attention",
                        lambda *a: calls.append(1) or fused_attention(*a))
    model = _model(dropout=0.05)
    n = sum(LENGTHS)
    first = _step(model)
    assert len(calls) == 2  # both layers took the fused 4-bit LoRA attention
    again = _step(model)  # control: the step itself is reproducible bit for bit
    unpad_state["tail_id"] = 7
    other = _step(model)
    assert len(first["outputs"]) == 4  # 2 layers carrying (mlp_out, residual) each
    assert first["outputs"][0].shape[1] == n + 12  # the layers ran on the tail too
    for run in (again, other):
        assert torch.equal(run["loss"], first["loss"])
        for a, b in zip(run["outputs"], first["outputs"]):
            assert torch.equal(_real(a, n), _real(b, n))
        for a, b in zip(run["grads"], first["grads"]):
            assert torch.equal(_real(a, n), _real(b, n))
        for field in ("token_loss", "token_entropy", "correct", "labelled"):
            assert torch.equal(getattr(run["step"], field), getattr(first["step"], field))
        assert _metrics(run) == _metrics(first)
        assert len(run["lora"]) == len(first["lora"]) == 8
        for a, b in zip(run["lora"], first["lora"]):
            assert torch.equal(a, b)
    # the tail itself did change
    assert not torch.equal(other["outputs"][0][:, n:], first["outputs"][0][:, n:])


def test_tail_gets_exactly_zero_gradient():
    """The gradient reaching the tail's rows is 0 at every layer output, so their contribution to
    every weight gradient (a sum over all rows) is exactly 0."""
    run = _step(_model(dropout=0.05))
    n = sum(LENGTHS)
    assert run["grads"] and all(g.shape[1] == n + 12 for g in run["grads"])
    for g in run["grads"]:
        assert torch.all(g[:, n:] == 0)
        assert torch.isfinite(g).all()
    for t in run["outputs"]:
        assert torch.isfinite(t).all()  # real values in the tail rows, never empty() garbage


def test_counted_tokens_and_per_token_values_exclude_the_tail(unpad_state):
    """The per-token values the trainer and Soup's rewind read are (B, T), the batch's own shape
    (there is no position for the tail), zero where the batch is padding; the counts are the
    batch's, with the tail on and off alike."""
    model = _model(dropout=0.0)
    on = _step(model)
    unpad_state["pad"] = 0
    off = _step(model)
    ids, mask, labels = _batch()
    for run in (on, off):
        step = run["step"]
        for field in ("token_loss", "token_entropy", "correct", "labelled"):
            assert getattr(step, field).shape == ids.shape
        assert torch.all(step.token_loss[mask == 0] == 0)
        assert torch.all(step.correct[mask == 0] == 0)
        assert torch.equal(step.labelled, torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:]
                           != -100)
    m_on, m_off = _metrics(on), _metrics(off)
    assert m_on["num_tokens"] == m_off["num_tokens"] == sum(LENGTHS)
    assert m_on["rewind"][1] == m_off["rewind"][1]  # rewind's row token counts
    assert m_on["rewind"][1] == [int(c) for c in (labels[:, 1:] != -100).sum(dim=1)]
    assert torch.equal(on["step"].labelled, off["step"].labelled)
    assert torch.equal(on["step"].token_loss != 0, off["step"].token_loss != 0)


def test_tail_on_and_off_agree_to_rounding(unpad_state):
    """Tail on and off: the GEMMs run on 224 or 212 rows (cuBLAS may pick other kernels), so the
    results agree to rounding, the loss and metrics included (LoRA dropout off: its masks are drawn
    over all rows, so their random stream depends on the row count)."""
    model = _model(dropout=0.0)
    n = sum(LENGTHS)
    on = _step(model)
    unpad_state["pad"] = 0
    off = _step(model)
    assert on["outputs"][0].shape[1] == n + 12 and off["outputs"][0].shape[1] == n

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    assert abs(on["loss"].item() - off["loss"].item()) <= 1e-3 * abs(off["loss"].item())
    for a, b in zip(on["outputs"], off["outputs"]):
        assert rel(_real(a, n), b) < 2e-3
    for a, b in zip(on["grads"], off["grads"]):
        assert rel(_real(a, n), b) < 2e-2
    for a, b in zip(on["lora"], off["lora"]):
        assert rel(a, b) < 2e-2
    m_on, m_off = _metrics(on), _metrics(off)
    assert abs(m_on["entropy"] - m_off["entropy"]) < 1e-3 * abs(m_off["entropy"])
    for a, b in zip(m_on["rewind"][0], m_off["rewind"][0]):
        assert abs(a - b) < 1e-3 * abs(b)


@pytest.mark.parametrize("varlen", [True, False])
def test_attention_of_real_rows_never_reads_the_tail(unpad_state, varlen):
    """NaN in the tail's q, k and v: the real rows' attention output and its gradients are bitwise
    those with ordinary tail values, for the one-launch path and the per-row SDPA fallback."""
    from solution import unpad
    from solution.test_unpad import HEAD_DIM, HEADS, SCALE, _qkv, _spans

    unpad_state["varlen"] = varlen
    lengths = [64, 1, 40, 33]
    n, tail = sum(lengths), 10
    bounds = unpad.sequences(_spans(lengths), n + tail)
    grad = torch.randn(1, n + tail, HEADS, HEAD_DIM, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(4)).half()
    grad[:, n:] = 0
    results = []
    for poison in (False, True):
        qkv, q, k, v = _qkv(n + tail, seed=5)
        if poison:
            with torch.no_grad():
                qkv[n:] = float("nan")
        out = unpad.attention(q, k, v, bounds, SCALE)
        out.backward(grad)
        results.append((out.detach(), qkv.grad))
    (out0, dqkv0), (out1, dqkv1) = results
    assert torch.isnan(out1[:, n:]).any()  # the poison is there...
    assert torch.equal(out1[:, :n], out0[:, :n])  # ...and no real row sees it
    assert torch.equal(dqkv1[:n], dqkv0[:n])
