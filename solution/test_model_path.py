"""The whole patched training step on a tiny random Qwen2 against the original transformers path:
loss, per-token metrics and parameter gradients, with padding in the batch."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _tiny_model(seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=640, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                         tie_word_embeddings=True, attn_implementation="sdpa")
    model = Qwen2ForCausalLM(config).cuda().float()
    model.train()
    # as on the task: the tied embedding / lm_head is frozen (the fused head returns no gradient
    # for the head weight), everything else trains
    model.get_input_embeddings().weight.requires_grad_(False)
    return model


def _batch(seed=1, b=3, t=24, vocab=640):
    torch.manual_seed(seed)
    ids = torch.randint(0, vocab, (b, t), device="cuda")
    lengths = torch.tensor([t, t - 7, 9], device="cuda")[:b]
    mask = (torch.arange(t, device="cuda")[None, :] < lengths[:, None]).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, :4] = -100  # a prompt part without loss, as the collator does
    return ids, mask, labels


def _reference_metrics(logits, labels, mask):
    import torch.nn.functional as F

    shift_logits = logits[..., :-1, :].float()
    shift_labels = labels[..., 1:]
    token_loss = F.cross_entropy(shift_logits.transpose(1, 2), shift_labels, ignore_index=-100,
                                 reduction="none")
    logps = F.log_softmax(logits.float(), dim=-1)
    entropy = -(logps.exp() * logps).sum(-1)
    entropy_mean = (entropy * mask).sum() / mask.sum()
    correct = ((shift_logits.argmax(-1) == shift_labels) & (shift_labels != -100)).sum()
    return token_loss, entropy_mean, correct


@pytest.mark.parametrize("unpad", [True, False])
def test_patched_step_matches_original(unpad):
    import solution
    from solution import fused_ce

    solution.apply()
    fused_ce.UNPAD["enabled"] = unpad
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

    original_forward = Qwen2ForCausalLM._soup_original_forward
    ids, mask, labels = _batch()
    num_items = (labels[:, 1:] != -100).sum().item()

    ref = _tiny_model()
    with torch.autocast("cuda", dtype=torch.float16):
        out = original_forward(ref, input_ids=ids, attention_mask=mask, labels=labels,
                               num_items_in_batch=num_items)
    out.loss.backward()
    ref_token_loss, ref_entropy, ref_correct = _reference_metrics(out.logits.detach(), labels, mask)

    ours = _tiny_model()
    with torch.autocast("cuda", dtype=torch.float16):
        res = ours(input_ids=ids, attention_mask=mask, labels=labels, num_items_in_batch=num_items)
    assert res.logits is None
    res.loss.backward()
    step = fused_ce.LAST_STEP["step"]

    torch.testing.assert_close(res.loss.float(), out.loss.float(), rtol=2e-2, atol=2e-3)
    labelled = labels[:, 1:] != -100
    torch.testing.assert_close(step.token_loss[:, :-1][labelled], ref_token_loss[labelled],
                               rtol=2e-2, atol=2e-2)
    assert torch.all(step.token_loss[mask == 0] == 0)
    ours_entropy = (step.token_entropy * mask).sum() / mask.sum()
    torch.testing.assert_close(ours_entropy, ref_entropy, rtol=2e-2, atol=2e-2)
    assert abs(int(step.correct.sum()) - int(ref_correct)) <= 1  # fp16 argmax ties may differ by one
    assert torch.equal(step.labelled, torch.nn.functional.pad(labels, (0, 1), value=-100)[:, 1:] != -100)

    for (name, p_ref), (_, p_ours) in zip(ref.named_parameters(), ours.named_parameters()):
        if p_ref.grad is None:
            assert p_ours.grad is None, name
            continue
        scale = p_ref.grad.abs().max().clamp(min=1e-6)
        err = (p_ours.grad - p_ref.grad).abs().max() / scale
        assert err < 5e-2, f"{name}: relative grad error {err:.4f}"


def test_pack_batch_roundtrip():
    from solution.unpad import pack_batch, scatter_rows

    ids, mask, _ = _batch(b=3, t=10)
    packed, pos, idx, spans = pack_batch(ids, mask)
    assert packed.shape[1] == int(mask.sum()) == idx.numel()
    assert spans == [(0, 10), (10, 13), (13, 22)]
    assert pos[0, 10:13].tolist() == [0, 1, 2]
    back = scatter_rows(packed[0].float(), idx, ids.shape)
    assert torch.equal(back[mask.bool()], ids[mask.bool()].float())
    assert torch.all(back[~mask.bool()] == 0)
