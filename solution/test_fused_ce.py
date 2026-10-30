"""The fused head against the base path, forward and backward."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

V, D = 151936, 1536


def _case(n, labelled_frac, seed):
    torch.manual_seed(seed)
    hidden = (torch.randn(n, D, device="cuda", dtype=torch.float32) * 2.0).requires_grad_(True)
    weight = (torch.randn(V, D, device="cuda", dtype=torch.float32) * 0.02).half()
    targets = torch.randint(0, V, (n,), device="cuda")
    drop = torch.rand(n, device="cuda") >= labelled_frac
    targets[drop] = -100
    return hidden, weight, targets


@pytest.mark.parametrize("n,labelled_frac,num_items,chunk", [
    (37, 0.6, None, 16), (64, 0.5, 90.0, 1024), (5, 1.0, 5.0, 2), (12, 0.0, None, 8),
])
def test_fused_head_matches_base(n, labelled_frac, num_items, chunk):
    from solution.fused_ce import fused_lm_head_loss, reference_lm_head_loss

    hidden, weight, targets = _case(n, labelled_frac, seed=n)
    ref_loss, ref_token_loss, ref_entropy, ref_correct = reference_lm_head_loss(
        hidden.detach().clone().requires_grad_(True), weight, targets, num_items)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_loss, *_ = reference_lm_head_loss(ref_hidden, weight, targets, num_items)
    ref_loss.backward()

    loss, token_loss, entropy, correct = fused_lm_head_loss(hidden, weight, targets, num_items, chunk)
    loss.backward()

    labelled = targets != -100
    torch.testing.assert_close(loss, ref_loss, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(token_loss[labelled], ref_token_loss[labelled], rtol=1e-3, atol=1e-3)
    assert torch.all(token_loss[~labelled] == 0)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-3, atol=1e-3)
    assert torch.equal(correct, ref_correct)
    if labelled.any():
        scale = (hidden.grad.abs().max() + 1e-12)
        assert (hidden.grad - ref_hidden.grad).abs().max() / scale < 2e-2
    else:
        assert torch.all(hidden.grad == 0)


def test_grad_scales_with_upstream_gradient():
    from solution.fused_ce import fused_lm_head_loss

    hidden, weight, targets = _case(9, 1.0, seed=3)
    loss, *_ = fused_lm_head_loss(hidden, weight, targets, None, 4)
    (loss * 8.0).backward()
    g8 = hidden.grad.clone()
    hidden.grad = None
    loss, *_ = fused_lm_head_loss(hidden, weight, targets, None, 4)
    loss.backward()
    torch.testing.assert_close(g8, hidden.grad * 8.0, rtol=1e-3, atol=1e-5)


def test_kernel_argmax_ties_first_occurrence():
    """torch.argmax returns the first index of the maximum; the kernel must agree on ties."""
    from solution.fused_ce import BLOCK_V, IGNORE_INDEX, fused_ce_rows_kernel

    rows = 3
    logits = torch.zeros(rows, V, device="cuda", dtype=torch.float16)
    logits[0, 10] = logits[0, 5000] = 3.0  # tie inside one block
    logits[1, 4100] = logits[1, 9000] = 2.0  # tie across blocks
    logits[2, V - 1] = 1.0  # last column, partial block
    targets = torch.tensor([10, 9000, V - 1], device="cuda")
    loss = torch.empty(rows, device="cuda")
    entropy = torch.empty(rows, device="cuda")
    correct = torch.empty(rows, device="cuda", dtype=torch.int32)
    fused_ce_rows_kernel[(rows,)](logits, logits.stride(0), targets, loss, entropy, correct, 1.0, 0, V,
                                  IGNORE=IGNORE_INDEX, BLOCK=BLOCK_V, num_warps=8)
    assert correct.tolist() == [1, 0, 1]
    ref = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="none")
    torch.testing.assert_close(loss, ref, rtol=1e-4, atol=1e-4)
