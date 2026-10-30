"""Fused output head: lm_head matmul + cross-entropy + TRL metrics without the (B, T, V) logits.

Base Soup materializes fp16 logits for every position, upcasts them to fp32 for the loss, keeps the
log-softmax for the backward, and reads them three more times (TRL entropy, TRL token accuracy, Soup's
per-row rewind losses). With a 151,936-token vocabulary that is about 4 GB at the peak and several
passes over the memory bus per step.

Here the head runs in chunks of rows. For each chunk the logits are an fp16 GEMM (cuBLAS, tensor
cores), then one Triton kernel per row does two passes: pass 1 is an online reduction that yields the
log-sum-exp, the entropy, the argmax and the target logit (so the loss); pass 2, only for rows that
carry a label, overwrites the chunk in place with `(softmax - onehot) * grad_scale`, the gradient of
the loss with respect to the logits. The gradient with respect to the hidden states, `dL/dh = g @ W`,
is then a second GEMM on the labelled rows only, computed right away; the autograd backward only
rescales it. No tensor of shape (rows, V) survives the chunk.

Rows are reordered so that labelled rows come first: a chunk made of prompt or padding rows does
pass 1 only (TRL's entropy is averaged over every real position, so those rows still need the
reduction) and no gradient GEMM.

The per-token loss, entropy and correctness flags are handed to the trainer (see trainer_patch.py),
which logs TRL's `entropy`, `mean_token_accuracy` and `num_tokens` with TRL's own formulas and gives
Soup's rewind recorder the per-token losses through the hook Soup already has for steps without
logits (`soup_cli.trainer.rewind_hf`, attribute `_soup_chunked_step`).

Numerics match the base to rounding: the base takes a softmax of the same fp16 logits in fp32 and
casts the fp32 gradient to fp16 before the backward GEMM; here the statistics are fp32 and the
gradient is written in fp16 once.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

IGNORE_INDEX = -100
DEFAULT_CHUNK = 1024  # rows of logits alive at once: 1024 x 151936 x 2 bytes = 311 MB
BLOCK_V = 4096

# What the last labelled forward left for the trainer's metrics (set by the patched model forward,
# read by trainer_patch.py in the same step).
LAST_STEP = {"step": None}


@dataclass(frozen=True)
class HeadStep:
    """Same field names as Soup's `ChunkedStep` (soup_cli.utils.stream_loss_chunk): Soup's rewind
    recorder reads `token_loss` from the trainer attribute `_soup_chunked_step`."""

    token_entropy: torch.Tensor  # (B, T) fp32: entropy of each position's distribution
    token_loss: torch.Tensor  # (B, T) fp32: cross-entropy of labels[:, t + 1] at position t, 0 elsewhere
    correct: torch.Tensor  # (B, T) int32: 1 where the argmax equals the label
    labelled: torch.Tensor  # (B, T) bool: positions with a label


@triton.jit
def fused_ce_rows_kernel(
    logits_ptr, stride_row,  # fp16 logits of this chunk, (rows, V), row-major
    targets_ptr,  # int64 (rows,): target token id, or IGNORE_INDEX
    loss_ptr, entropy_ptr, correct_ptr,  # fp32 (rows,), fp32 (rows,), int32 (rows,)
    grad_scale,  # fp32: multiplies (softmax - onehot) in pass 2
    n_grad_rows,  # rows [0, n_grad_rows) carry a label: pass 2 overwrites them with the gradient
    V,
    IGNORE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride_row
    target = tl.load(targets_ptr + row)
    labelled = target != IGNORE
    target32 = tl.where(labelled, target, 0).to(tl.int32)

    # pass 1: online log-sum-exp, sum(p * x) for the entropy, argmax (first occurrence, as torch)
    m = -float("inf")
    s = 0.0
    e = 0.0
    best_val = -float("inf")
    best_idx = 0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < V
        x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
        block_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, block_max)
        alpha = tl.exp(m - m_new)
        p = tl.exp(x - m_new)
        s = s * alpha + tl.sum(p, axis=0)
        e = e * alpha + tl.sum(p * tl.where(mask, x, 0.0), axis=0)
        block_idx = tl.min(tl.where(x == block_max, offs, V), axis=0)
        take = block_max > best_val
        best_idx = tl.where(take, block_idx, best_idx)
        best_val = tl.maximum(best_val, block_max)
        m = m_new
    lse = m + tl.log(s)
    entropy = lse - e / s
    x_t = tl.load(base + target32).to(tl.float32)
    loss = tl.where(labelled, lse - x_t, 0.0)
    correct = (labelled & (best_idx == target32)).to(tl.int32)
    tl.store(loss_ptr + row, loss)
    tl.store(entropy_ptr + row, entropy)
    tl.store(correct_ptr + row, correct)

    # pass 2: the gradient of the loss w.r.t. the logits, in place, labelled rows only
    if row < n_grad_rows:
        for start in range(0, V, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            p = tl.exp(x - lse)
            g = (p - tl.where(offs == target32, 1.0, 0.0)) * grad_scale
            tl.store(base + offs, g.to(logits_ptr.dtype.element_ty), mask=mask)


def _head_chunks(hidden16, weight16, targets, n_labelled, grad_scale, chunk):
    """Runs the chunk loop on rows already ordered labelled-first. Returns per-row loss, entropy,
    correctness and dL/dh for the labelled rows (zeros elsewhere)."""
    n, d = hidden16.shape
    v = weight16.shape[0]
    loss = torch.empty(n, dtype=torch.float32, device=hidden16.device)
    entropy = torch.empty(n, dtype=torch.float32, device=hidden16.device)
    correct = torch.empty(n, dtype=torch.int32, device=hidden16.device)
    dh = torch.zeros(n, d, dtype=torch.float16, device=hidden16.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        logits = torch.matmul(hidden16[start:end], weight16.t())  # (rows, V) fp16, cuBLAS
        n_grad = max(0, min(end, n_labelled) - start)
        fused_ce_rows_kernel[(end - start,)](
            logits, logits.stride(0), targets[start:end],
            loss[start:end], entropy[start:end], correct[start:end],
            grad_scale, n_grad, v,
            IGNORE=IGNORE_INDEX, BLOCK=BLOCK_V, num_warps=8,
        )
        if n_grad > 0:
            torch.matmul(logits[:n_grad], weight16, out=dh[start:start + n_grad])
        del logits
    return loss, entropy, correct, dh


class FusedLMHeadLoss(torch.autograd.Function):
    """loss = sum_i CE_i * grad_scale over labelled rows, where grad_scale = 1 / num_items when the
    trainer gives num_items_in_batch (transformers' `ForCausalLMLoss` with reduction "sum" divided by
    num_items) and 1 / n_labelled otherwise (reduction "mean")."""

    @staticmethod
    def forward(ctx, hidden2d, weight, targets, num_items, chunk):
        n, d = hidden2d.shape
        labelled = targets != IGNORE_INDEX
        order = torch.argsort(labelled.to(torch.int8), descending=True, stable=True)
        n_labelled = int(labelled.sum().item())
        if num_items is None:
            grad_scale = 1.0 / max(n_labelled, 1)
        else:
            grad_scale = 1.0 / float(num_items)
        hidden16 = hidden2d[order].to(torch.float16)
        weight16 = weight if weight.dtype == torch.float16 else weight.to(torch.float16)
        targets_sorted = targets[order].contiguous()
        loss_rows, entropy_rows, correct_rows, dh_sorted = _head_chunks(
            hidden16, weight16, targets_sorted, n_labelled, grad_scale, chunk)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=order.device)
        loss = loss_rows.sum() * grad_scale
        token_loss = loss_rows[inverse]
        token_entropy = entropy_rows[inverse]
        correct = correct_rows[inverse]
        ctx.save_for_backward(dh_sorted, inverse)
        ctx.in_dtype = hidden2d.dtype
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)
        return loss, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _g1, _g2, _g3):
        dh_sorted, inverse = ctx.saved_tensors
        grad = dh_sorted[inverse]
        if grad_loss is not None:
            grad = grad * grad_loss.to(grad.dtype)
        return grad.to(ctx.in_dtype), None, None, None, None


def fused_lm_head_loss(hidden2d, weight, targets, num_items=None, chunk=DEFAULT_CHUNK):
    """(loss, token_loss, token_entropy, correct) for hidden states (N, D), head weight (V, D) and
    next-token targets (N,) with IGNORE_INDEX where there is no label."""
    return FusedLMHeadLoss.apply(hidden2d, weight, targets, num_items, chunk)


def reference_lm_head_loss(hidden2d, weight, targets, num_items=None):
    """What the base does, for tests: fp16 logits, fp32 cross-entropy, TRL entropy, TRL accuracy."""
    logits = torch.matmul(hidden2d.to(torch.float16), weight.to(torch.float16).t())
    logits32 = logits.float()
    token_loss = F.cross_entropy(logits32, targets, ignore_index=IGNORE_INDEX, reduction="none")
    labelled = targets != IGNORE_INDEX
    denom = float(num_items) if num_items is not None else labelled.sum().clamp(min=1)
    loss = token_loss.sum() / denom
    logps = F.log_softmax(logits32, dim=-1)
    entropy = -(logps.exp() * logps).sum(-1)
    correct = (logits.argmax(-1) == targets) & labelled
    return loss, token_loss, entropy, correct.to(torch.int32)


def shift_labels(labels):
    """`ForCausalLMLoss`'s shift: position t is scored against labels[:, t + 1]; the last position
    against nothing."""
    return F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]


def patch_qwen2_causal_lm(chunk=DEFAULT_CHUNK):
    """Qwen2ForCausalLM.forward: in training with labels, skip the logits and return the fused loss.
    Everything else (eval, generation, no labels, CPU) keeps the original forward."""
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2ForCausalLM
    if getattr(cls, "_soup_fused_ce", False):
        return
    original = cls.forward

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, labels=None, use_cache=None, logits_to_keep=0, **kwargs):
        weight = getattr(getattr(self, "lm_head", None), "weight", None)
        if (labels is None or not self.training or not torch.is_tensor(labels) or weight is None
                or not weight.is_cuda or labels.dim() != 2):
            return original(self, input_ids=input_ids, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                            logits_to_keep=logits_to_keep, **kwargs)
        num_items = kwargs.pop("num_items_in_batch", None)
        kwargs.pop("return_dict", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             position_ids=position_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs)
        hidden = outputs.last_hidden_state
        b, t, d = hidden.shape
        targets = shift_labels(labels.to(hidden.device))
        loss, token_loss, token_entropy, correct = fused_lm_head_loss(
            hidden.reshape(-1, d), weight, targets.reshape(-1).contiguous(), num_items, chunk)
        LAST_STEP["step"] = HeadStep(token_entropy=token_entropy.view(b, t),
                                     token_loss=token_loss.view(b, t),
                                     correct=correct.view(b, t),
                                     labelled=targets != IGNORE_INDEX)
        return CausalLMOutputWithPast(loss=loss, logits=None,
                                      past_key_values=outputs.past_key_values,
                                      hidden_states=outputs.hidden_states,
                                      attentions=outputs.attentions)

    cls.forward = forward
    cls._soup_fused_ce = True
