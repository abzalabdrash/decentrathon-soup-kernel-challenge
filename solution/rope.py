"""Rotary position embedding for the Qwen2 attention: q and k in one Triton launch each way.

transformers' `apply_rotary_pos_emb` computes `q * cos + rotate_half(q) * sin` and the same for k:
per tensor two slices, a negation, a concatenation, two products and a sum, each one a kernel that
reads and writes the whole tensor, and as many again in the backward. With q (1, 12, 826, 128) fp32
that is about 5 MB moved per op, 28 layers, every step.

Here one launch reads q, k, cos and sin once and writes the rotated q and k. With x1, x2 the two
halves of a head vector and c1, c2, s1, s2 the matching halves of cos and sin:

    forward:   y1 = x1 * c1 - x2 * s1        y2 = x2 * c2 + x1 * s2
    backward:  g1 = d1 * c1 + d2 * s2        g2 = d2 * c2 - d1 * s1

(the backward is cos * dy plus the transpose of rotate_half, cat(x2, -x1), applied to sin * dy).
The two halves of cos and sin are read separately, so the kernel does not rely on them being equal.
Math in fp32, the result in the dtype the original would return (`q * cos` type promotion).

Layout: q and k arrive as transposed views of the projections, (B, H, N, D) with the heads of one
token next to each other in memory. The kernel takes any strides over B, H and N (D must be
contiguous), writes the outputs in the same layout as the inputs (`empty_like`), and the backward
writes dq and dk in that layout too, so the gradient goes back through `transpose` and `view` of
the projection without a copy. Only cos and sin are saved for the backward, as in the original.

Anything else (CPU, other ranks, `unsqueeze_dim != 1`, odd or unusual head sizes, cos and sin with
a batch that is neither 1 nor B, cos or sin that need a gradient) takes the original function.
"""

import torch
import triton
import triton.language as tl

BLOCK_TOKENS = 16
NUM_WARPS = 4
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


@triton.jit(do_not_specialize=["n_tokens"])
def rope_qk_kernel(
    q_ptr, k_ptr, qo_ptr, ko_ptr, cos_ptr, sin_ptr,
    q_sb, q_sh, q_sn, k_sb, k_sh, k_sn,  # input strides over (B, H, N), in elements
    qo_sb, qo_sh, qo_sn, ko_sb, ko_sh, ko_sn,  # output strides over (B, H, N)
    c_sb, c_sn,  # cos and sin strides over (B, N); c_sb is 0 when one table serves the batch
    n_tokens, n_q_heads,
    HALF: tl.constexpr, BLOCK_N: tl.constexpr, BACKWARD: tl.constexpr,
):
    # one program: one head of q or k (q heads first), BLOCK_N tokens, one batch row
    head = tl.program_id(0)
    batch = tl.program_id(2).to(tl.int64)
    tokens = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inside = (tokens < n_tokens)[:, None]
    lanes = tl.arange(0, HALF)[None, :]

    is_q = head < n_q_heads
    kv_head = head - n_q_heads
    src = tl.where(is_q, q_ptr + batch * q_sb + head * q_sh, k_ptr + batch * k_sb + kv_head * k_sh)
    dst = tl.where(is_q, qo_ptr + batch * qo_sb + head * qo_sh,
                   ko_ptr + batch * ko_sb + kv_head * ko_sh)
    src_row = tl.where(is_q, q_sn, k_sn)
    dst_row = tl.where(is_q, qo_sn, ko_sn)

    src = src + tokens[:, None] * src_row + lanes
    dst = dst + tokens[:, None] * dst_row + lanes
    table = batch * c_sb + tokens[:, None] * c_sn + lanes

    x1 = tl.load(src, mask=inside, other=0.0).to(tl.float32)
    x2 = tl.load(src + HALF, mask=inside, other=0.0).to(tl.float32)
    c1 = tl.load(cos_ptr + table, mask=inside, other=0.0).to(tl.float32)
    c2 = tl.load(cos_ptr + table + HALF, mask=inside, other=0.0).to(tl.float32)
    s1 = tl.load(sin_ptr + table, mask=inside, other=0.0).to(tl.float32)
    s2 = tl.load(sin_ptr + table + HALF, mask=inside, other=0.0).to(tl.float32)
    if BACKWARD:
        y1 = x1 * c1 + x2 * s2
        y2 = x2 * c2 - x1 * s1
    else:
        y1 = x1 * c1 - x2 * s1
        y2 = x2 * c2 + x1 * s2
    tl.store(dst, y1.to(dst.dtype.element_ty), mask=inside)
    tl.store(dst + HALF, y2.to(dst.dtype.element_ty), mask=inside)


def _launch(q, k, q_out, k_out, cos, sin, backward):
    """q, k: (B, H, N, D) inputs; q_out, k_out: outputs of the same shapes; cos, sin: (1 or B, N, D)
    with equal strides. Every last dimension contiguous."""
    batch, q_heads, n, d = q.shape
    grid = (q_heads + k.shape[1], triton.cdiv(n, BLOCK_TOKENS), batch)
    rope_qk_kernel[grid](
        q, k, q_out, k_out, cos, sin,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        q_out.stride(0), q_out.stride(1), q_out.stride(2),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        cos.stride(0) if cos.shape[0] > 1 else 0, cos.stride(1),
        n, q_heads,
        HALF=d // 2, BLOCK_N=BLOCK_TOKENS, BACKWARD=backward, num_warps=NUM_WARPS,
    )


def _last_dim_contiguous(x):
    return x if x.stride(-1) == 1 else x.contiguous()


class RotaryQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q, k = _last_dim_contiguous(q), _last_dim_contiguous(k)
        if cos.stride() != sin.stride() or cos.stride(-1) != 1:
            cos, sin = cos.contiguous(), sin.contiguous()
        out_dtype = torch.promote_types(q.dtype, cos.dtype)
        q_out = torch.empty_like(q, dtype=out_dtype)  # same strides as q when q is dense
        k_out = torch.empty_like(k, dtype=out_dtype)
        _launch(q, k, q_out, k_out, cos, sin, backward=False)
        ctx.save_for_backward(cos, sin)
        ctx.q_layout = (q.shape, q_out.stride(), q.dtype)
        ctx.k_layout = (k.shape, k_out.stride(), k.dtype)
        return q_out, k_out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k):
        cos, sin = ctx.saved_tensors
        grad_q, grad_k = _last_dim_contiguous(grad_q), _last_dim_contiguous(grad_k)
        dq = torch.empty_strided(*ctx.q_layout[:2], dtype=ctx.q_layout[2], device=grad_q.device)
        dk = torch.empty_strided(*ctx.k_layout[:2], dtype=ctx.k_layout[2], device=grad_k.device)
        _launch(grad_q, grad_k, dq, dk, cos, sin, backward=True)
        return dq, dk, None, None


def _original():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    return getattr(mq, "_soup_original_apply_rotary_pos_emb", mq.apply_rotary_pos_emb)


def _row_offsets_fit(x):
    """The kernel indexes inside one batch row with 32-bit offsets."""
    return sum(stride * (size - 1) for stride, size in zip(x.stride()[1:], x.shape[1:])) < 2**31


def _supported(q, k, cos, sin, unsqueeze_dim):
    if unsqueeze_dim != 1 or not all(torch.is_tensor(t) for t in (q, k, cos, sin)):
        return False
    if q.dim() != 4 or k.dim() != 4 or cos.dim() != 3 or sin.shape != cos.shape:
        return False
    if not q.is_cuda or any(t.device != q.device for t in (k, cos, sin)):
        return False
    if q.dtype not in _DTYPES or k.dtype != q.dtype or cos.dtype not in _DTYPES \
            or sin.dtype != cos.dtype:
        return False
    batch, _, n, d = q.shape
    half = d // 2
    if k.shape[0] != batch or k.shape[2] != n or k.shape[3] != d or q.numel() == 0 \
            or k.numel() == 0:
        return False
    if d % 2 or half < 8 or half > 256 or half & (half - 1):  # halves of 8..256 lanes, a power of 2
        return False
    if cos.shape[0] not in (1, batch) or cos.shape[1] != n or cos.shape[2] != d:
        return False
    if not all(_row_offsets_fit(t) for t in (q, k, cos, sin)):
        return False
    if torch.is_grad_enabled() and (cos.requires_grad or sin.requires_grad):
        return False
    return not torch.compiler.is_compiling()


def apply_rotary_pos_emb_triton(q, k, cos, sin, unsqueeze_dim=1):
    """Drop-in for `modeling_qwen2.apply_rotary_pos_emb`: same arguments, same results."""
    if not _supported(q, k, cos, sin, unsqueeze_dim):
        return _original()(q, k, cos, sin, unsqueeze_dim)
    return RotaryQK.apply(q, k, cos, sin)


def patch_qwen2_rope():
    """Replace `modeling_qwen2.apply_rotary_pos_emb`. Qwen2Attention.forward looks the name up in
    the module's globals on every call, and unpad.py calls it through the module, so both attention
    paths take the kernel."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    if getattr(mq, "_soup_rope", False):
        return
    mq._soup_original_apply_rotary_pos_emb = mq.apply_rotary_pos_emb
    mq.apply_rotary_pos_emb = apply_rotary_pos_emb_triton
    mq._soup_rope = True
