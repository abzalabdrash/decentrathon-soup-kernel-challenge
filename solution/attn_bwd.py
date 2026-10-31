"""The packed causal attention's backward as batched cuBLAS GEMMs and own Triton kernels (task A2,
SOUP_SOLUTION_ATTN_BWD=math; the default `sdpa` keeps aten's memory-efficient backward).

unpad.py's variable-length attention runs `aten::_efficient_attention_backward` (CUTLASS fmha, sm_75)
once per layer: ~0.8 ms on the T4 for four rows of ~200 tokens, a tiny problem for that kernel
(2.7 GFLOP with the full squares). Here, for one layer, with B real sequences of the packed batch
padded to LP rows (the longest, rounded up to 16), 12 query heads in 2 groups of R = 6 sharing a
key/value head:

    gather   q, grad_out (H heads) and k, v (G heads), token-major (N, heads, D), into head-major
             (B, heads, LP, D) buffers, zeros past each sequence (own kernel, one launch);
    S  = Q K^T          per (sequence, kv group): (R*LP x D) @ (D x LP), fp32 out (bmm out_dtype);
    P  = softmax(scale * S) under the causal mask of each row's own length (own kernel), fp16;
    dV = P^T dO         (LP x R*LP) @ (R*LP x D): the group's six heads summed inside the GEMM;
    dP = dO V^T         fp32 out, into S's buffer;
    dS = scale * P * (dP - delta), delta_i = sum_j P_ij dP_ij (own kernel; no O needed), over P;
    dQ = dS K, dK = dS^T Q (dK again summed over the group inside the GEMM);
    scatter  dQ, dK, dV back to token-major (N, heads, D), zeros for the alignment tail (own kernel).

Five strided-batched GEMMs (B*G batches each) and four own launches per layer. The query heads of one
kv group are consecutive rows of the head-major buffers, so the GQA expansion and the group sums of
the original path disappear into the GEMMs' M and K dimensions. Padded rows and columns are zeros in
every operand and masked to zero in P and dS, so they add nothing.

The alignment tail (unpad.varlen_model_forward) is not computed: its rows get exactly zero gradient
in the original path too (their grad_out is zero: the tail is cut before the head, and no real token
attends to it; with dO = 0, dV = P^T dO = 0, dP = 0, dS = 0), so the scatter writes zeros there.

Not bitwise: P, dP and dS are rounded where the fmha kernel rounds them too (P and dS to fp16 for
the tensor-core products), but the GEMM order differs and delta comes from P and dP instead of
dO . O. test_attn_bwd.py compares the gradients with the original path and an fp32 reference.

When (SOUP_SOLUTION_ATTN_BWD=math): the step has at least SOUP_SOLUTION_ATTN_BWD_MIN_TOKENS real
tokens (600: shorter steps are CPU-bound and pay for the extra launches) and its longest row is at
most SOUP_SOLUTION_ATTN_BWD_MAX_LEN (256: the padded squares grow as B * LP^2, the fmha's work as the
sum of the causal halves); fp16/bf16 CUDA tensors of the expected shapes, contiguous, dropout 0
(unpad's guards), the boundaries known from the forward. Anything else, or any error, takes the
original path.

Scratch: S (fp32) and P (fp16), B*H*LP^2 each, go into the free tail of fused_proj.py's shared
dequantization buffer when it is there and large enough (in a layer's backward that buffer holds
the q|k|v|o weights, 10.5 MiB of 26.25, which the o_proj and q/k/v backward read around this one;
everything runs on the one compute stream), else they are allocated
(SOUP_SOLUTION_ATTN_BWD_SCRATCH=own always allocates).
"""

import os

import torch
import triton
import triton.language as tl


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


_STATE = {
    "mode": os.environ.get("SOUP_SOLUTION_ATTN_BWD", "sdpa").strip().lower(),
    "min_tokens": _env_int("SOUP_SOLUTION_ATTN_BWD_MIN_TOKENS", 600),
    "max_len": _env_int("SOUP_SOLUTION_ATTN_BWD_MAX_LEN", 256),
    "scratch": os.environ.get("SOUP_SOLUTION_ATTN_BWD_SCRATCH", "nf4").strip().lower(),
    "calls": 0, "fallbacks": 0, "shared": 0, "error": None,
}
ALIGN = 16  # padded sequence length: a multiple of this (16-byte aligned fp16 rows for cuBLAS)
NUM_WARPS_ROWS = 2


def enabled():
    return _STATE["mode"] == "math"


# ---------------------------------------------------------------------------------------------
# kernels


@triton.jit
def attn_bwd_gather_kernel(q_ptr, k_ptr, v_ptr, g_ptr, cu_ptr, qp_ptr, kp_ptr, vp_ptr, gp_ptr, LP,
                           H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                           HB: tl.constexpr, GB: tl.constexpr):
    """Padded row `i` of sequence `b`: the H query and grad_out rows and the G key and value rows of
    token start_b + i, token-major (N, heads, D) -> head-major (B, heads, LP, D); zeros past the
    sequence's end."""
    b = tl.program_id(0)
    i = tl.program_id(1)
    start = tl.load(cu_ptr + b)
    end = tl.load(cu_ptr + b + 1)
    live = start + i < end
    tok = (start + i).to(tl.int64)
    col = tl.arange(0, D)[None, :]
    hq = tl.arange(0, HB)[:, None]
    in_q = hq < H
    src = tok * (H * D) + hq * D + col
    dst = ((b * H + hq) * LP + i).to(tl.int64) * D + col
    tl.store(qp_ptr + dst, tl.load(q_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    tl.store(gp_ptr + dst, tl.load(g_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    hk = tl.arange(0, GB)[:, None]
    in_k = hk < G
    src = tok * (G * D) + hk * D + col
    dst = ((b * G + hk) * LP + i).to(tl.int64) * D + col
    tl.store(kp_ptr + dst, tl.load(k_ptr + src, mask=in_k & live, other=0.0), mask=in_k)
    tl.store(vp_ptr + dst, tl.load(v_ptr + src, mask=in_k & live, other=0.0), mask=in_k)


@triton.jit
def attn_bwd_probs_kernel(s_ptr, p_ptr, cu_ptr, LP, H, scale_log2e, BLOCK: tl.constexpr):
    """One row of P = softmax(scale * S) for query row i of head h of sequence b (row (b*H + h)*LP + i
    of the (B*H*LP, LP) scores): keys 0..i only, a row past the sequence all zeros."""
    row = tl.program_id(0)
    i = row % LP
    b = row // (H * LP)
    length = tl.load(cu_ptr + b + 1) - tl.load(cu_ptr + b)
    j = tl.arange(0, BLOCK)
    alive = i < length
    keep = (j <= i) & alive
    at = row.to(tl.int64) * LP + j
    x = tl.load(s_ptr + at, mask=keep, other=float("-inf"))
    top = tl.where(alive, tl.max(x, axis=0), 0.0)
    e = tl.exp2((x - top) * scale_log2e)
    total = tl.sum(e, axis=0)
    p = e / tl.where(total > 0.0, total, 1.0)
    tl.store(p_ptr + at, p.to(p_ptr.dtype.element_ty), mask=j < LP)


@triton.jit
def attn_bwd_dscores_kernel(p_ptr, dp_ptr, LP, scale, BLOCK: tl.constexpr):
    """dS = scale * P * (dP - delta) for one row, delta = sum_j P_j dP_j (the softmax's backward
    without the attention output), written over P."""
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    inside = j < LP
    at = row.to(tl.int64) * LP + j
    p = tl.load(p_ptr + at, mask=inside, other=0.0).to(tl.float32)
    dp = tl.load(dp_ptr + at, mask=inside, other=0.0)
    delta = tl.sum(p * dp, axis=0)
    ds = (dp - delta) * p * scale
    tl.store(p_ptr + at, ds.to(p_ptr.dtype.element_ty), mask=inside)


@triton.jit
def attn_bwd_scatter_kernel(dqp_ptr, dkp_ptr, dvp_ptr, cu_ptr, n_seq, dq_ptr, dk_ptr, dv_ptr, LP,
                            H: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                            HB: tl.constexpr, GB: tl.constexpr, SB: tl.constexpr):
    """Token `tok`: its rows of dQ (H heads), dK and dV (G heads) from the head-major padded buffers
    into the token-major gradients; zeros for a token after the last computed sequence (the tail)."""
    tok = tl.program_id(0)
    s = tl.arange(0, SB)
    ends = tl.load(cu_ptr + 1 + s, mask=s < n_seq, other=2147483647)
    b = tl.sum((ends <= tok).to(tl.int32), axis=0)
    live = b < n_seq
    i = tok - tl.load(cu_ptr + b)
    col = tl.arange(0, D)[None, :]
    hq = tl.arange(0, HB)[:, None]
    in_q = hq < H
    src = ((b * H + hq) * LP + i).to(tl.int64) * D + col
    dst = tok.to(tl.int64) * (H * D) + hq * D + col
    tl.store(dq_ptr + dst, tl.load(dqp_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    hk = tl.arange(0, GB)[:, None]
    in_k = hk < G
    src = ((b * G + hk) * LP + i).to(tl.int64) * D + col
    dst = tok.to(tl.int64) * (G * D) + hk * D + col
    tl.store(dk_ptr + dst, tl.load(dkp_ptr + src, mask=in_k & live, other=0.0), mask=in_k)
    tl.store(dv_ptr + dst, tl.load(dvp_ptr + src, mask=in_k & live, other=0.0), mask=in_k)


# ---------------------------------------------------------------------------------------------
# the backward


def _pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _shared_tail(device, dtype, numel):
    """A (numel,) view of the free end of fused_proj.py's shared dequantization buffer: the elements
    after everything it holds now (_HELD's views), None when it is missing or too short."""
    try:
        from . import fused_proj
    except ImportError:
        return None
    buf = fused_proj._BUFFERS.get((device, dtype))
    if buf is None:
        return None
    end = 0
    stack = [fused_proj._HELD["views"]]
    while stack:
        item = stack.pop()
        if isinstance(item, (tuple, list)):
            stack.extend(item)
        elif isinstance(item, torch.Tensor):
            if item.untyped_storage().data_ptr() != buf.untyped_storage().data_ptr():
                return None
            end = max(end, item.storage_offset() + item.numel())
    end = -(-end // 64) * 64  # 128-byte aligned start
    if buf.numel() - end < numel:
        return None
    return buf[end:end + numel]


def _scratch(device, half, elems):
    """(S fp32, P half) flat buffers of `elems` elements: in the shared buffer's free tail when they
    fit there, else allocated."""
    if _STATE["scratch"] == "nf4" and half in (torch.float16, torch.bfloat16):
        tail = _shared_tail(device, half, 3 * elems)
        if tail is not None:
            _STATE["shared"] += 1
            return tail[:2 * elems].view(torch.float32), tail[2 * elems:]
    return (torch.empty(elems, dtype=torch.float32, device=device),
            torch.empty(elems, dtype=half, device=device))


def math_backward(grad_out, q, k, v, cu, bounds, n_seq, scale):
    """(grad_q, grad_k, grad_v) of causal attention within the first `n_seq` sequences of `bounds`
    (host ints; `cu` the same boundaries as int32 on the GPU), zeros for the tokens after them.
    grad_out, q: (1, N, H, D); k, v: (1, N, G, D); all contiguous, one half dtype."""
    _, n, heads, dim = q.shape
    groups = k.shape[2]
    per = heads // groups
    lmax = max(bounds[i + 1] - bounds[i] for i in range(n_seq))
    lp = -(-lmax // ALIGN) * ALIGN
    half, device = q.dtype, q.device
    rq, rk = n_seq * heads * lp * dim, n_seq * groups * lp * dim
    work = torch.empty(2 * rq + 3 * rk, dtype=half, device=device)
    qp, gp = work[:rq], work[rq:2 * rq]
    kp, vp, dvp = work[2 * rq:2 * rq + rk], work[2 * rq + rk:2 * rq + 2 * rk], work[2 * rq + 2 * rk:]
    elems = n_seq * heads * lp * lp
    s, p = _scratch(device, half, elems)
    hb, gb = _pow2(heads), _pow2(groups)
    attn_bwd_gather_kernel[(n_seq, lp)](q, k, v, grad_out, cu, qp, kp, vp, gp, lp,
                                        H=heads, G=groups, D=dim, HB=hb, GB=gb, num_warps=4)
    batch = n_seq * groups
    qg, gg = qp.view(batch, per * lp, dim), gp.view(batch, per * lp, dim)
    kg, vg, dvg = kp.view(batch, lp, dim), vp.view(batch, lp, dim), dvp.view(batch, lp, dim)
    s3, p3 = s.view(batch, per * lp, lp), p.view(batch, per * lp, lp)
    block = _pow2(lp)
    rows = n_seq * heads * lp
    torch.bmm(qg, kg.transpose(1, 2), out_dtype=torch.float32, out=s3)
    attn_bwd_probs_kernel[(rows,)](s, p, cu, lp, heads, scale * 1.4426950408889634, BLOCK=block,
                                   num_warps=NUM_WARPS_ROWS)
    torch.bmm(p3.transpose(1, 2), gg, out=dvg)
    torch.bmm(gg, vg.transpose(1, 2), out_dtype=torch.float32, out=s3)
    attn_bwd_dscores_kernel[(rows,)](p, s, lp, scale, BLOCK=block, num_warps=NUM_WARPS_ROWS)
    torch.bmm(p3, kg, out=gg)  # dQ, over grad_out's buffer (read by the two GEMMs above only)
    torch.bmm(p3.transpose(1, 2), qg, out=vg)  # dK, over v's buffer (read by dP only)
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    attn_bwd_scatter_kernel[(n,)](gp, vp, dvp, cu, n_seq, grad_q, grad_k, grad_v, lp,
                                  H=heads, G=groups, D=dim, HB=hb, GB=gb, SB=_pow2(len(bounds)),
                                  num_warps=4)
    return grad_q, grad_k, grad_v


def taken(cu):
    """In _VarlenAttention.forward (inside unpad.varlen): (bounds, the number of real sequences) of
    this step's packed batch for the backward when the math path is on, else None."""
    if _STATE["mode"] != "math":
        return None
    from .unpad import _CTX

    cached, spans = _CTX["cu"], _CTX["spans"]
    if cached is None or cached[2] is not cu or not spans:
        return None
    bounds = cached[0]
    real = len(spans)
    if real >= len(bounds) or bounds[real] != spans[-1][1]:
        return None
    return bounds, real


def _fits(grad_out, q, k, v, cu, bounds, n_seq):
    if not (q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
        return False
    if q.dim() != 4 or k.dim() != 4 or q.shape[0] != 1 or k.shape != v.shape or grad_out.shape != q.shape:
        return False
    _, n, heads, dim = q.shape
    if k.shape[1] != n or k.shape[3] != dim or k.shape[2] == 0 or heads % k.shape[2]:
        return False
    if dim not in (64, 128) or heads > 64:
        return False
    if any(t.dtype != q.dtype or t.device != q.device or not t.is_contiguous()
           for t in (grad_out, k, v)) or not q.is_contiguous():
        return False
    if cu.dtype != torch.int32 or cu.device != q.device or cu.numel() != len(bounds):
        return False
    if bounds[0] != 0 or bounds[-1] != n or not 0 < n_seq < len(bounds) or len(bounds) > 64:
        return False
    lengths = [bounds[i + 1] - bounds[i] for i in range(n_seq)]
    if min(lengths) <= 0 or max(lengths) > _STATE["max_len"]:
        return False
    return bounds[n_seq] >= _STATE["min_tokens"]


def backward(grad_out, q, k, v, cu, seqs, scale):
    """_VarlenAttention.backward's gradients (grad_q, grad_k, grad_v) by the math path when this step
    fits it (see the module doc), else None (the caller runs the original backward)."""
    if seqs is None or _STATE["mode"] != "math" or _STATE["error"] is not None:
        return None
    bounds, n_seq = seqs
    if not _fits(grad_out, q, k, v, cu, bounds, n_seq):
        _STATE["fallbacks"] += 1
        return None
    try:
        grads = math_backward(grad_out, q, k, v, cu, bounds, n_seq, scale)
    except (RuntimeError, TypeError, ValueError) as exc:  # e.g. a torch without bmm's out_dtype
        _STATE["error"] = repr(exc)
        print(f"[solution] attention math backward off: {exc!r}", flush=True)
        return None
    _STATE["calls"] += 1
    return grads


def stats():
    return {k: _STATE[k] for k in ("mode", "calls", "fallbacks", "shared", "error")}


def _warm():
    """warmup.py registration: the four kernels as training launches them (fp16 on the T4; the rows
    kernels for each power-of-two block up to SOUP_SOLUTION_ATTN_BWD_MAX_LEN)."""
    from .warmup import HEAD_DIM, KV_HEADS, Q_HEADS, compile_kernel, training_half_dtype

    half = training_half_dtype()
    i32, f32 = torch.int32, torch.float32
    compile_kernel(attn_bwd_gather_kernel, half, half, half, half, i32, half, half, half, half, 208,
                   H=Q_HEADS, G=KV_HEADS, D=HEAD_DIM, HB=_pow2(Q_HEADS), GB=_pow2(KV_HEADS),
                   num_warps=4)
    # a batch of 4 rows with at least min_tokens has a row of min_tokens / 4 or more
    shortest = -(-max(1, _STATE["min_tokens"] // 4) // ALIGN) * ALIGN
    block = _pow2(shortest)
    while block <= _pow2(-(-_STATE["max_len"] // ALIGN) * ALIGN):
        compile_kernel(attn_bwd_probs_kernel, f32, half, i32, 208, Q_HEADS, 0.1, BLOCK=block,
                       num_warps=NUM_WARPS_ROWS)
        compile_kernel(attn_bwd_dscores_kernel, half, f32, 208, 0.1, BLOCK=block,
                       num_warps=NUM_WARPS_ROWS)
        block *= 2
    for sb in (4, 8):
        compile_kernel(attn_bwd_scatter_kernel, half, half, half, i32, 4, half, half, half, 208,
                       H=Q_HEADS, G=KV_HEADS, D=HEAD_DIM, HB=_pow2(Q_HEADS), GB=_pow2(KV_HEADS),
                       SB=sb, num_warps=4)


def register_warmup():
    if _STATE["mode"] != "math":
        return
    try:
        from .warmup import register
    except ImportError:
        return
    register(_warm, name="attn_bwd_math")
