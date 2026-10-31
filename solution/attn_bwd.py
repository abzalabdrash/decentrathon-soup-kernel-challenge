"""The packed causal attention's backward as batched cuBLAS GEMMs and own Triton kernels (task A2,
SOUP_SOLUTION_ATTN_BWD=math; the default `sdpa` keeps aten's memory-efficient backward).

unpad.py's variable-length attention runs `aten::_efficient_attention_backward` (CUTLASS fmha, sm_75)
once per layer: ~0.8 ms on the T4 for four rows of ~200 tokens, a tiny problem for that kernel
(2.7 GFLOP with the full squares). Here, for one layer: the real sequences of the packed batch go
into bins of LP rows (LP = the longest row rounded up to 16; first fit, longest first, so two short
rows share one bin and its square), 12 query heads in 2 groups of R = 6 sharing a key/value head:

    gather   q, grad_out (H heads) and k, v (G heads), token-major (N, heads, D), into head-major
             (bins, heads, LP, D) buffers, zeros in the rows no sequence fills (own kernel);
    S  = Q K^T          per (bin, kv group): (R*LP x D) @ (D x LP), fp32 out (bmm out_dtype);
    P  = softmax(scale * S), each row over its own sequence's keys up to itself (own kernel), fp16;
    dV = P^T dO         (LP x R*LP) @ (R*LP x D): the group's six heads summed inside the GEMM;
    dP = dO V^T         fp32 out, into S's buffer;
    dS = scale * P * (dP - delta), delta_i = sum_j P_ij dP_ij (own kernel; no O needed), over P;
    dQ = dS K, dK = dS^T Q (dK again summed over the group inside the GEMM);
    scatter  dQ, dK, dV back to token-major (N, heads, D), zeros for the alignment tail (own kernel).

Five strided-batched GEMMs (bins*G batches) and two own row kernels per chunk of (bin, kv group)
entries, plus the gather and the scatter. The query heads of one kv group are consecutive rows of
the head-major buffers, so the GQA expansion and the group sums of the original path disappear into
the GEMMs' M and K dimensions. Empty rows and columns are zeros in every operand and masked to zero
in P and dS, so they add nothing; nor do the other sequences of a bin (masked like the causal part).
The maps between tokens and bin rows (`Plan`) are built once per step on the host and copied from
pinned memory without a host wait.

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

Scratch: S (fp32) and P (fp16), R*LP^2 elements each per (bin, kv group) entry, go into the free
tail of fused_proj.py's shared dequantization buffer (in a layer's backward that buffer holds the
q|k|v|o weights, 10.5 MiB of 26.25, which the o_proj and q/k/v backward read around this one;
everything runs on the one compute stream), as many entries per chunk as fit there (15.75 MiB: all 8
entries at LP = 208, 3 at LP = 384, 1 at LP = 512); without that buffer, or with
SOUP_SOLUTION_ATTN_BWD_SCRATCH=own, allocated chunks of at most SOUP_SOLUTION_ATTN_BWD_SCRATCH_MB (16).
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
    "scratch_mb": _env_int("SOUP_SOLUTION_ATTN_BWD_SCRATCH_MB", 16),
    # query-row blocks per bin (the products skip the causal square's upper right part above them)
    "blocks": max(1, _env_int("SOUP_SOLUTION_ATTN_BWD_BLOCKS", 2)),
    # the step's work_ratio at most this (the math path's score entries over the fmha's causal ones)
    "max_ratio": float(os.environ.get("SOUP_SOLUTION_ATTN_BWD_MAX_RATIO", "1e9") or "1e9"),
    "calls": 0, "fallbacks": 0, "shared": 0, "error": None,
}
ALIGN = 16  # a block of query rows: a multiple of this (16-byte aligned fp16 rows for cuBLAS)
NUM_WARPS_ROWS = 2


def enabled():
    return _STATE["mode"] == "math"


# ---------------------------------------------------------------------------------------------
# kernels


@triton.jit
def attn_bwd_gather_kernel(q_ptr, k_ptr, v_ptr, g_ptr, rowmap_ptr, qp_ptr, kp_ptr, vp_ptr, gp_ptr, LP,
                           H: tl.constexpr, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                           HB: tl.constexpr, GB: tl.constexpr):
    """Row r of bin `bin` (program bin * LP + r), token rowmap[bin * LP + r] (-1: none, zeros):
    its query and grad_out rows to (bins * G, LP, R, D), the R heads of a kv group after each other
    for each row, and its key and value rows to (bins * G, LP, D)."""
    at = tl.program_id(0)
    bin = at // LP
    r = at % LP
    tok = tl.load(rowmap_ptr + at)
    live = tok >= 0
    tok = tl.where(live, tok, 0).to(tl.int64)
    col = tl.arange(0, D)[None, :]
    hq = tl.arange(0, HB)[:, None]
    in_q = hq < H
    src = tok * (H * D) + hq * D + col
    dst = (((bin * G + hq // R) * LP + r) * R + hq % R).to(tl.int64) * D + col
    tl.store(qp_ptr + dst, tl.load(q_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    tl.store(gp_ptr + dst, tl.load(g_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    hk = tl.arange(0, GB)[:, None]
    in_k = hk < G
    src = tok * (G * D) + hk * D + col
    dst = ((bin * G + hk) * LP + r).to(tl.int64) * D + col
    tl.store(kp_ptr + dst, tl.load(k_ptr + src, mask=in_k & live, other=0.0), mask=in_k)
    tl.store(vp_ptr + dst, tl.load(v_ptr + src, mask=in_k & live, other=0.0), mask=in_k)


@triton.jit(do_not_specialize=["e0", "r0"])
def attn_bwd_probs_kernel(s_ptr, p_ptr, seg_ptr, e0, r0, W, LP, RT, scale_log2e,
                          G: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    """One row of P = softmax(scale * S) in a block of query rows r0 .. r0 + RT / R - 1 against keys
    0 .. W - 1: local row pid is head pid % R of query row r0 + (pid % RT) // R of entry
    e0 + pid // RT (bin entry // G); its keys run from its sequence's first row in the bin
    (seg[bin * LP + r], -1: no sequence, a zero row) to r."""
    pid = tl.program_id(0)
    r = r0 + (pid % RT) // R
    bin = (e0 + pid // RT) // G
    first = tl.load(seg_ptr + bin * LP + r)
    j = tl.arange(0, BLOCK)
    alive = first >= 0
    keep = (j <= r) & (j >= first) & alive
    at = pid.to(tl.int64) * W + j
    x = tl.load(s_ptr + at, mask=keep, other=float("-inf"))
    top = tl.where(alive, tl.max(x, axis=0), 0.0)
    e = tl.exp2((x - top) * scale_log2e)
    total = tl.sum(e, axis=0)
    p = e / tl.where(total > 0.0, total, 1.0)
    tl.store(p_ptr + at, p.to(p_ptr.dtype.element_ty), mask=j < W)


@triton.jit
def attn_bwd_dscores_kernel(p_ptr, dp_ptr, W, scale, BLOCK: tl.constexpr):
    """dS = scale * P * (dP - delta) for one row of W keys, delta = sum_j P_j dP_j (the softmax's
    backward without the attention output), written over P."""
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    inside = j < W
    at = row.to(tl.int64) * W + j
    p = tl.load(p_ptr + at, mask=inside, other=0.0).to(tl.float32)
    dp = tl.load(dp_ptr + at, mask=inside, other=0.0)
    delta = tl.sum(p * dp, axis=0)
    ds = (dp - delta) * p * scale
    tl.store(p_ptr + at, ds.to(p_ptr.dtype.element_ty), mask=inside)


@triton.jit
def attn_bwd_scatter_kernel(dqp_ptr, dkp_ptr, dvp_ptr, inv_ptr, dq_ptr, dk_ptr, dv_ptr, LP,
                            H: tl.constexpr, G: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
                            HB: tl.constexpr, GB: tl.constexpr):
    """Token `tok`: its rows of dQ (H heads, half, (bins * G, LP, R, D)) and of the fp32 dK and dV
    sums ((bins * G, LP, D)) at bin row inv[tok] into the token-major half gradients; zeros where
    inv is -1 (the tail)."""
    tok = tl.program_id(0)
    at = tl.load(inv_ptr + tok)
    live = at >= 0
    at = tl.where(live, at, 0)
    bin = at // LP
    r = at % LP
    col = tl.arange(0, D)[None, :]
    hq = tl.arange(0, HB)[:, None]
    in_q = hq < H
    src = (((bin * G + hq // R) * LP + r) * R + hq % R).to(tl.int64) * D + col
    dst = tok.to(tl.int64) * (H * D) + hq * D + col
    tl.store(dq_ptr + dst, tl.load(dqp_ptr + src, mask=in_q & live, other=0.0), mask=in_q)
    hk = tl.arange(0, GB)[:, None]
    in_k = hk < G
    src = ((bin * G + hk) * LP + r).to(tl.int64) * D + col
    dst = tok.to(tl.int64) * (G * D) + hk * D + col
    dk = tl.load(dkp_ptr + src, mask=in_k & live, other=0.0)
    tl.store(dk_ptr + dst, dk.to(dk_ptr.dtype.element_ty), mask=in_k)
    dv = tl.load(dvp_ptr + src, mask=in_k & live, other=0.0)
    tl.store(dv_ptr + dst, dv.to(dv_ptr.dtype.element_ty), mask=in_k)


# ---------------------------------------------------------------------------------------------
# the backward


def _pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _free_tail(device, dtype):
    """A view of the free end of fused_proj.py's shared dequantization buffer: the elements after
    everything it holds now (_HELD's views, 128-byte aligned), None when there is no such buffer."""
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
    end = -(-end // 64) * 64
    return buf[end:] if end < buf.numel() else None


def _balanced(entries, fit):
    """Entries per chunk: as few chunks as `fit` allows, of near-equal size (8 at most 7: 4 + 4)."""
    chunks = -(-entries // min(entries, fit))
    return -(-entries // chunks)


def _scratch(device, half, per_entry, entries):
    """(S fp32, P half, entries per chunk): flat buffers for the scores of as many (bin, kv group)
    entries as fit, per_entry elements each; in the shared buffer's free tail when one entry fits
    there, else allocated (at most SOUP_SOLUTION_ATTN_BWD_SCRATCH_MB, one entry at least)."""
    if _STATE["scratch"] == "nf4":
        tail = _free_tail(device, half)
        fit = tail.numel() // (3 * per_entry) if tail is not None else 0
        if fit >= 1:
            m = _balanced(entries, fit)
            _STATE["shared"] += 1
            return (tail[:2 * m * per_entry].view(torch.float32), tail[2 * m * per_entry:3 * m * per_entry],
                    m)
    m = _balanced(entries, max(1, (_STATE["scratch_mb"] << 20) // (6 * per_entry)))
    return (torch.empty(m * per_entry, dtype=torch.float32, device=device),
            torch.empty(m * per_entry, dtype=half, device=device), m)


def layout(lens, nb):
    """(LP, bins, where) for these row lengths: LP the longest rounded up to NB blocks of a multiple
    of ALIGN rows, rows placed first fit, longest first, into bins of LP rows; where[b] = (bin, first
    row of row b in it)."""
    lp = -(-max(lens) // (ALIGN * nb)) * (ALIGN * nb)
    used, where = [], [None] * len(lens)
    for b in sorted(range(len(lens)), key=lambda i: -lens[i]):
        for i, u in enumerate(used):
            if u + lens[b] <= lp:
                where[b] = (i, u)
                used[i] += lens[b]
                break
        else:
            where[b] = (len(used), 0)
            used.append(lens[b])
    return lp, len(used), tuple(where)


def work_ratio(lens, nb):
    """The math path's score entries (bins x NB blocks of LP / NB rows against (t+1) LP / NB keys)
    over the fmha's causal ones (sum of L^2 / 2)."""
    lp, bins, _ = layout(lens, nb)
    return bins * lp * lp * (nb + 1) / nb / sum(length * length for length in lens)


class Plan:
    """One step's layout for the math path: the real sequences (bounds[:n_seq + 1]) packed into
    bins of LP rows (`layout`: short rows share a bin, so the squares the GEMMs compute are fewer),
    and on the GPU one int32 tensor of three maps: rowmap (bin row -> token, -1 none), seg (bin row
    -> its sequence's first row in the bin, -1 none) and inv (token -> bin row, -1 for the tail).
    Built once per step, copied from pinned memory without a host wait."""

    __slots__ = ("bounds", "n_seq", "blocks", "lp", "bins", "where", "host", "maps", "rowmap", "seg",
                 "inv")

    def __init__(self, bounds, n_seq, device, blocks=None):
        self.bounds, self.n_seq = bounds, n_seq
        self.blocks = blocks or _STATE["blocks"]
        lens = [bounds[i + 1] - bounds[i] for i in range(n_seq)]
        self.lp, self.bins, self.where = lp, bins, where = layout(lens, self.blocks)
        n, rows = bounds[-1], bins * lp
        try:
            host = torch.empty(2 * rows + n, dtype=torch.int32, pin_memory=True)
        except RuntimeError:
            host = torch.empty(2 * rows + n, dtype=torch.int32)
        host.fill_(-1)
        rowmap, seg, inv = host[:rows], host[rows:2 * rows], host[2 * rows:]
        for b, (bin, off) in enumerate(where):
            base, start, length = bin * lp + off, bounds[b], lens[b]
            rowmap[base:base + length] = torch.arange(start, start + length, dtype=torch.int32)
            seg[base:base + length] = off
            inv[start:start + length] = torch.arange(base, base + length, dtype=torch.int32)
        self.host = host  # kept until the step's backward is done (the copy reads it)
        self.maps = host.to(device, non_blocking=host.is_pinned())
        self.rowmap, self.seg, self.inv = self.maps[:rows], self.maps[rows:2 * rows], self.maps[2 * rows:]


_PLAN = {"key": None, "plan": None}


def plan_for(bounds, n_seq, device):
    """The step's Plan when its lengths fit the math path (else None), decided and built at the
    first layer that asks and shared by the others."""
    key = (bounds, n_seq, device, _STATE["blocks"], _STATE["min_tokens"], _STATE["max_len"],
           _STATE["max_ratio"])
    if _PLAN["key"] != key:
        lens = [bounds[i + 1] - bounds[i] for i in range(n_seq)]
        fit = _lengths_fit(bounds, n_seq) and work_ratio(lens, _STATE["blocks"]) <= _STATE["max_ratio"]
        _PLAN["key"], _PLAN["plan"] = key, Plan(bounds, n_seq, device) if fit else None
    return _PLAN["plan"]


def math_backward(grad_out, q, k, v, plan, scale):
    """(grad_q, grad_k, grad_v) of causal attention within each real sequence of `plan`, zeros for
    the tokens after them. grad_out, q: (1, N, H, D); k, v: (1, N, G, D); contiguous, one half dtype.
    The query rows of a bin go in NB blocks of T = LP / NB rows; block t needs keys 0 .. (t+1)T - 1
    only (causal), so the products skip the square's upper right part."""
    _, n, heads, dim = q.shape
    groups = k.shape[2]
    per = heads // groups
    lp, bins, nb = plan.lp, plan.bins, plan.blocks
    rows_t = lp // nb
    rt = per * rows_t  # rows of one block of one entry (query rows x heads of the group)
    half, device = q.dtype, q.device
    entries = bins * groups
    rq, rk = entries * lp * per * dim, entries * lp * dim
    work = torch.empty(2 * rq + 2 * rk, dtype=half, device=device)
    qp, gp, kp, vp = work[:rq], work[rq:2 * rq], work[2 * rq:2 * rq + rk], work[2 * rq + rk:]
    sums = torch.empty(2 * rk, dtype=torch.float32, device=device)  # dK, dV accumulated over blocks
    hb, gb = _pow2(heads), _pow2(groups)
    shape = dict(H=heads, G=groups, R=per, D=dim, HB=hb, GB=gb, num_warps=4)
    attn_bwd_gather_kernel[(bins * lp,)](q, k, v, grad_out, plan.rowmap, qp, kp, vp, gp, lp, **shape)
    qe, ge = qp.view(entries, lp * per, dim), gp.view(entries, lp * per, dim)
    ke, ve = kp.view(entries, lp, dim), vp.view(entries, lp, dim)
    dke, dve = sums[:rk].view(entries, lp, dim), sums[rk:].view(entries, lp, dim)
    s, p, chunk = _scratch(device, half, rt * lp, entries)
    log2e_scale = scale * 1.4426950408889634
    f32 = torch.float32
    for e0 in range(0, entries, chunk):
        e1 = min(entries, e0 + chunk)
        m = e1 - e0
        for t in reversed(range(nb)):  # the last block first: it covers every key row
            width = (t + 1) * rows_t
            block = _pow2(width)
            s3, p3 = s[:m * rt * width].view(m, rt, width), p[:m * rt * width].view(m, rt, width)
            q_t, g_t = qe[e0:e1, t * rt:(t + 1) * rt], ge[e0:e1, t * rt:(t + 1) * rt]
            k_w, v_w = ke[e0:e1, :width], ve[e0:e1, :width]
            dk_w, dv_w = dke[e0:e1, :width], dve[e0:e1, :width]
            torch.bmm(q_t, k_w.transpose(1, 2), out_dtype=f32, out=s3)
            attn_bwd_probs_kernel[(m * rt,)](s3, p3, plan.seg, e0, t * rows_t, width, lp, rt,
                                             log2e_scale, G=groups, R=per, BLOCK=block,
                                             num_warps=NUM_WARPS_ROWS)
            if t == nb - 1:
                torch.bmm(p3.transpose(1, 2), g_t, out_dtype=f32, out=dv_w)
            else:
                torch.baddbmm(dv_w, p3.transpose(1, 2), g_t, out_dtype=f32, out=dv_w)
            torch.bmm(g_t, v_w.transpose(1, 2), out_dtype=f32, out=s3)
            attn_bwd_dscores_kernel[(m * rt,)](p3, s3, width, scale, BLOCK=block,
                                               num_warps=NUM_WARPS_ROWS)
            torch.bmm(p3, k_w, out=g_t)  # dQ over this block's grad_out (read above only)
            if t == nb - 1:
                torch.bmm(p3.transpose(1, 2), q_t, out_dtype=f32, out=dk_w)
            else:
                torch.baddbmm(dk_w, p3.transpose(1, 2), q_t, out_dtype=f32, out=dk_w)
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    attn_bwd_scatter_kernel[(n,)](gp, dke, dve, plan.inv, grad_q, grad_k, grad_v, lp, **shape)
    return grad_q, grad_k, grad_v


def taken(cu):
    """In _VarlenAttention.forward (inside unpad.varlen): the step's Plan when the math path is on
    and the step's lengths fit it, else None."""
    if _STATE["mode"] != "math" or _STATE["error"] is not None:
        return None
    from .unpad import _CTX

    cached, spans = _CTX["cu"], _CTX["spans"]
    if cached is None or cached[2] is not cu or not spans:
        return None
    bounds = cached[0]
    real = len(spans)
    if real >= len(bounds) or bounds[real] != spans[-1][1] or bounds[0] != 0:
        return None
    return plan_for(bounds, real, cu.device)


def _lengths_fit(bounds, n_seq):
    lengths = [bounds[i + 1] - bounds[i] for i in range(n_seq)]
    return (min(lengths) > 0 and max(lengths) <= _STATE["max_len"]
            and bounds[n_seq] >= _STATE["min_tokens"])


def _fits(grad_out, q, k, v, plan):
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
    if plan.bounds[-1] != n or plan.maps.device != q.device:
        return False
    return _lengths_fit(plan.bounds, plan.n_seq)


def backward(grad_out, q, k, v, cu, plan, scale):
    """_VarlenAttention.backward's gradients (grad_q, grad_k, grad_v) by the math path when this step
    fits it (see the module doc), else None (the caller runs the original backward). `plan`: what
    `taken` gave the forward."""
    if plan is None or _STATE["mode"] != "math" or _STATE["error"] is not None:
        return None
    if not _fits(grad_out, q, k, v, plan):
        _STATE["fallbacks"] += 1
        return None
    try:
        grads = math_backward(grad_out, q, k, v, plan, scale)
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
    kernels for each power-of-two block a fitting step can take)."""
    from .warmup import HEAD_DIM, KV_HEADS, Q_HEADS, compile_kernel, training_half_dtype

    half = training_half_dtype()
    i32, f32 = torch.int32, torch.float32
    per = Q_HEADS // KV_HEADS
    shape = dict(H=Q_HEADS, G=KV_HEADS, R=per, D=HEAD_DIM, HB=_pow2(Q_HEADS), GB=_pow2(KV_HEADS),
                 num_warps=4)
    compile_kernel(attn_bwd_gather_kernel, half, half, half, half, i32, half, half, half, half, 208,
                   **shape)
    compile_kernel(attn_bwd_scatter_kernel, half, f32, f32, i32, half, half, half, 208, **shape)
    nb = _STATE["blocks"]
    align = ALIGN * nb
    # a batch of 4 rows with at least min_tokens has a row of min_tokens / 4 or more
    low = -(-max(1, _STATE["min_tokens"] // 4) // align) * align
    high = -(-_STATE["max_len"] // align) * align
    blocks = sorted({_pow2((t + 1) * lp // nb) for lp in range(low, high + 1, align) for t in range(nb)})
    for block in blocks:
        compile_kernel(attn_bwd_probs_kernel, f32, half, i32, 0, 0, 208, 208, 96 * 16, 0.1,
                       G=KV_HEADS, R=per, BLOCK=block, num_warps=NUM_WARPS_ROWS)
        compile_kernel(attn_bwd_dscores_kernel, half, f32, 208, 0.1, BLOCK=block,
                       num_warps=NUM_WARPS_ROWS)


def _report():
    st = stats()
    print(f"[solution] attention math backward: {st['calls']} layer calls ({st['shared']} with the "
          f"scratch in the shared buffer), {st['fallbacks']} to the original backward"
          + (f", off after {st['error']}" if st["error"] else ""), flush=True)


def register_warmup():
    """unpad.patch_qwen2_attention: with SOUP_SOLUTION_ATTN_BWD=math, the kernels' warm-up and a log
    line at exit."""
    if _STATE["mode"] != "math" or _STATE.get("registered"):
        return
    _STATE["registered"] = True
    import atexit

    atexit.register(_report)
    try:
        from .warmup import register
    except ImportError:
        return
    register(_warm, name="attn_bwd_math")
