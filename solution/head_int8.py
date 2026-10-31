"""The streamed output head in int8 (task H1, SOUP_SOLUTION_HEAD_INT8; the organizers' answer 29
allows int8 for the lm_head, forward and backward by the input).

head_stream.py streams the tied fp16 weight W (151,936 x 1,536, 445 MiB) from pinned host memory
through the GPU once per step: 38.9 ms of PCIe at 12 GB/s, plus about 0.58 TFLOP of fp16 GEMMs at
~830 tokens. Here a second host copy holds W in int8 with one scale per vocabulary row,

    Q[v, d] = round(W[v, d] c[d] / s_v[v]),   s_v[v] = max_d |W[v, d] c[d]| / 127,

with c the smoothing vector (below), built once per run from the fp16 copy in a thread (pinning,
the quantization on the GPU chunk by chunk, the kernels' compilation; inside the measured time).
The ring then carries 233 MB per step instead of 445 (each chunk's int8 rows followed by its fp32
scales), and both head GEMMs run on the T4's int8 tensor cores (`torch._int_mm`, cuBLASLt IMMA,
int32 accumulation). The embedding lookup still reads the exact fp16 copy (the answer covers the
lm_head only), and eval and generation (`HeadStreamer.linear`) stay fp16 as well.

Smoothing: the final hidden states h = gamma * x / rms(x) (the decoder's final RMSNorm) have outlier
channels (absmax / rms about 19 per token on the public rows), which per-token int8 rounds badly
(loss +0.5 %, accuracy -0.8 % against the fp16 head on real hidden states). The head quantizes h / c
and the copy holds W * c, c = |gamma|: the same products, the outliers moved into the weight's
per-row scales (loss +0.07 %, accuracy 0.0 %); dL/dh is divided by c at the end.

For N hidden rows (labelled first) quantized per token, h_q[n, d] = round(h[n, d] / (c[d] s_h[n])),
and chunk c of the vocabulary:
    L_c = h_q @ Q_c^T                              int32 (N, R), `_int_mm`, in blocks of FWD_ROWS
                                                   hidden rows (the int32 buffer stays the size of
                                                   the fp16 head's logits)
    head_i8_chunk_kernel (own, one program per row): x = L_c * s_h[n] * s_v[v] in fp32, then the
        online max / sum / entropy / argmax / target logit of head_stream.py; for labelled rows
        u = exp(x - m_new) * s_v[v] quantized per row and chunk, P_q = round(u * 127 / max u)
        (int8), its scale and exp(m_old - m_new) kept for the accumulation; the target's row of
        W_q copied
    Q_c^T: int8_transpose_kernel (own), into the int32 logits' buffer (dead by then): the backward
        contracts over the vocabulary, and IMMA wants both operands contiguous along the contraction
        ("TN"; the row-major chunk as is was 2x slower on the T4); s_v sits on that axis, so it goes
        into P before the quantization and one int8 copy of the weight serves both GEMMs
    G_c = P_q @ Q_c                                int32 (n_lab, D), `_int_mm`, labelled rows only
    head_i8_acc_kernel (own): acc = acc * exp(m_old - m_new) + G_c * scale_P (fp32)
    head_finish_kernel (head_stream.py): loss, entropy, correctness, dL/dh = (acc / s - W_q[t]) * g
    head_i8_rescore_kernel (own, labelled rows): the accuracy and the loss from exact logits where
        int8 rounding matters: a row whose int8 top two (or top and target) are closer than DELTA
        takes the exact fp16 logits of these tokens (their fp16 rows read from the pinned host copy
        by the kernel, over PCIe) for its argmax, and every labelled row's loss is corrected to first
        order with the exact logits of its top two and its target (real hidden states, 6 steps of
        the public rows, against the fp16 head: accuracy 0.92 -> 0.04 %, loss 0.15 -> 0.05 %, rewind
        rows 0.41 -> 0.09 %; entropy 0.19 % as is; dL/dh is the int8 head's, untouched)
The int32 logits never exist for more than one chunk; nothing of shape (N, V) is made.

Numerics: the forward sees W_q and h_q instead of W and h (per-row / per-token absmax int8, round
half up); the softmax statistics are fp32 as before. dL/dh is the gradient of that int8 head's
loss with P quantized per row and chunk (u below 1/254 of its chunk's largest value rounds to 0).
nan and inf are not hidden: a non-finite hidden row gives a non-finite scale, so its logits, loss
and dL/dh are not finite either and GradScaler skips the step as in the base.

Which heads take this path never depends on timing: the first START_HEADS heads of the run (the
kernels compile and the int8 copy is built meanwhile) and any head when the build failed take the
fp16 path; head START_HEADS + 1 waits for the thread if it is not done. Switches (environment):
SOUP_SOLUTION_HEAD_INT8 (off by default in every process, see _STATE; solution/conftest.py pins the fp16
head for solution's own tests, test_head_int8.py builds the int8 copy itself), SOUP_SOLUTION_HEAD_INT8_START (START_HEADS), SOUP_SOLUTION_HEAD_INT8_AT
(where the build thread starts: "setup", the end of Soup's setup, or "head", the run's first head),
SOUP_SOLUTION_HEAD_INT8_WARM (where the kernels compile: "thread" or "warmup"),
SOUP_SOLUTION_HEAD_INT8_SMOOTH / _ALPHA (smoothing()), SOUP_SOLUTION_HEAD_INT8_FWD_ROWS,
SOUP_SOLUTION_HEAD_INT8_WARPS.
"""

import atexit
import os
import threading
import time
import weakref

import torch
import triton
import triton.language as tl

from .fused_ce import IGNORE_INDEX
from .head_stream import BLOCK_D, head_finish_kernel

# off by default in every process: with the rescoring and the loss correction the head gains 3-7 ms per step
# after the switch, and the build thread's compilations cost steps 0-4 +0.3..+0.5 s (wall +0.10 s Kaggle
# soup-h1-abba4, +0.56 s Modal h1-abba7); SOUP_SOLUTION_HEAD_INT8=1 turns it on
_STATE = {"on": os.environ.get("SOUP_SOLUTION_HEAD_INT8", "0").strip().lower() not in ("0", "false", "no", "")}
START_HEADS = int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_START", "5"))
# the accuracy's rescoring (head_i8_rescore_kernel): rows whose int8 top two (or top and target) are
# closer than DELTA logits take exact logits for these tokens; DELTA <= 0 switches it off
DELTA = float(os.environ.get("SOUP_SOLUTION_HEAD_INT8_DELTA", "1.0"))
# SOUP_SOLUTION_HEAD_INT8_LOSS_FIX: the first-order loss correction of every labelled row (above)
LOSS_FIX = os.environ.get("SOUP_SOLUTION_HEAD_INT8_LOSS_FIX", "1").strip().lower() not in ("0", "false", "no", "")
# where the build thread starts: "head" (the first head; soup-h1-abba2: setup end -> train begin +0.07 s against
# +0.35 s for "setup", the end of Soup's setup, where the thread fought the trainer build for the GIL)
BUILD_AT = os.environ.get("SOUP_SOLUTION_HEAD_INT8_AT", "head").strip().lower()
# where the four kernels compile: "thread" (the build thread, first) or "warmup" (warmup.py's threads during the
# model load; the build thread then finds them compiled)
WARM_AT = os.environ.get("SOUP_SOLUTION_HEAD_INT8_WARM", "thread").strip().lower()
# int8_transpose_kernel's square tile and warps: T4, one head's 38 transposes, 2.47 ms against 2.77 for 64 x 64
# with 4 warps (soup-h1-kern1)
T_TILE, T_WARPS = 128, 8
QUANT_ROWS = 1024  # vocabulary rows per quantization step of the build (4.5 MiB of GPU buffers)
ACC_BLOCK = 512
# warps of the chunk kernel: T4, one head at 830 tokens, 16 warps 5.8-6.0 ms against 8.5-9.8 for 8 and 8.0-9.5
# for 4 (soup-h1-kern1; head_stream's fp16 kernel is fastest with 8); 2-8 rows per program were slower
WARPS = int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_WARPS", "16"))
# hidden rows per forward GEMM: a chunk's int32 logits are twice the fp16 head's per row. 1024: one
# GEMM and one kernel launch per chunk up to 1024 tokens (each launch costs CPU on the training path);
# the head's memory then +1 MiB over the fp16 head's at 830 tokens, 512 kept it 2-7 MiB below
# (soup-h1-bench1)
FWD_ROWS = max(int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_FWD_ROWS", "1024")) // 8 * 8, 24)


def smoothing(streamer):
    """The smoothing vector for this streamer's copy (SOUP_SOLUTION_HEAD_INT8_SMOOTH): "gamma" (default)
    takes |gamma|^alpha of the decoder's final RMSNorm weight (frozen in this task; any positive vector
    keeps the products exact, so a later change of it would only change the rounding), alpha from
    SOUP_SOLUTION_HEAD_INT8_ALPHA (1: the norm weight folded into W whole: on real final hidden
    states of the public rows loss +0.07 % and accuracy +0.0 % against the fp16 head, against +0.48 % and -0.8 %
    unsmoothed and +0.40 % with alpha 0.5); "none": no smoothing."""
    mode = os.environ.get("SOUP_SOLUTION_HEAD_INT8_SMOOTH", "gamma").strip().lower()
    weight = getattr(getattr(streamer, "norm", None), "weight", None)
    if mode != "gamma" or not isinstance(weight, torch.Tensor) or weight.numel() != streamer.hidden:
        return None
    try:
        alpha = float(os.environ.get("SOUP_SOLUTION_HEAD_INT8_ALPHA", "1"))
    except ValueError:
        alpha = 1.0
    c = weight.detach().float().abs().clamp_min(1e-3) ** alpha
    return c / c.mean()


def enabled():
    """SOUP_SOLUTION_HEAD_INT8 (off by default in every process; solution/conftest.py pins the fp16 head for
    solution's own tests of it)."""
    return _STATE["on"]


@triton.jit(do_not_specialize=["row0", "chunk", "acc_mode", "first", "c0", "vc", "n_lab"])
def head_i8_chunk_kernel(
    l32_ptr, stride_l, row0,  # int32 (block rows, vc): this chunk's logits for rows row0 + i
    pq_ptr, stride_pq,  # int8 (n_lab, vc): P quantized, labelled rows
    sh_ptr, sv_ptr,  # fp32 (rows,) the tokens' scales, (vc,) the chunk's vocabulary row scales
    targets_ptr,  # int64 (rows,): target token id or IGNORE
    m_ptr, s_ptr, e_ptr, best_ptr, xt_ptr,  # fp32 (rows,): running max, sum, entropy sum, best, x_t
    idx_ptr,  # int32 (rows,): running argmax
    cmax_ptr, cidx_ptr, stride_c, chunk,  # fp32, int32 (n_lab, chunks): each chunk's max (the rescoring)
    sec_ptr, sec_idx_ptr,  # fp32, int32 (n_lab,): the runner-up inside the chunk that holds the running top
    alpha_ptr, sp_ptr,  # fp32 (n_lab,): exp(m_old - m_new) and P's scale for this chunk
    acc_ptr, stride_acc, g_ptr, stride_g,  # fp32 acc (n_lab, D); int32 G (n_lab, D) of the previous chunk
    alpha_prev_ptr, sp_prev_ptr,  # fp32 (n_lab,): the previous chunk's alpha and P scale
    acc_mode,  # 0: no previous chunk; 1: acc = G * sp; 2: acc = acc * alpha + G * sp (head_i8_acc_kernel's)
    q_ptr, stride_q,  # int8 (vc, D): the chunk of Q on the GPU
    wt_ptr, stride_wt,  # half (n_lab, D): W_q[target], written by the chunk that holds the target
    first, c0, vc, n_lab, D,  # first: 1 for the vocabulary's first chunk (a runtime flag: one compile)
    IGNORE: tl.constexpr, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
    RUNNER_UP: tl.constexpr,  # keep the chunk's max for head_i8_rescore_kernel's runner-up
):
    row = row0 + tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < vc
    base = l32_ptr + tl.program_id(0).to(tl.int64) * stride_l
    sh = tl.load(sh_ptr + row)
    sv = tl.load(sv_ptr + offs, mask=mask, other=0.0)
    xi = tl.load(base + offs, mask=mask, other=0)
    target = tl.load(targets_ptr + row)
    t_local = target - c0
    hit = (target != IGNORE) & (t_local >= 0) & (t_local < vc)
    t_safe = tl.where(hit, t_local, 0)
    # the target's logit and scale (one load each, as head_stream.head_chunk_kernel)
    sv_t = tl.load(sv_ptr + t_safe, mask=hit, other=0.0)
    x_hit = tl.load(base + t_safe, mask=hit, other=0).to(tl.float32) * (sh * sv_t)
    x = tl.where(mask, xi.to(tl.float32) * (sh * sv), -float("inf"))
    c_max, c_idx = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
    m_old = -float("inf")
    s = 0.0
    e = 0.0
    best = -float("inf")
    best_idx = 0
    x_t = 0.0
    if first == 0:
        m_old = tl.load(m_ptr + row)
        s = tl.load(s_ptr + row)
        e = tl.load(e_ptr + row)
        best = tl.load(best_ptr + row)
        best_idx = tl.load(idx_ptr + row)
        x_t = tl.load(xt_ptr + row)
    best_old = best
    m_new = tl.maximum(m_old, c_max)
    alpha = tl.exp(m_old - m_new)
    p = tl.exp(x - m_new)
    s = s * alpha + tl.sum(p, axis=0)
    e = e * alpha + tl.sum(p * tl.where(mask, x, 0.0), axis=0)
    best_idx = tl.where(c_max > best, c0 + c_idx, best_idx)  # strictly greater: earlier chunks win ties
    best = tl.maximum(best, c_max)
    x_t = tl.where(hit, x_hit, x_t)
    tl.store(m_ptr + row, m_new)
    tl.store(s_ptr + row, s)
    tl.store(e_ptr + row, e)
    tl.store(best_ptr + row, best)
    tl.store(idx_ptr + row, best_idx)
    tl.store(xt_ptr + row, x_t)
    if row < n_lab:
        # the previous chunk's backward GEMM output into acc (here rather than in a launch of its own:
        # every Triton launch costs ~50 us of CPU on the training path, fused_proj.py's notes)
        if acc_mode > 0:
            a_prev = tl.load(alpha_prev_ptr + row)
            s_prev = tl.load(sp_prev_ptr + row)
            acc_row = acc_ptr + row.to(tl.int64) * stride_acc
            g_row = g_ptr + row.to(tl.int64) * stride_g
            for d0 in range(0, D, BLOCK_D):
                od = d0 + tl.arange(0, BLOCK_D)
                md = od < D
                gv = tl.load(g_row + od, mask=md, other=0).to(tl.float32) * s_prev
                if acc_mode == 2:
                    gv = tl.load(acc_row + od, mask=md, other=0.0) * a_prev + gv
                tl.store(acc_row + od, gv, mask=md)
        u = p * sv  # s_v on the contraction axis of the backward GEMM goes into P
        u_max = tl.max(u, axis=0)
        inv = tl.where(u_max > 0.0, 127.0 / u_max, 0.0)
        pq = tl.minimum(tl.floor(u * inv + 0.5), 127.0).to(tl.int8)
        tl.store(pq_ptr + row.to(tl.int64) * stride_pq + offs, pq, mask=mask)
        tl.store(sp_ptr + row, u_max / 127.0)
        tl.store(alpha_ptr + row, alpha)
        # the runner-up of head_i8_rescore_kernel: each chunk's max (the best of the other chunks), and the
        # second of the chunk that takes over the running top (the one that holds the top at the end), a
        # second reduction only there, a few chunks of the 38 (in every chunk it cost 1.3-2.5 ms per head
        # on the T4; without it a runner-up in the top's own chunk was missed: accuracy gap 0.24 % vs 0.04)
        if RUNNER_UP:
            tl.store(cmax_ptr + row.to(tl.int64) * stride_c + chunk, c_max)
            tl.store(cidx_ptr + row.to(tl.int64) * stride_c + chunk, c0 + c_idx)
            if c_max > best_old:
                x2 = tl.where(offs == c_idx, -float("inf"), x)
                c_max2, c_idx2 = tl.max(x2, axis=0, return_indices=True, return_indices_tie_break_left=True)
                tl.store(sec_ptr + row, c_max2)
                tl.store(sec_idx_ptr + row, c0 + c_idx2)
        if hit:
            q_row = q_ptr + t_local * stride_q
            wt_row = wt_ptr + row.to(tl.int64) * stride_wt
            for d0 in range(0, D, BLOCK_D):
                od = d0 + tl.arange(0, BLOCK_D)
                md = od < D
                w = tl.load(q_row + od, mask=md, other=0).to(tl.float32) * sv_t
                tl.store(wt_row + od, w.to(wt_ptr.dtype.element_ty), mask=md)


@triton.jit(do_not_specialize=["n_lab"])
def head_i8_rescore_kernel(
    h_ptr, stride_h,  # half (n, D): the hidden rows (labelled first)
    w_ptr, stride_w,  # half (V, D): the exact weight in pinned host memory, read over PCIe (mapped)
    targets_ptr, idx_ptr, best_ptr, xt_ptr,  # the int8 pass's top and the target's logit
    cmax_ptr, cidx_ptr, stride_c, n_chunks,  # fp32, int32 (n_lab, chunks): each chunk's max
    sec_ptr, sec_idx_ptr,  # fp32, int32 (n_lab,): the runner-up inside the top's own chunk
    m_ptr, s_ptr,  # fp32 (n,): the int8 pass's max and sum (its lse)
    correct_ptr, loss_ptr,  # int32 (n,): rewritten for the rows rescored; fp32 (n,): corrected (FIX_LOSS)
    delta, n_lab, D,
    BLOCK_D: tl.constexpr, BLOCK_C: tl.constexpr, FIX_LOSS: tl.constexpr,
):
    # TRL's accuracy is argmax(logits) == label; int8 logits flip near-ties (1-2 % of the labelled rows
    # on real hidden states). The runner-up is the best of the other chunks' maxima. A labelled row
    # whose int8 top two, or its top and its target, are less
    # than delta apart takes the exact logits of these (at most three) tokens: fp16 weight rows from the
    # host copy, the fp16 hidden row, an fp32 dot rounded to fp16 as the base's fp16 GEMM output, the
    # first id among equal values as torch.argmax. FIX_LOSS (every labelled row): the loss to first
    # order with these exact logits in place of the int8 ones, lse - x_t corrected by
    # sum_k p_k (x8_k - x_k) - (x8_t - x_t) over the top two and the target (p from the int8 pass)
    row = tl.program_id(0)
    t = tl.load(targets_ptr + row).to(tl.int32)
    a1 = tl.load(idx_ptr + row)
    b1 = tl.load(best_ptr + row)
    oc = tl.arange(0, BLOCK_C)
    mc = oc < n_chunks
    cm = tl.load(cmax_ptr + row.to(tl.int64) * stride_c + oc, mask=mc, other=-float("inf"))
    ci = tl.load(cidx_ptr + row.to(tl.int64) * stride_c + oc, mask=mc, other=0)
    cm = tl.where(ci == a1, -float("inf"), cm)  # the top token's own chunk
    b2, j = tl.max(cm, axis=0, return_indices=True, return_indices_tie_break_left=True)
    a2 = tl.sum(tl.where(oc == j, ci, 0), axis=0)
    s2 = tl.load(sec_ptr + row)
    i2 = tl.load(sec_idx_ptr + row)
    take2 = (s2 > b2) | ((s2 == b2) & (i2 < a2))
    b2 = tl.where(take2, s2, b2)
    a2 = tl.where(take2, i2, a2)
    x_t = tl.load(xt_ptr + row)
    near = (b1 - b2 < delta) | ((a1 != t) & (b1 - x_t < delta))
    if near | FIX_LOSS:
        od = tl.arange(0, BLOCK_D)
        md = od < D
        h = tl.load(h_ptr + row.to(tl.int64) * stride_h + od, mask=md, other=0.0).to(tl.float32)
        w1 = tl.load(w_ptr + a1.to(tl.int64) * stride_w + od, mask=md, other=0.0).to(tl.float32)
        w2 = tl.load(w_ptr + a2.to(tl.int64) * stride_w + od, mask=md, other=0.0).to(tl.float32)
        wt = tl.load(w_ptr + t.to(tl.int64) * stride_w + od, mask=md, other=0.0).to(tl.float32)
        x1 = tl.sum(h * w1, axis=0).to(tl.float16).to(tl.float32)
        x2 = tl.sum(h * w2, axis=0).to(tl.float16).to(tl.float32)
        xt = tl.sum(h * wt, axis=0).to(tl.float16).to(tl.float32)
        win_v, win_i = x1, a1
        take = (x2 > win_v) | ((x2 == win_v) & (a2 < win_i))
        win_v = tl.where(take, x2, win_v)
        win_i = tl.where(take, a2, win_i)
        take = (xt > win_v) | ((xt == win_v) & (t < win_i))
        win_i = tl.where(take, t, win_i)
        if near:
            tl.store(correct_ptr + row, (win_i == t).to(tl.int32))
        if FIX_LOSS:
            lse = tl.load(m_ptr + row) + tl.log(tl.load(s_ptr + row))
            other = (t != a1) & (t != a2)
            fix = tl.exp(b1 - lse) * (b1 - x1) + tl.exp(b2 - lse) * (b2 - x2) - (x_t - xt)
            fix = fix + tl.where(other, tl.exp(x_t - lse) * (x_t - xt), 0.0)
            tl.store(loss_ptr + row, tl.load(loss_ptr + row) - fix)


@triton.jit
def int8_transpose_kernel(src_ptr, dst_ptr, n_rows, n_cols, T: tl.constexpr):
    # src (n_rows, n_cols) row-major -> dst (n_cols, n_rows) row-major, one T x T tile per program
    r = tl.program_id(0) * T + tl.arange(0, T)
    c = tl.program_id(1) * T + tl.arange(0, T)
    live = (r < n_rows)[:, None] & (c < n_cols)[None, :]
    x = tl.load(src_ptr + r[:, None].to(tl.int64) * n_cols + c[None, :], mask=live, other=0)
    tl.store(dst_ptr + c[:, None].to(tl.int64) * n_rows + r[None, :], tl.trans(x), mask=tl.trans(live))


@triton.jit(do_not_specialize=["first"])
def head_i8_acc_kernel(acc_ptr, stride_acc, g_ptr, stride_g, alpha_ptr, sp_ptr, first, D, BLOCK: tl.constexpr):
    # acc[row] = acc[row] * exp(m_old - m_new) + G[row] * scale_P (fp32), labelled rows only
    row = tl.program_id(0)
    od = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    md = od < D
    g = tl.load(g_ptr + row.to(tl.int64) * stride_g + od, mask=md, other=0).to(tl.float32)
    g = g * tl.load(sp_ptr + row)
    at = acc_ptr + row.to(tl.int64) * stride_acc + od
    if first == 0:
        g = tl.load(at, mask=md, other=0.0) * tl.load(alpha_ptr + row) + g
    tl.store(at, g, mask=md)


def _rescoring():
    """Whether head_i8_rescore_kernel runs (and the chunk kernel keeps the runner-up for it)."""
    return DELTA > 0 or LOSS_FIX


def _pad_rows(n):
    """Rows of an `_int_mm` operand: more than 16 and a multiple of 8."""
    return max(-(-n // 8) * 8, 24)


def quant_rows(x, out, col, scales=None):
    """Per-row absmax int8 of x (n, D) times `col` (D,) per column into out[:n] (rows past n are left
    as they are), torch ops (a kernel of its own here was one more compilation in the build thread,
    whose GIL share slows the first steps: soup-h1-abba4): returns the fp32 scales absmax / 127 (into
    `scales` when given). A zero row gets scale 0; a non-finite row a non-finite scale (amax keeps nan)."""
    xf = x.float() * col
    amax = xf.abs().amax(dim=1)
    inv = torch.where(amax > 0, 127.0 / amax.clamp_min(1e-30), torch.zeros_like(amax))
    out[:x.shape[0]].copy_(torch.floor(xf.mul_(inv[:, None]).add_(0.5)).clamp_(-127, 127))
    if scales is None:
        return amax.div_(127.0)
    return torch.div(amax, 127.0, out=scales)


def _quant_rows_into(x, q, col, xf, tmp, amax, inv, scales):
    """quant_rows into given buffers only (out= and in-place ops: nothing allocated on the stream it runs
    on). A zero row: 127 / 1e-30 times zeros, so int8 zeros and scale 0."""
    torch.mul(x, col, out=xf)
    torch.abs(xf, out=tmp)
    torch.amax(tmp, dim=1, out=amax)
    torch.clamp(amax, min=1e-30, out=inv)
    torch.reciprocal(inv, out=inv)
    inv.mul_(127.0)
    xf.mul_(inv[:, None]).add_(0.5).floor_().clamp_(-127, 127)
    q.copy_(xf)
    torch.div(amax, 127.0, out=scales)


class HeadSkipInt8(Exception):
    """This streamer's weight does not get an int8 copy."""


class HeadInt8:
    """The int8 form of a HeadStreamer's weight in pinned host memory (built by `start()` in a
    thread) and the head that streams it. Chunk c of the host buffer (and of a ring slot): its
    vc x D int8 rows, then its vc fp32 scales."""

    def __init__(self, streamer, log=print, smooth=None):
        """`smooth`: a positive (D,) vector c: the copy holds W * c (per hidden channel) and the head
        quantizes h / c, which moves the hidden states' outlier channels into the weight (the
        products are the same; only the rounding changes). None: c = 1."""
        if streamer.hidden % 64 or streamer.rows % 64 or streamer.vocab % 8:
            raise HeadSkipInt8(f"shape {streamer.vocab} x {streamer.hidden}, {streamer.rows} rows per chunk")
        self.rows, self.hidden, self.vocab = streamer.rows, streamer.hidden, streamer.vocab
        self.n_chunks = streamer.n_chunks
        device = torch.device(streamer.device)
        self.device = device if device.index is not None else torch.device("cuda", torch.cuda.current_device())
        self.slot_bytes = self.rows * (self.hidden + 4)
        self.block = triton.next_power_of_2(self.rows)
        from .head_stream import _num_warps

        self.num_warps = WARPS if WARPS in (1, 2, 4, 8, 16, 32) else _num_warps(self.block)
        self.ready, self.error, self.seconds, self.times = False, None, 0.0, {}
        self.calls = 0  # heads that asked (take())
        self.log = log
        self._source = streamer.host  # the fp16 host copy
        self._stop = threading.Event()
        self._thread = None
        self.host = self._unpin = None
        self._deferred = False
        self.total_bytes = self.vocab * (self.hidden + 4)
        self.smoothed = smooth is not None
        c = torch.ones(self.hidden) if smooth is None else smooth.detach().float().reshape(-1)
        if c.numel() != self.hidden or not bool(torch.isfinite(c).all()) or not bool((c > 0).all()):
            raise HeadSkipInt8("the smoothing vector is not a positive (D,) vector")
        self.col, self.inv_col = c.to(self.device), (1.0 / c).to(self.device)

    def chunk(self, c):
        """(first row, rows, first byte, bytes) of chunk c."""
        c0 = c * self.rows
        vc = min(self.rows, self.vocab - c0)
        return c0, vc, c * self.slot_bytes, vc * (self.hidden + 4)

    def host_chunk(self, c):
        _, _, b0, nb = self.chunk(c)
        return self.host[b0:b0 + nb]

    # ---- the build ----------------------------------------------------------------------------

    def start(self, defer=False):
        """Starts the build thread now, or (`defer`) at the first take()."""
        self._deferred = defer
        if not defer:
            self._start_thread()
        return self

    def _start_thread(self):
        self._deferred = False
        self._thread = threading.Thread(target=self._build_safely, name="soup-head-int8", daemon=True)
        self._thread.start()
        atexit.register(_join_at_exit, weakref.ref(self))

    def join(self):
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def stop(self):
        self._stop.set()
        self.join()
        self.ready = False
        if self._unpin is not None:
            self._unpin()
            self._unpin = None
        self.host = None

    def _build_safely(self):
        started = time.perf_counter()
        try:
            began = time.perf_counter()
            warm_kernels(self)
            self.times["compile"] = time.perf_counter() - began
            self.build()
            self.ready = True
        except Exception as exc:  # noqa: BLE001 - an optimisation must not break the run
            self.error = f"{type(exc).__name__}: {exc}"
        self.seconds = time.perf_counter() - started
        if self.ready:
            t = self.times
            self.log(f"[solution] head int8: {self.vocab} x {self.hidden} in {self.n_chunks} chunks of "
                     f"{self.rows} rows ({self.total_bytes / 2**20:.0f} MiB pinned, per-row scales, smoothing "
                     f"{'on' if self.smoothed else 'off'}) in {self.seconds:.2f} s (compilation "
                     f"{t.get('compile', 0):.2f}: " + ", ".join(f"{k} {v:.2f}" for k, v in t.get("kernels", {}).items())
                     + f"; pinning {t.get('pin', 0):.2f}, quantization "
                     f"{t.get('quant', 0):.2f}), ready after {self.calls} heads; heads from number "
                     f"{START_HEADS + 1} on")
        elif not self._stop.is_set():
            self.log(f"[solution] head int8 off: {self.error}")

    def build(self):
        """The int8 copy, QUANT_ROWS vocabulary rows at a time: the fp16 rows to the GPU, quant_rows,
        the int8 rows and their scales back into the pinned buffer, all queued on a stream of its own (the
        training's queue is not waited for, nor the GIL held while the GPU works) and waited for once."""
        from .weight_stream import pinned_host_buffer

        torch.cuda.set_device(self.device)
        began = time.perf_counter()
        host, unpin = pinned_host_buffer(self.total_bytes, self)
        self.times["pin"] = time.perf_counter() - began
        began = time.perf_counter()
        dev, d, r = self.device, self.hidden, QUANT_ROWS
        # every buffer of the build allocated here, on the training's stream (its pool takes them back),
        # and freed only after the build stream is synchronized (no record_stream): an allocation on the
        # build stream itself would get a pool of its own that training never reuses (+22 MiB reserved
        # and nvidia-smi peak with quant_rows' temporaries there, Modal h1-abba6)
        w = torch.empty(r, d, dtype=self._source.dtype, device=dev)
        q = torch.empty(r, d, dtype=torch.int8, device=dev)
        xf = torch.empty(2, r, d, dtype=torch.float32, device=dev)
        vec = torch.empty(3, r, dtype=torch.float32, device=dev)
        stream = torch.cuda.Stream(device=dev)
        stream.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(stream), torch.no_grad():
            for c in range(self.n_chunks):
                if self._stop.is_set():
                    break
                c0, vc, b0, _ = self.chunk(c)
                q_host = host[b0:b0 + vc * d].view(torch.int8).view(vc, d)
                s_host = host[b0 + vc * d:b0 + vc * (d + 4)].view(torch.float32)
                for r0 in range(0, vc, r):
                    n = min(r, vc - r0)
                    w[:n].copy_(self._source[c0 + r0:c0 + r0 + n], non_blocking=True)
                    _quant_rows_into(w[:n], q[:n], self.col, xf[0, :n], xf[1, :n], vec[0, :n], vec[1, :n],
                                     vec[2, :n])
                    q_host[r0:r0 + n].copy_(q[:n], non_blocking=True)
                    s_host[r0:r0 + n].copy_(vec[2, :n], non_blocking=True)
            stream.synchronize()
        if self._stop.is_set():
            unpin()
            raise HeadSkipInt8("stopped")
        scales = [host[b0 + vc * d:b0 + vc * (d + 4)].view(torch.float32)
                  for _, vc, b0, _ in (self.chunk(c) for c in range(self.n_chunks))]
        if not all(bool(torch.isfinite(x).all()) for x in scales):
            unpin()
            raise HeadSkipInt8("a non-finite weight row")
        self.times["quant"] = time.perf_counter() - began
        self.host, self._unpin = host, unpin

    # ---- which heads ---------------------------------------------------------------------------

    def take(self):
        """Whether the head being prefetched streams the int8 copy (asked once per head)."""
        self.calls += 1
        if self._thread is None and self._deferred and not self._stop.is_set():
            self._start_thread()  # SOUP_SOLUTION_HEAD_INT8_AT=head
        if self.calls <= START_HEADS or self.error is not None:
            return False
        if not self.ready:
            self.join()
        return self.ready

    # ---- the head ------------------------------------------------------------------------------

    def head(self, st, hidden16, targets, n_lab, grad_scale):
        """HeadStreamer.head with this int8 copy in the ring (st.prefetch() chose it)."""
        n, d = hidden16.shape
        dev = hidden16.device
        loss = torch.empty(n, dtype=torch.float32, device=dev)
        entropy = torch.empty(n, dtype=torch.float32, device=dev)
        correct = torch.empty(n, dtype=torch.int32, device=dev)
        acc = torch.empty(max(n_lab, 1), d, dtype=torch.float32, device=dev)
        compute = st._compute()
        if n == 0:
            self._finish(st, compute)
            return loss, entropy, correct, acc.zero_()
        with torch.autocast("cuda", enabled=False):
            self._run(st, compute, hidden16, targets, n_lab, grad_scale, loss, entropy, correct, acc)
        self._finish(st, compute)
        return loss, entropy, correct, acc

    def _run(self, st, compute, hidden16, targets, n_lab, grad_scale, loss, entropy, correct, acc):
        n, d = hidden16.shape
        dev = hidden16.device
        blocks = [(b0, min(FWD_ROWS, n - b0)) for b0 in range(0, n, FWD_ROWS)]
        n_pad = max(b0 + _pad_rows(nb) for b0, nb in blocks)  # every block's GEMM rows: > 16, multiple of 8
        h_q = torch.empty(n_pad, d, dtype=torch.int8, device=dev)
        s_h = quant_rows(hidden16, h_q, self.inv_col)
        stride = -(-n // 4) * 4  # 16-byte aligned rows of the state, whatever n
        state = torch.empty(5, stride, dtype=torch.float32, device=dev)
        m, s, e, best, x_t = state
        idx = torch.empty(n, dtype=torch.int32, device=dev)
        stride_c = triton.next_power_of_2(self.n_chunks)  # each labelled row's chunk maxima (the rescoring)
        cmax = torch.empty(max(n_lab, 1), stride_c, dtype=torch.float32, device=dev)
        cidx = torch.empty(max(n_lab, 1), stride_c, dtype=torch.int32, device=dev)
        second = torch.empty(2, -(-max(n_lab, 1) // 4) * 4, dtype=torch.float32, device=dev)
        sec_v, sec_i = second[0], second[1].view(torch.int32)
        lab = max(n_lab, 1)
        wt = torch.empty(lab, d, dtype=hidden16.dtype, device=dev)
        # alpha and P's scale, by the chunk's parity: chunk c's kernel accumulates chunk c - 1's G with them
        alpha_sp = torch.empty(2, 2, -(-lab // 4) * 4, dtype=torch.float32, device=dev)
        rows, vocab = self.rows, self.vocab
        # the int32 logits of a chunk, then (as int8) the chunk's transpose for the backward GEMM
        # (not in place over the slot: the next chunk's kernel reads this chunk's G, not its Q)
        l_rows = max(_pad_rows(nb) for _, nb in blocks)
        l32 = torch.empty(max(l_rows * rows, -(-d * rows // 4)), dtype=torch.int32, device=dev)
        lab_pad = _pad_rows(n_lab)
        g32 = torch.empty(lab_pad, d, dtype=torch.int32, device=dev) if n_lab else None
        pq_buf = torch.empty(lab_pad * rows, dtype=torch.int8, device=dev)
        q_t = l32.view(torch.int8)
        g_arg = g32 if n_lab else l32
        runner_up = _rescoring()
        views = {}  # chunk rows -> (int32 logits, P_q, the transpose's buffer): two sizes at most
        slots = {}  # (ring slot, chunk rows) -> (Q_c, Q_c^T view, s_v): views made once per head
        for c in range(self.n_chunks):
            c0 = c * rows
            vc = min(rows, vocab - c0)
            raw = st._acquire(c, compute)
            key = (c % st.slots, vc)
            slot = slots.get(key)
            if slot is None:
                q = raw[:vc * d].view(torch.int8).view(vc, d)
                slot = slots[key] = (q, q.t(), raw[vc * d:vc * (d + 4)].view(torch.float32))
            q, q_tv, sv = slot
            chunk = views.get(vc)
            if chunk is None:
                q_cols = q_t[:d * vc].view(d, vc)
                chunk = views[vc] = ([(b0, nb, (nb,), h_q[b0:b0 + _pad_rows(nb)], l32[:_pad_rows(nb) * vc].view(-1, vc))
                                      for b0, nb in blocks], pq_buf[:lab_pad * vc].view(lab_pad, vc), q_cols,
                                     q_cols.t(), (triton.cdiv(vc, T_TILE), triton.cdiv(d, T_TILE)))
            gemms, p_q, q_cols, q_cols_t, t_grid = chunk
            new, old = alpha_sp[c % 2], alpha_sp[(c - 1) % 2]
            acc_mode = 0 if c == 0 or not n_lab else (1 if c == 1 else 2)
            for b0, nb, grid, h_b, logits in gemms:
                torch._int_mm(h_b, q_tv, out=logits)
                head_i8_chunk_kernel[grid](
                    logits, vc, b0, p_q, vc, s_h, sv, targets, m, s, e, best, x_t, idx, cmax, cidx, stride_c, c,
                    sec_v, sec_i, new[0], new[1], acc, acc.stride(0), g_arg, d, old[0], old[1], acc_mode,
                    q, d, wt, wt.stride(0), int(c == 0), c0, vc, n_lab, d,
                    IGNORE=IGNORE_INDEX, BLOCK=self.block, BLOCK_D=BLOCK_D, RUNNER_UP=runner_up,
                    num_warps=self.num_warps,
                )
            if n_lab:
                int8_transpose_kernel[t_grid](q, q_cols, vc, d, T=T_TILE, num_warps=T_WARPS)
                st._release(c, compute)  # the slot is read no more: its next chunk may come
                torch._int_mm(p_q, q_cols_t, out=g32)
            else:
                st._release(c, compute)
        if n_lab:  # the last chunk's G
            last = alpha_sp[(self.n_chunks - 1) % 2]
            head_i8_acc_kernel[(n_lab, triton.cdiv(d, ACC_BLOCK))](
                acc, acc.stride(0), g32, g32.stride(0), last[0], last[1], int(self.n_chunks == 1), d,
                BLOCK=ACC_BLOCK, num_warps=4)
        head_finish_kernel[(n,)](
            targets, m, s, e, x_t, idx, loss, entropy, correct,
            acc, acc.stride(0), wt, wt.stride(0), grad_scale, n_lab, d,
            IGNORE=IGNORE_INDEX, BLOCK_D=BLOCK_D, num_warps=4,
        )
        if n_lab and _rescoring() and st.host.is_pinned():
            head_i8_rescore_kernel[(n_lab,)](
                hidden16, hidden16.stride(0), st.host, st.host.stride(0), targets, idx, best, x_t,
                cmax, cidx, stride_c, self.n_chunks, sec_v, sec_i, m, s, correct, loss, DELTA, n_lab, d,
                BLOCK_D=triton.next_power_of_2(d), BLOCK_C=stride_c, FIX_LOSS=LOSS_FIX, num_warps=4)
        if n_lab == 0:
            acc.zero_()
        elif self.smoothed:
            acc[:n_lab].mul_(self.inv_col)  # dL/dh = sum_v g_v (W_v * c) / c

    def _finish(self, st, compute):
        # every copy into the ring has been waited for by the compute stream (st._acquire): freeing
        # it on the compute stream needs no record_stream
        for c in range(st.n_chunks):
            if st._owner[c % st.slots] == c:
                compute.wait_event(st._ready[c % st.slots])
        st._ring, st._owner, st._slot_views, st._ring_pack = None, [-1] * st.slots, None, None
        st.heads += 1
        st.int8_heads += 1
        if st._spans:
            st._harvest()


def _join_at_exit(ref):
    obj = ref()
    if obj is not None:
        obj._stop.set()
        obj.join()


def warm_kernels(obj=None):
    """Compiles the four kernels for the classes the head launches (in the build thread); with `obj`,
    each one's seconds go to obj.times["kernels"]."""
    from .warmup import HIDDEN, compile_kernel, training_half_dtype

    half, f32, i8, i32, i64 = training_half_dtype(), torch.float32, torch.int8, torch.int32, torch.int64
    rows = obj.rows if obj is not None else 4096
    hidden = obj.hidden if obj is not None else HIDDEN
    warps = obj.num_warps if obj is not None else WARPS
    spent = {}

    def timed(name, *args, **kwargs):
        began = time.perf_counter()
        compile_kernel(*args, **kwargs)
        spent[name] = time.perf_counter() - began

    # stride_l = rows of the chunk (4096 and the last 384: divisible by 16), the hidden size 1536;
    # row0, first, c0, vc, n_lab not specialized
    stride_c = triton.next_power_of_2(obj.n_chunks if obj is not None else 38)
    timed("chunk", head_i8_chunk_kernel, i32, rows, 0, i8, rows, f32, f32, i64, f32, f32, f32, f32, f32, i32,
          f32, i32, stride_c, 3, f32, i32, f32, f32, f32, hidden, i32, hidden, f32, f32, 2,
          i8, hidden, half, hidden, 1, 0, rows, 387, hidden,
          IGNORE=IGNORE_INDEX, BLOCK=triton.next_power_of_2(rows), BLOCK_D=BLOCK_D, RUNNER_UP=_rescoring(),
          num_warps=warps)
    timed("acc", head_i8_acc_kernel, f32, hidden, i32, hidden, f32, f32, 1, hidden, BLOCK=ACC_BLOCK, num_warps=4)
    timed("transpose", int8_transpose_kernel, i8, i8, rows, hidden, T=T_TILE, num_warps=T_WARPS)
    if _rescoring():
        timed("rescore", head_i8_rescore_kernel, half, hidden, torch.float16, hidden, i64, i32, f32, f32,
              f32, i32, stride_c, 38, f32, i32, f32, f32, i32, f32, 1.0, 387, hidden,
              BLOCK_D=triton.next_power_of_2(hidden),
              BLOCK_C=stride_c, FIX_LOSS=LOSS_FIX, num_warps=4)
    if obj is not None:
        obj.times["kernels"] = spent
