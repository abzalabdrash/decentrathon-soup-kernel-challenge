"""int8 tensor cores for the fused MLP's forward GEMM of gate|up (task I1, SOUP_SOLUTION_INT8_FWD; on
by default in the `soup train` process only, like int8_bwd.py: pytest keeps the fp16 forward that
test_fused_proj compares bit for bit, =1 turns it on there, =0 off everywhere). Organizers' answer 29
allows int8 for the frozen linear layers in the forward and the backward by input when all 10 checks
pass.

The fp16 forward of fused_proj._MLP rebuilds W_gate, W_up and W_down in fp16 (nf4.py), runs two fp16
GEMMs into (gate|up) (2I, M), the SwiGLU, and the down GEMM. Here (MODE "gu", the default), with M the
step's tokens (padded to 8):

1. x (M, D) fp32 -> int8 with one scale per token, absmax / 127 (`quant_rows_kernel`; padding rows
   are zeros with scale 0), without the layer's TOPK input channels of the largest max |x|: those go
   to x_k (M, TOPK) in fp32 (an LLM.int8-style decomposition; see `channels`);
2. W_gate, W_up (I, D) NF4 -> int8 (2I, D) in their natural row-major layout, one scale per output
   row (`nf4_int8_gu_kernel`): the row's largest |w| is the largest absmax of its 64-element blocks
   (every NF4 block holds a +-1 code), so the kernel reads the row's absmax codes first and decodes
   the row with that scale in the same program: no pass over the weight, nothing kept between steps;
   extra programs of the same launch decode the TOPK columns of both in fp32 (w_k (2I, TOPK));
3. acc (2I, M) int32 = W_q @ x_q^T (`torch._int_mm`, "TN": x_q^T is column-major), CHUNKS products
   of 2I / CHUNKS rows each into one reused int32 buffer (the int8 weight rows are interleaved by
   chunk: gate rows of chunk k, then up rows of chunk k);
4. `int8_gate_up_kernel`: gate and up = acc * s_w[row] * s_x[token] + w_k @ x_k^T in fp32, rounded to
   half into (gate|up) (2I, M), the tensor the backward keeps (offload.py moves it to the host as
   before), and the SwiGLU output silu(gate) * up in half (I, M) for fused_proj's fp16 down GEMM.
MODE "all" runs down in int8 too (`int8_hidden_kernel`, `nf4_int8_rows_kernel`, `int8_out_kernel`:
per-token int8 of the SwiGLU output, whose peak / RMS per token is 45-95 on the real activations:
5-12 % error of the MLP output, kept for measurements only).
The int8 weights take the place of the fp16 rebuilds in fused_proj's shared buffer (int8 gate|up is
exactly the size of one fp16 (I, D) weight). The int32 product of step 3 is 2I * M / CHUNKS * 4 bytes
(transient; the forward MLP is the moment of the run's peak).

Schedule: the kernels are compiled by a thread started at the COMPILE_CALLS-th eligible forward (in
step 2); the first START_CALLS eligible MLP forwards (eight steps of the 28-layer model) take the fp16
path, and the next one waits for the thread if it is not done, so which forwards run int8 never
depends on timing. Before the switch every fp16 forward records the layer's max |x| per input channel
(`observe`); the first int8 forward of the layer fixes its TOPK channels from it for the rest of the
run. Only forwards with autograd on count (the training's), no-grad forwards (Soup's rewind or
generation) stay fp16.

Numerics: per-token x per-row absmax int8, round half up; the backward is unchanged (it reads the
saved (gate|up), which now carries the int8 GEMM's noise). On the real model's activations
(bench_int8_fwd.py --errors) the loss of a row moves by 0.41 % on average with TOPK = 16 and 1.33 %
without; on the T4 the training curves stay within 0.10-0.15 % of the fp16 run.
"""

import atexit
import os
import threading

import torch
import triton
import triton.language as tl

from . import nf4
from .int8_bwd import _soup_train_process


def _env_on(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


_STATE = {"on": _env_on("SOUP_SOLUTION_INT8_FWD", "1" if _soup_train_process() else "0"),
          "thread": None, "calls": 0}
# the first START_CALLS eligible forwards stay fp16 (8 steps x 28 layers); the compile thread starts
# at eligible forward COMPILE_CALLS (step 2: steps 0 and 1 are the CPU-heavy ones, the later steps
# wait on the GPU and leave the CPU to the thread). T4, soup-i1-abba4: started at the first forward
# with the switch after 4 steps, the thread cost steps 0-4 +100 / +200 / +30 / +67 / +450 ms (the
# last one the switch waiting for it)
START_CALLS = int(os.environ.get("SOUP_SOLUTION_INT8_FWD_START_CALLS", "224"))
COMPILE_CALLS = int(os.environ.get("SOUP_SOLUTION_INT8_FWD_COMPILE_CALLS", "57"))
# products of the gate|up GEMM (2I / CHUNKS weight rows each, one int32 buffer reused)
CHUNKS = int(os.environ.get("SOUP_SOLUTION_INT8_FWD_CHUNKS", "5"))
# which GEMMs run int8: "all" (gate|up and down) or "gu" (gate|up only: its epilogue writes the
# SwiGLU output in half as well, and down stays the fp16 GEMM of fused_proj.py)
MODE = os.environ.get("SOUP_SOLUTION_INT8_FWD_MODE", "gu").strip().lower()


def _layers(text):
    """'0-4' / '1,2,4' / '1.2.4' / '1:3' -> frozenset of layer indices (ranges inclusive)."""
    out = set()
    for part in text.replace(",", " ").replace(".", " ").replace(":", "-").split():
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return frozenset(out)


# layers (decoder layer_idx, rmsnorm.py's decoder patch tags the MLP) whose MLP forward stays fp16
FP16_LAYERS = _layers(os.environ.get("SOUP_SOLUTION_INT8_FWD_FP16_LAYERS", ""))
# input channels of gate|up per layer kept out of the int8 product and added in fp32 (0: none): the
# TOPK largest max |x| of the layer's fp16 forwards before the switch (see `channels`)
TOPK = int(os.environ.get("SOUP_SOLUTION_INT8_FWD_TOPK", "16"))
_READY = threading.Event()
QBLOCK, ABLOCK = nf4.QBLOCK, nf4.ABLOCK
GU_ROWS, GU_CB = 8, 8  # gate/up decode: rows per program, 64-element blocks per row per iteration
KC_ROWS = 64  # weight rows per program of the top-k column extraction (same launch as the decode)
DN_ROWS, DN_CB = 4, 16  # down decode (rows of 8960: 140 blocks)
QX_ROWS = 4  # token rows per program of the x quantization
# gate|up epilogue: (features, tokens) tile, features per group. bench_int8_fwd --sweep on the T4
# (soup-i1-sweep1, us at 832 / 1440 / 2048 tokens): (32, 64, 224) 498 / 935 / 1212, (32, 64, 448)
# 512 / 1023 / 1223, (32, 128, 448) 562 / 919 / 1138, (64, 32, 448) 598 / 1060 / 1481
TI, TM, PG = 32, 64, 224
TR, TH = 64, 64  # hidden quantization: (features, tokens) tile
OUT_ROWS, OUT_BLOCK = 8, 512  # down epilogue tile


@triton.jit
def _decode_rows(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, out_ptr,
                 scale_ptr, program, blocks_per_row, out_stride, chunk, shift,
                 ROWS: tl.constexpr, CB: tl.constexpr, QBLOCK: tl.constexpr,
                 ABLOCK: tl.constexpr):
    # rows [program*ROWS, +ROWS) of an NF4 (n_rows, blocks_per_row * QBLOCK) weight -> int8 rows of
    # out at dest = (row // chunk) * 2 * chunk + shift + row % chunk, scale[dest] = rowmax / 127.
    # The (ROWS, CB) blocks of an iteration are flattened to ROWS * CB block rows of QBLOCK
    R: tl.constexpr = ROWS * CB
    idx = tl.arange(0, R)
    row = program * ROWS + idx // CB
    off = tl.load(offset_ptr)
    # pass 1: the largest absmax of each row's blocks (the block absmax as nf4.py computes it)
    peak = tl.zeros([R], dtype=tl.float32)
    for c0 in range(0, blocks_per_row, CB):
        cb = c0 + idx % CB
        live = cb < blocks_per_row
        blk = row * blocks_per_row + cb
        q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
        a = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
        a = a + off
        peak = tl.maximum(peak, tl.where(live, tl.abs(a), 0.0))
    rowmax = tl.max(tl.reshape(peak, (ROWS, CB)), axis=1)
    rmax = tl.reshape(tl.broadcast_to(rowmax[:, None], (ROWS, CB)), (R,))
    inv = tl.where(rmax > 0.0, 127.0 / rmax, 0.0)
    dest = (row // chunk) * (2 * chunk) + shift + row % chunk
    first = (idx % CB) == 0
    tl.store(scale_ptr + dest, rmax / 127.0, mask=first)
    # pass 2: decode (nf4.py's word layout) and scale
    words: tl.constexpr = QBLOCK // 8
    j = tl.arange(0, 8)
    for c0 in range(0, blocks_per_row, CB):
        cb = c0 + idx % CB
        live = cb < blocks_per_row
        blk = row * blocks_per_row + cb
        q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
        a = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
        a = a + off
        w = tl.load(words_ptr + blk[:, None].to(tl.int64) * words + tl.arange(0, words)[None, :],
                    mask=live[:, None], other=0)
        codes = (w[:, :, None] >> ((j // 2) * 8 + (1 - j % 2) * 4)[None, None, :]) & 15
        vals = tl.reshape(tl.load(nf4_ptr + codes), (R, QBLOCK)) * a[:, None]
        scaled = tl.floor(vals * inv[:, None] + 0.5)
        q8 = tl.minimum(tl.maximum(scaled, -127.0), 127.0).to(tl.int8)
        col = cb[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
        tl.store(out_ptr + dest[:, None].to(tl.int64) * out_stride + col, q8, mask=live[:, None])


@triton.jit
def _topk_cols(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, idx_ptr, wk_ptr,
               program, n_cols, chunk, shift, KROWS: tl.constexpr, K: tl.constexpr,
               QBLOCK: tl.constexpr, ABLOCK: tl.constexpr):
    # the fp32 decode of columns idx[0:K] of rows [program*KROWS, +KROWS) of an NF4 weight ->
    # wk[dest, 0:K] (dest as in _decode_rows), element by element in nf4.py's word layout
    rows = program * KROWS + tl.arange(0, KROWS)
    j = tl.arange(0, K)
    col = tl.load(idx_ptr + j)
    elem = rows[:, None].to(tl.int64) * n_cols + col[None, :]
    w = tl.load(words_ptr + elem // 8)
    e = (elem % 8).to(tl.int32)
    code = (w >> ((e // 2) * 8 + (1 - e % 2) * 4)) & 15
    blk = elem // QBLOCK
    q = tl.load(qabsmax_ptr + blk).to(tl.int32)
    a = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + blk // ABLOCK)
    a = a + tl.load(offset_ptr)
    val = tl.load(nf4_ptr + code) * a
    dest = (rows // chunk) * (2 * chunk) + shift + rows % chunk
    tl.store(wk_ptr + dest[:, None].to(tl.int64) * K + j[None, :], val)


@triton.jit
def nf4_int8_gu_kernel(w0, qa0, c0, s0, o0, l0, w1, qa1, c1, s1, o1, l1, out_ptr, scale_ptr,
                       idx_ptr, wk_ptr, programs, kprograms, blocks_per_row, out_stride, chunk,
                       ROWS: tl.constexpr, CB: tl.constexpr, QBLOCK: tl.constexpr,
                       ABLOCK: tl.constexpr, K: tl.constexpr, KROWS: tl.constexpr):
    # gate (programs [0, programs)) and up (the next `programs`) into one interleaved int8 buffer;
    # with K, the next 2 * kprograms take the top-k columns of gate and of up in fp32 (wk)
    pid = tl.program_id(0)
    if pid < programs:
        _decode_rows(w0, qa0, c0, s0, o0, l0, out_ptr, scale_ptr, pid, blocks_per_row, out_stride,
                     chunk, 0, ROWS=ROWS, CB=CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK)
    elif pid < 2 * programs:
        _decode_rows(w1, qa1, c1, s1, o1, l1, out_ptr, scale_ptr, pid - programs, blocks_per_row,
                     out_stride, chunk, chunk, ROWS=ROWS, CB=CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK)
    if K:
        if pid >= 2 * programs:
            kp = pid - 2 * programs
            if kp < kprograms:
                _topk_cols(w0, qa0, c0, s0, o0, l0, idx_ptr, wk_ptr, kp, out_stride, chunk, 0,
                           KROWS=KROWS, K=K, QBLOCK=QBLOCK, ABLOCK=ABLOCK)
            else:
                _topk_cols(w1, qa1, c1, s1, o1, l1, idx_ptr, wk_ptr, kp - kprograms, out_stride,
                           chunk, chunk, KROWS=KROWS, K=K, QBLOCK=QBLOCK, ABLOCK=ABLOCK)


@triton.jit
def nf4_int8_rows_kernel(w0, qa0, c0, s0, o0, l0, out_ptr, scale_ptr, blocks_per_row, out_stride,
                         n_rows, ROWS: tl.constexpr, CB: tl.constexpr, QBLOCK: tl.constexpr,
                         ABLOCK: tl.constexpr):
    # one weight, rows in place (chunk = n_rows: dest = row)
    _decode_rows(w0, qa0, c0, s0, o0, l0, out_ptr, scale_ptr, tl.program_id(0), blocks_per_row,
                 out_stride, n_rows, 0, ROWS=ROWS, CB=CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK)


@triton.jit(do_not_specialize=["n_rows"])
def quant_rows_kernel(x_ptr, x_stride, q_ptr, scale_ptr, pos_ptr, xk_ptr, n_rows, padded, n_cols,
                      ROWS: tl.constexpr, BLOCK: tl.constexpr, K: tl.constexpr):
    # x (n_rows, n_cols) fp32/half -> q (padded, n_cols) int8, scale = absmax / 127 per row; rows
    # [n_rows, padded) are zeros with scale 0. With K: the columns with pos[col] >= 0 go to
    # xk[row, pos[col]] in fp32 and count as zeros in q and its scale
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    in_cols = cols < n_cols
    x = tl.load(x_ptr + rows[:, None].to(tl.int64) * x_stride + cols[None, :],
                mask=(rows < n_rows)[:, None] & in_cols[None, :], other=0.0).to(tl.float32)
    if K:
        pos = tl.load(pos_ptr + cols, mask=in_cols, other=-1)
        top = pos >= 0
        tl.store(xk_ptr + rows[:, None].to(tl.int64) * K + pos[None, :], x,
                 mask=(rows < padded)[:, None] & top[None, :])
        x = tl.where(top[None, :], 0.0, x)
    amax = tl.max(tl.abs(x), axis=1)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q8 = tl.minimum(tl.maximum(tl.floor(x * inv[:, None] + 0.5), -127.0), 127.0).to(tl.int8)
    live = rows < padded
    tl.store(q_ptr + rows[:, None].to(tl.int64) * n_cols + cols[None, :], q8,
             mask=live[:, None] & in_cols[None, :])
    tl.store(scale_ptr + rows, amax / 127.0, mask=live)


@triton.jit
def _silu_mul(g, u):
    # silu(gate) * up in fp32 as swiglu.py's forward computes it (one helper for both kernels that
    # need it, so the group maxima and the quantization see the same values)
    return g * tl.sigmoid(g) * u


@triton.jit(do_not_specialize=["feat0", "group0"])  # one compiled variant for every chunk
def int8_gate_up_kernel(acc_ptr, w_scale_ptr, x_scale_ptr, gate_up_ptr, pmax_ptr, xk_ptr, wk_ptr,
                        chunk, inter, tokens, feat0, per_group, group0, TI: tl.constexpr,
                        TM: tl.constexpr, HIDDEN: tl.constexpr, K: tl.constexpr):
    # program (t, grp) of one chunk: tokens [t*TM, +TM) over the chunk's features
    # [grp*per_group, +per_group), TI at a time. acc (2 * chunk, tokens) int32: gate rows then up
    # rows; w_scale the chunk's 2 * chunk row scales in the same order. HIDDEN: pmax_ptr is the
    # SwiGLU output (I, tokens) in half, written here (MODE "gu"), instead of the group maxima.
    # With K: + wk (2 * chunk, K) @ xk (tokens, K)^T in fp32, the top-k input channels' share
    tok = tl.program_id(0) * TM + tl.arange(0, TM)
    grp = tl.program_id(1)
    live = tok < tokens
    s_tok = tl.load(x_scale_ptr + tok, mask=live, other=0.0)
    peak = tl.zeros([TI, TM], dtype=tl.float32)
    up_acc = chunk.to(tl.int64) * tokens
    up_out = inter.to(tl.int64) * tokens
    if K:
        kk = tl.arange(0, K)
        xkt = tl.trans(tl.load(xk_ptr + tok[:, None].to(tl.int64) * K + kk[None, :],
                               mask=live[:, None], other=0.0))
    for f0 in range(0, per_group, TI):
        feat = grp * per_group + f0 + tl.arange(0, TI)
        at = feat[:, None].to(tl.int64) * tokens + tok[None, :]
        g = tl.load(acc_ptr + at, mask=live[None, :], other=0).to(tl.float32)
        u = tl.load(acc_ptr + up_acc + at, mask=live[None, :], other=0).to(tl.float32)
        g = g * tl.load(w_scale_ptr + feat)[:, None] * s_tok[None, :]
        u = u * tl.load(w_scale_ptr + chunk + feat)[:, None] * s_tok[None, :]
        if K:
            wg = tl.load(wk_ptr + feat[:, None].to(tl.int64) * K + kk[None, :])
            wu = tl.load(wk_ptr + (chunk + feat)[:, None].to(tl.int64) * K + kk[None, :])
            g = g + tl.dot(wg, xkt, input_precision="ieee")
            u = u + tl.dot(wu, xkt, input_precision="ieee")
        g = g.to(gate_up_ptr.dtype.element_ty)
        u = u.to(gate_up_ptr.dtype.element_ty)
        dest = (feat0 + feat)[:, None].to(tl.int64) * tokens + tok[None, :]
        tl.store(gate_up_ptr + dest, g, mask=live[None, :])
        tl.store(gate_up_ptr + up_out + dest, u, mask=live[None, :])
        h = _silu_mul(g.to(tl.float32), u.to(tl.float32))
        if HIDDEN:
            tl.store(pmax_ptr + dest, h.to(gate_up_ptr.dtype.element_ty), mask=live[None, :])
        else:
            peak = tl.maximum(peak, tl.abs(h))
    if not HIDDEN:
        tl.store(pmax_ptr + (group0 + grp) * tokens + tok, tl.max(peak, axis=0), mask=live)


@triton.jit
def int8_hidden_kernel(gate_up_ptr, pmax_ptr, scale_ptr, q_ptr, inter, tokens,
                       GROUPS: tl.constexpr, TR: tl.constexpr, TM: tl.constexpr):
    # silu(gate) * up of (gate|up) (2I, tokens) half -> q (tokens, I) int8 with the token's scale:
    # amax[t] = the max of its GROUPS group maxima, scale = amax / 127 (written by the first row tile)
    rows = tl.program_id(0) * TR + tl.arange(0, TR)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    amax = tl.zeros([TM], dtype=tl.float32)
    for grp in range(GROUPS):
        amax = tl.maximum(amax, tl.load(pmax_ptr + grp * tokens + tok, mask=live, other=0.0))
    if tl.program_id(0) == 0:
        tl.store(scale_ptr + tok, amax / 127.0, mask=live)
    at = rows[:, None].to(tl.int64) * tokens + tok[None, :]
    g = tl.load(gate_up_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + inter.to(tl.int64) * tokens + at, mask=live[None, :],
                other=0.0).to(tl.float32)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    h = _silu_mul(g, u)
    q8 = tl.minimum(tl.maximum(tl.floor(h * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    dest = tok[:, None].to(tl.int64) * inter + rows[None, :]
    tl.store(q_ptr + dest, tl.trans(q8), mask=live[:, None])


@triton.jit
def int8_out_kernel(acc_ptr, row_scale_ptr, col_scale_ptr, out_ptr, n_cols,
                    ROWS: tl.constexpr, BLOCK: tl.constexpr):
    # out (rows, n_cols) half = acc int32 * row_scale[row] * col_scale[col]
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    at = rows[:, None].to(tl.int64) * n_cols + cols[None, :]
    acc = tl.load(acc_ptr + at).to(tl.float32)
    v = acc * tl.load(col_scale_ptr + cols)[None, :] * tl.load(row_scale_ptr + rows)[:, None]
    tl.store(out_ptr + at, v.to(out_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------------------------


# When the int8 kernels of both directions are compiled (SOUP_SOLUTION_INT8_COMPILE): "first" (the
# default) by each module's thread at its first eligible MLP forward / backward, "apply" by both
# threads started when apply() patches the MLP (`compile_early`), under the model load
COMPILE = os.environ.get("SOUP_SOLUTION_INT8_COMPILE", "first").strip().lower()


def compile_early():
    """COMPILE == "apply": starts the compile threads of this module and of int8_bwd.py now (each
    once; a module switched off compiles nothing). The switch points stay START_CALLS."""
    if COMPILE != "apply":
        return
    from . import int8_bwd

    if _STATE["on"] and not _READY.is_set():
        _start_compile()
    if int8_bwd._STATE["on"] and not int8_bwd._READY.is_set():
        int8_bwd._start_compile()


def _start_compile():
    """The thread that compiles the kernels (once), while the fp16 forward runs."""
    if _STATE["thread"] is not None:
        return

    def work():
        try:
            if _warm() is None:
                _READY.set()
        except Exception as exc:  # noqa: BLE001 - stay on the fp16 forward
            print(f"[solution] int8 forward kernels not compiled ({type(exc).__name__}: {exc}); "
                  "the fp16 forward stays", flush=True)

    thread = threading.Thread(target=work, name="soup-int8-fwd-compile", daemon=True)
    _STATE["thread"] = thread
    thread.start()
    atexit.register(thread.join, 120.0)  # never leave the interpreter while it compiles


def compile_now():
    """Compiles the kernels in the calling thread, so that the next eligible call takes the int8
    path (tests and benchmarks)."""
    if not _READY.is_set() and _warm() is None:
        _READY.set()
    _STATE["calls"] = max(_STATE["calls"], START_CALLS)


def shapes_ok(hidden, inter, down_rows):
    """The kernels' tiles and `_int_mm`'s constraints at these MLP sizes."""
    chunk = inter // max(CHUNKS, 1)
    return (CHUNKS >= 1 and inter % CHUNKS == 0 and chunk % GU_ROWS == 0 and chunk % PG == 0
            and PG % TI == 0 and inter % TR == 0 and inter % 8 == 0 and hidden % 8 == 0
            and hidden % QBLOCK == 0 and inter % QBLOCK == 0 and down_rows % DN_ROWS == 0
            and hidden % OUT_BLOCK == 0 and hidden <= 8192 and inter % KC_ROWS == 0
            and TOPK in (0, 16, 32, 64) and hidden >= TOPK)


def observing():
    """Whether the forwards still run fp16 before the switch (when `observe` collects the channel
    maxima for TOPK); call after `usable` counted the forward."""
    return TOPK > 0 and _STATE["on"] and _STATE["calls"] <= START_CALLS


def usable(tokens):
    """Whether the int8 path takes this MLP forward (the caller has checked the plan: NF4 weights in
    nf4.py's layout, the shapes, autograd on, (gate|up) kept): switched on, more than 16 tokens, a
    multiple of OUT_ROWS, START_CALLS such forwards ran before (the kernels compile meanwhile, from
    the COMPILE_CALLS-th)."""
    if not _STATE["on"] or tokens <= 16 or tokens % OUT_ROWS:
        return False
    _STATE["calls"] += 1
    if not _READY.is_set() and _STATE["calls"] >= min(COMPILE_CALLS, START_CALLS + 1):
        _start_compile()
    if _STATE["calls"] <= START_CALLS:
        return False  # fp16 for the first START_CALLS, compiled or not: the switch point is fixed
    if not _READY.is_set():
        _STATE["thread"].join()
    return _READY.is_set()


_DUMMY = {}


def _dummy(device, dtype):
    """A one-element tensor for the pointer arguments a K = 0 launch does not read."""
    key = (device, dtype)
    t = _DUMMY.get(key)
    if t is None:
        t = _DUMMY[key] = torch.zeros(1, dtype=dtype, device=device)
    return t


def _decode_gate_up(gate_ops, up_ops, out_i8, scales, inter, hidden, chunk, idx=None, wk=None):
    """W_gate and W_up (fused_proj._nf4_args operands) -> out_i8 (2I, D) int8 rows interleaved by
    chunk, scales (2I,) fp32 in the same order; with idx (K,) int32, wk (2I, K) fp32 gets the fp32
    decode of those columns in the same row order."""
    programs = inter // GU_ROWS
    k = 0 if idx is None else idx.numel()
    kprograms = inter // KC_ROWS if k else 0
    if not k:
        idx, wk = _dummy(out_i8.device, torch.int32), _dummy(out_i8.device, torch.float32)
    nf4_int8_gu_kernel[(2 * programs + 2 * kprograms,)](
        *gate_ops[:6], *up_ops[:6], out_i8, scales, idx, wk, programs, kprograms, hidden // QBLOCK,
        hidden, chunk, ROWS=GU_ROWS, CB=GU_CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK, K=k, KROWS=KC_ROWS,
        num_warps=4, enable_fp_fusion=False)


def _decode_down(down_ops, out_i8, scales, hidden, inter):
    nf4_int8_rows_kernel[(hidden // DN_ROWS,)](
        *down_ops[:6], out_i8, scales, inter // QBLOCK, inter, hidden,
        ROWS=DN_ROWS, CB=DN_CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4, enable_fp_fusion=False)


def quantize_tokens(x2, n, padded, pos=None):
    """x2 (n, D) -> (x_q (padded, D) int8, s_x (padded,) fp32, x_k): with pos (D,) int32 (channel ->
    its slot among the top k, or -1), x_k (padded, k) fp32 holds those channels and x_q / s_x leave
    them out; else x_k is None."""
    d = x2.shape[1]
    device = x2.device
    x_q = torch.empty((padded, d), dtype=torch.int8, device=device)
    s_x = torch.empty(padded, dtype=torch.float32, device=device)
    k = 0 if pos is None else TOPK
    x_k = torch.empty((padded, k), dtype=torch.float32, device=device) if k else None
    quant_rows_kernel[(triton.cdiv(padded, QX_ROWS),)](
        x2, x2.stride(0), x_q, s_x, pos if k else _dummy(device, torch.int32),
        x_k if k else _dummy(device, torch.float32), n, padded, d, ROWS=QX_ROWS,
        BLOCK=triton.next_power_of_2(d), K=k)
    return x_q, s_x, x_k


def observe(mlp, x2):
    """Before the switch (fp16 forwards): the running max |x| per input channel of this MLP, kept
    on the module (one fp32 (D,) tensor, GPU ops only, no sync)."""
    peak = torch.linalg.vector_norm(x2.detach(), ord=float("inf"), dim=0)
    if peak.dtype != torch.float32:
        peak = peak.float()
    prev = mlp.__dict__.get("_soup_int8_peak")
    if prev is None:
        mlp.__dict__["_soup_int8_peak"] = peak
    else:
        torch.maximum(prev, peak, out=prev)


def channels(mlp, x2):
    """(pos (D,) int32, idx (TOPK,) int32) of this MLP's top-k input channels by the max |x| that
    `observe` collected (this call's x when nothing was collected), fixed for the rest of the run;
    None with TOPK = 0."""
    if TOPK <= 0:
        return None
    found = mlp.__dict__.get("_soup_int8_topk")
    if found is not None:
        return found
    peak = mlp.__dict__.pop("_soup_int8_peak", None)
    if peak is None:
        observe(mlp, x2)
        peak = mlp.__dict__.pop("_soup_int8_peak")
    idx = peak.topk(TOPK).indices.to(torch.int32)
    pos = torch.full((peak.numel(),), -1, dtype=torch.int32, device=peak.device)
    pos[idx.long()] = torch.arange(TOPK, dtype=torch.int32, device=peak.device)
    found = mlp.__dict__["_soup_int8_topk"] = (pos, idx)
    return found


def gate_up_forward(x_q, s_x, gate_ops, up_ops, buffer, half, inter, acquire=None, release=None,
                    hidden=False, idx=None, x_k=None, before_alloc=None):
    """(gate|up) (2I, M) in `half` from the quantized input, and the per-group maxima of
    silu(gate) * up (GROUPS, M) for `down_forward`, or with `hidden` silu(gate) * up itself (I, M) in
    `half` (MODE "gu"). `buffer` is fused_proj's flat half buffer of one (I, D) weight (taken as int8
    bytes); acquire / release: weight_stream ring hooks around the decode (or None). idx / x_k: the
    top-k channels (`channels`) and their values (`quantize_tokens`), added in fp32. before_alloc:
    called once the decode and the first int8 product are queued, right before (gate|up) is
    allocated (fused_proj.py lets the previous layer's (gate|up) copy to the host finish there)."""
    padded, d = x_q.shape
    device = x_q.device
    chunk = inter // CHUNKS
    w = buffer.view(torch.int8)[:2 * inter * d].view(2 * inter, d)
    s_w = torch.empty(2 * inter, dtype=torch.float32, device=device)
    k = 0 if idx is None else idx.numel()
    w_k = torch.empty((2 * inter, k), dtype=torch.float32, device=device) if k else None
    if acquire is not None:
        acquire()
    _decode_gate_up(gate_ops, up_ops, w, s_w, inter, d, chunk, idx, w_k)
    if release is not None:
        release()
    x_t = x_q.t()
    if not k:
        x_k, w_k = _dummy(device, torch.float32), _dummy(device, torch.float32)
    acc = gate_up = pmax = None
    for c in range(CHUNKS):
        rows = w[2 * chunk * c:2 * chunk * (c + 1)]
        acc = torch._int_mm(rows, x_t) if acc is None else torch._int_mm(rows, x_t, out=acc)
        if gate_up is None:
            if before_alloc is not None:
                before_alloc()
            gate_up = torch.empty((2 * inter, padded), dtype=half, device=device)
            pmax = (torch.empty((inter, padded), dtype=half, device=device) if hidden else
                    torch.empty((inter // PG, padded), dtype=torch.float32, device=device))
        int8_gate_up_kernel[(triton.cdiv(padded, TM), chunk // PG)](
            acc, s_w[2 * chunk * c:], s_x, gate_up, pmax, x_k, w_k[2 * chunk * c:] if k else w_k,
            chunk, inter, padded, c * chunk, PG, c * (chunk // PG), TI=TI, TM=TM, HIDDEN=hidden,
            K=k)
    return gate_up, pmax


def down_forward(gate_up, pmax, down_ops, buffer, half, hidden, acquire=None, release=None):
    """The MLP output (M, D) in `half` from (gate|up) and its group maxima."""
    inter2, padded = gate_up.shape
    inter = inter2 // 2
    device = gate_up.device
    h_q = torch.empty((padded, inter), dtype=torch.int8, device=device)
    s_h = torch.empty(padded, dtype=torch.float32, device=device)
    int8_hidden_kernel[(inter // TR, triton.cdiv(padded, TH))](
        gate_up, pmax, s_h, h_q, inter, padded, GROUPS=pmax.shape[0], TR=TR, TM=TH)
    w = buffer.view(torch.int8)[:hidden * inter].view(hidden, inter)
    s_w = torch.empty(hidden, dtype=torch.float32, device=device)
    if acquire is not None:
        acquire()
    _decode_down(down_ops, w, s_w, hidden, inter)
    if release is not None:
        release()
    acc = torch._int_mm(h_q, w.t())
    del h_q
    out = torch.empty((padded, hidden), dtype=half, device=device)
    int8_out_kernel[(padded // OUT_ROWS, hidden // OUT_BLOCK)](
        acc, s_h, s_w, out, hidden, ROWS=OUT_ROWS, BLOCK=OUT_BLOCK)
    return out


def _warm():
    """The five kernels in the classes training launches (warmup.compile_kernel)."""
    from .warmup import HIDDEN, INTERMEDIATE, SKIPPED, TOKENS, compile_kernel, training_half_dtype

    if not _STATE["on"]:
        return SKIPPED
    tokens = -(-TOKENS // 16) * 16  # the MLP's token count is a multiple of 16 (unpad.py's tail)
    half, f32, i8, i32, u8 = (training_half_dtype(), torch.float32, torch.int8, torch.int32,
                              torch.uint8)
    w_args = (i32, u8, f32, f32, f32, f32)
    chunk = INTERMEDIATE // CHUNKS
    k = max(TOPK, 0)
    compile_kernel(quant_rows_kernel, f32, HIDDEN, i8, f32, i32, f32, tokens - 3, tokens, HIDDEN,
                   ROWS=QX_ROWS, BLOCK=triton.next_power_of_2(HIDDEN), K=k)
    compile_kernel(nf4_int8_gu_kernel, *w_args, *w_args, i8, f32, i32, f32,
                   INTERMEDIATE // GU_ROWS, INTERMEDIATE // KC_ROWS if k else 0, HIDDEN // QBLOCK,
                   HIDDEN, chunk, ROWS=GU_ROWS, CB=GU_CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK, K=k,
                   KROWS=KC_ROWS, num_warps=4, enable_fp_fusion=False)
    gu_only = MODE == "gu"
    compile_kernel(int8_gate_up_kernel, i32, f32, f32, half, half if gu_only else f32, f32, f32,
                   chunk, INTERMEDIATE, tokens, 0, PG, 0, TI=TI, TM=TM, HIDDEN=gu_only, K=k)
    if gu_only:
        return None
    compile_kernel(int8_hidden_kernel, half, f32, f32, i8, INTERMEDIATE, tokens,
                   GROUPS=INTERMEDIATE // PG, TR=TR, TM=TH)
    compile_kernel(nf4_int8_rows_kernel, *w_args, i8, f32, INTERMEDIATE // QBLOCK, INTERMEDIATE,
                   HIDDEN, ROWS=DN_ROWS, CB=DN_CB, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4,
                   enable_fp_fusion=False)
    compile_kernel(int8_out_kernel, i32, f32, f32, half, HIDDEN, ROWS=OUT_ROWS, BLOCK=OUT_BLOCK)
    return None
