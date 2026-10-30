"""int8 tensor cores for the MLP's input-gradient GEMMs in the backward (night task N2,
SOUP_SOLUTION_INT8_BWD=1, off by default: the forward is untouched, only the noise of the gradient
that flows back through the frozen MLP weights changes).

On the T4 an int8 GEMM (IMMA, `torch._int_mm` -> cuBLASLt -> CUTLASS i8816) runs the backward's two
big MLP products 1.6-1.9x faster than fp16 at the step's token counts (solution/bench_int8.py,
docs/RESEARCH.md): it does half the bytes and draws less power per operation, so the card throttles
less. The MLP backward of fused_proj.py becomes, with M the step's tokens (padded to 8):

1. grad (M, D) half -> per-token absmax int8 (`quant_rows_int8_kernel`): g_q, s_g;
2. W_down (D, I) NF4 -> W_down^T (I, D) int8 with one scale per input feature, the column maxima
   of the weight (`nf4_int8_t_kernel`, the NF4 decode of nf4.py, then round(v * 127 / colmax));
3. acc_h (I, M) int32 = W_down^T_q @ g_q^T (`torch._int_mm`, "TN");
4. `int8_swiglu_bwd_kernel`: dL/dhidden = acc_h * s_w[i] * s_g[m] in fp32, the SwiGLU derivative
   with the saved (gate|up), dL/d(gate|up) written over (gate|up) in half, and every token's absmax
   of it by an atomic max;
5. dL/d(gate|up) (2I, M) -> (M, 2I) int8 with those per-token scales (`quant_t_int8_kernel`);
6. W_gate, W_up (I, D) NF4 -> [W_gate^T | W_up^T] (D, 2I) int8, one scale per input feature shared
   by both (the contraction runs over gate and up together);
7. acc_x (M, D) int32 = dgu_q @ W^T_q^T (`torch._int_mm`), and `int8_scale_kernel` gives the fp32
   input gradient acc_x * s_g2[m] * s_w2[k].
The weights' column maxima are computed once per weight and run (from its fp16 rebuild; the rules
allow precomputed auxiliary tensors for the whole run) and kept on the module. The int8 weights
take the place of the fp16 rebuild in fused_proj's shared buffer (the (D, 2I) int8 copy is exactly
the size of one (I, D) fp16 weight), so no memory is added there; the int32 product of step 3 is
2x the half buffer it replaces (36 KB per token, transient).

Numerics: per-token x per-feature absmax scaling, round to nearest; the forward, the loss, the first
step and the eval do not change; the input gradient of the MLP gets int8 quantization noise
(measured against the fp16 backward in test_int8_bwd.py).
"""

import os

import torch
import triton
import triton.language as tl

from . import nf4

_STATE = {"on": os.environ.get("SOUP_SOLUTION_INT8_BWD", "0").strip().lower()
          not in ("0", "false", "no", "")}
QBLOCK, ABLOCK = nf4.QBLOCK, nf4.ABLOCK
W_ROWS = 128  # weight rows per program of nf4_int8_t_kernel (128-byte rows of the transposed store)
TILE_I, TILE_M = 32, 128  # (features, tokens) tile of the SwiGLU backward: rows along the tokens
TQ_R, TQ_M = 64, 64  # (rows, tokens) tile of the transposing quantization: 128-byte loads, 64-byte stores


@triton.jit
def nf4_int8_t_kernel(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                      inv_ptr, out_ptr, blocks_per_row, out_stride, row_offset,
                      ROWS: tl.constexpr, QBLOCK: tl.constexpr, ABLOCK: tl.constexpr):
    # W (n_rows, n_cols) NF4 -> out[col, row_offset + row] = round(W[row, col] * inv[col]) in int8.
    # Program (r, c): rows [r*ROWS, +ROWS) of the c-th 64-column block, i.e. one absmax block per
    # row, read as 8 int32 words of packed codes each (nf4.py's decode), stored transposed
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cblk = tl.program_id(1)
    blk = rows * blocks_per_row + cblk
    q = tl.load(qabsmax_ptr + blk).to(tl.int32)
    absmax = tl.load(code2_ptr + q) * tl.load(absmax2_ptr + blk // ABLOCK)
    absmax = absmax + tl.load(offset_ptr)
    words: tl.constexpr = QBLOCK // 8
    w = tl.load(words_ptr + blk[:, None].to(tl.int64) * words + tl.arange(0, words)[None, :])
    j = tl.arange(0, 8)
    codes = (w[:, :, None] >> ((j // 2) * 8 + (1 - j % 2) * 4)[None, None, :]) & 15
    vals = tl.reshape(tl.load(nf4_ptr + codes), (ROWS, QBLOCK)) * absmax[:, None]
    cols = cblk * QBLOCK + tl.arange(0, QBLOCK)
    scaled = tl.floor(vals * tl.load(inv_ptr + cols)[None, :] + 0.5)
    q8 = tl.minimum(tl.maximum(scaled, -127.0), 127.0).to(tl.int8)
    dest = cols[:, None].to(tl.int64) * out_stride + (row_offset + rows)[None, :]
    tl.store(out_ptr + dest, tl.trans(q8))


@triton.jit
def quant_rows_int8_kernel(x_ptr, q_ptr, scale_ptr, n_cols, BLOCK: tl.constexpr):
    # one row of x (half) -> int8 with scale = absmax / 127 (0 for a row of zeros: the padding rows)
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=ok, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q8 = tl.minimum(tl.maximum(tl.floor(x * inv + 0.5), -127.0), 127.0).to(tl.int8)
    tl.store(q_ptr + row * n_cols + cols, q8, mask=ok)
    tl.store(scale_ptr + row, amax / 127.0)


@triton.jit
def int8_swiglu_bwd_kernel(acc_ptr, w_scale_ptr, g_scale_ptr, gate_up_ptr, amax_ptr, inter, tokens,
                           TI: tl.constexpr, TM: tl.constexpr):
    # dL/dhidden from the int32 product and its two scale vectors, then the SwiGLU derivative over
    # (gate | up) stored as (2I, tokens), written back over it in half; per token, the absmax of
    # what was written, by an atomic max on its fp32 bits (all non-negative: integer order)
    feat = tl.program_id(0) * TI + tl.arange(0, TI)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    at = feat[:, None].to(tl.int64) * tokens + tok[None, :]
    up_at = at + inter.to(tl.int64) * tokens
    dhid = tl.load(acc_ptr + at, mask=live[None, :], other=0).to(tl.float32)
    dhid = dhid * tl.load(w_scale_ptr + feat)[:, None]
    dhid = dhid * tl.load(g_scale_ptr + tok, mask=live, other=0.0)[None, :]
    g = tl.load(gate_up_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + up_at, mask=live[None, :], other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    d_u = (dhid * silu).to(gate_up_ptr.dtype.element_ty)
    d_g = (dhid * u * (sig + silu * (1.0 - sig))).to(gate_up_ptr.dtype.element_ty)
    tl.store(gate_up_ptr + at, d_g, mask=live[None, :])
    tl.store(gate_up_ptr + up_at, d_u, mask=live[None, :])
    peak = tl.maximum(tl.max(tl.abs(d_g.to(tl.float32)), axis=0),
                      tl.max(tl.abs(d_u.to(tl.float32)), axis=0))
    tl.atomic_max(amax_ptr + tok, peak.to(tl.int32, bitcast=True), mask=live)


@triton.jit
def quant_t_int8_kernel(src_ptr, amax_ptr, q_ptr, n_rows, tokens, TR: tl.constexpr, TM: tl.constexpr):
    # src (n_rows, tokens) half -> q (tokens, n_rows) int8, scale of token t = amax[t] / 127
    rows = tl.program_id(0) * TR + tl.arange(0, TR)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    x = tl.load(src_ptr + rows[:, None].to(tl.int64) * tokens + tok[None, :], mask=live[None, :],
                other=0.0).to(tl.float32)
    amax = tl.load(amax_ptr + tok, mask=live, other=0).to(tl.float32, bitcast=True)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q8 = tl.minimum(tl.maximum(tl.floor(x * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    dest = tok[:, None].to(tl.int64) * n_rows + rows[None, :]
    tl.store(q_ptr + dest, tl.trans(q8), mask=live[:, None])


@triton.jit
def int8_scale_kernel(acc_ptr, amax_ptr, w_scale_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    # out (fp32) = acc (int32) * (amax[row] / 127) * w_scale[col], one row per program
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    ok = cols < n_cols
    acc = tl.load(acc_ptr + row * n_cols + cols, mask=ok, other=0).to(tl.float32)
    s_row = tl.load(amax_ptr + row).to(tl.float32, bitcast=True) / 127.0
    out = acc * s_row * tl.load(w_scale_ptr + cols, mask=ok, other=0.0)
    tl.store(out_ptr + row * n_cols + cols, out, mask=ok)


# ---------------------------------------------------------------------------------------------


def _column_scales(base, half, buf):
    """(127 / colmax, colmax / 127) of a frozen 4-bit layer's weight, fp32 (in_features,), made once
    per layer from its fp16 rebuild in `buf` and kept on the module."""
    cached = base.__dict__.get("_soup_int8_scales")
    if cached is not None:
        return cached
    w = buf[:base.out_features * base.in_features].view(base.out_features, base.in_features)
    nf4.dequantize_nf4(base.weight, base.weight.quant_state, half, out=w)
    colmax = w.abs().amax(dim=0).float()
    cached = (_inverse(colmax), colmax / 127.0)
    base.__dict__["_soup_int8_scales"] = cached
    return cached


def _inverse(colmax):
    return torch.where(colmax > 0, 127.0 / colmax.clamp_min(1e-30), torch.zeros_like(colmax))


def _decode_t(base, inv, out_i8, out_stride, row_offset):
    weight = base.weight
    state = weight.quant_state
    state2 = state.state2
    n_rows, n_cols = base.out_features, base.in_features
    nf4_int8_t_kernel[(n_rows // W_ROWS, n_cols // QBLOCK)](
        weight.data.view(-1).view(torch.int32), state.absmax, state2.code, state2.absmax,
        state.offset, state.code, inv, out_i8, n_cols // QBLOCK, out_stride, row_offset,
        ROWS=W_ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4, enable_fp_fusion=False)


def usable(plan, tokens):
    """Whether the int8 path takes this MLP backward: CUDA int8 GEMMs exist, the weights are NF4 in
    nf4.py's layout, the shapes meet `_int_mm` (more than 16 rows, K and N multiples of 8)."""
    if not _STATE["on"] or tokens <= 16 or tokens % 8:
        return False
    for base in (plan.gate, plan.up, plan.down):
        if base.out_features % W_ROWS or base.in_features % QBLOCK:
            return False
        if not nf4._fast_path(base.weight.data, base.weight.quant_state, plan.half):
            return False
    return plan.down.in_features % 8 == 0 and plan.down.out_features % 8 == 0


def mlp_backward(plan, gate_up, grad16, buffer):
    """The fused MLP's backward from dL/dout (tokens, D) half and the saved (gate|up) (2I, tokens):
    returns dL/dx (tokens, D) fp32; (gate|up) is overwritten with its gradient. `buffer` is
    fused_proj's flat half buffer for one (I, D) weight (taken as int8 bytes)."""
    tokens, d = grad16.shape
    inter = plan.gate.out_features
    device = grad16.device
    i8 = buffer.view(torch.int8)
    # 1. the incoming gradient per token
    g_q = torch.empty((tokens, d), dtype=torch.int8, device=device)
    s_g = torch.empty(tokens, dtype=torch.float32, device=device)
    quant_rows_int8_kernel[(tokens,)](grad16, g_q, s_g, d, BLOCK=triton.next_power_of_2(d))
    # 2-3. dL/dhidden^T (I, tokens) in int32 through W_down^T int8
    inv_d, scale_d = _column_scales(plan.down, plan.half, buffer)
    w_down_t = i8[:inter * d].view(inter, d)
    _decode_t(plan.down, inv_d, w_down_t, d, 0)
    acc_h = torch._int_mm(w_down_t, g_q.t())
    # 4. the SwiGLU backward in place over (gate|up), per-token absmax
    amax = torch.zeros(tokens, dtype=torch.int32, device=device)
    int8_swiglu_bwd_kernel[(inter // TILE_I, triton.cdiv(tokens, TILE_M))](
        acc_h, scale_d, s_g, gate_up, amax, inter, tokens, TI=TILE_I, TM=TILE_M)
    del acc_h
    # 5. dL/d(gate|up) per token, transposed to (tokens, 2I)
    dgu_q = torch.empty((tokens, 2 * inter), dtype=torch.int8, device=device)
    quant_t_int8_kernel[(2 * inter // TQ_R, triton.cdiv(tokens, TQ_M))](
        gate_up, amax, dgu_q, 2 * inter, tokens, TR=TQ_R, TM=TQ_M)
    # 6. [W_gate^T | W_up^T] (D, 2I) int8 with one scale per input feature for both
    inv_g, scale_g = _column_scales(plan.gate, plan.half, buffer)
    inv_u, scale_u = _column_scales(plan.up, plan.half, buffer)
    scale_gu = torch.maximum(scale_g, scale_u)
    inv_gu = _inverse(scale_gu * 127.0)
    w_gu_t = i8[:d * 2 * inter].view(d, 2 * inter)
    _decode_t(plan.gate, inv_gu, w_gu_t, 2 * inter, 0)
    _decode_t(plan.up, inv_gu, w_gu_t, 2 * inter, inter)
    # 7. dL/dx (tokens, D) through the int32 product and the two scale vectors
    acc_x = torch._int_mm(dgu_q, w_gu_t.t())
    grad_x = torch.empty((tokens, d), dtype=torch.float32, device=device)
    int8_scale_kernel[(tokens,)](acc_x, amax, scale_gu, grad_x, d, BLOCK=triton.next_power_of_2(d))
    return grad_x


def _warm():
    """warmup.py registration of the five kernels in the classes training launches."""
    from .warmup import HIDDEN, INTERMEDIATE, SKIPPED, TOKENS, compile_kernel, training_half_dtype

    if not _STATE["on"]:
        return SKIPPED
    tokens = -(-TOKENS // 16) * 16  # the MLP's token count is a multiple of 16 (unpad.py's tail)
    half, f32, i8, i32, u8 = (training_half_dtype(), torch.float32, torch.int8, torch.int32,
                              torch.uint8)
    for n_cols, stride in ((INTERMEDIATE, HIDDEN), (HIDDEN, 2 * INTERMEDIATE)):
        for offset in (0, INTERMEDIATE):
            compile_kernel(nf4_int8_t_kernel, i32, u8, f32, f32, f32, f32, f32, i8,
                           n_cols // QBLOCK, stride, offset, ROWS=W_ROWS, QBLOCK=QBLOCK,
                           ABLOCK=ABLOCK, num_warps=4, enable_fp_fusion=False)
    compile_kernel(quant_rows_int8_kernel, half, i8, f32, HIDDEN, BLOCK=triton.next_power_of_2(HIDDEN))
    compile_kernel(int8_swiglu_bwd_kernel, i32, f32, f32, half, i32, INTERMEDIATE, tokens,
                   TI=TILE_I, TM=TILE_M)
    compile_kernel(quant_t_int8_kernel, half, i32, i8, 2 * INTERMEDIATE, tokens, TR=TQ_R, TM=TQ_M)
    compile_kernel(int8_scale_kernel, i32, i32, f32, f32, HIDDEN, BLOCK=triton.next_power_of_2(HIDDEN))
    return None
