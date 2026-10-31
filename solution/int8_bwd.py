"""int8 tensor cores for the MLP's input-gradient GEMMs in the backward (night task N2,
SOUP_SOLUTION_INT8_BWD, on by default (organizers' answer 29 allows int8 for the frozen linear layers;
=0 turns it off; the bitwise tests of the fp16 path pin it off in solution/conftest.py, the int8 path
has its own tests in test_int8_bwd.py): the forward is untouched, only the noise of the gradient
that flows back through the frozen MLP weights changes).

On the T4 an int8 GEMM (IMMA, `torch._int_mm` -> cuBLASLt -> CUTLASS i8816) runs the backward's two
big MLP products 1.6-1.9x faster than fp16 at the step's token counts (solution/bench_int8.py,
docs/RESEARCH.md): it does half the bytes and draws less power per operation, so the card throttles
less. The MLP backward of fused_proj.py becomes, with M the step's tokens (padded to 8):

1. grad (M, D) half -> per-token absmax int8 (torch ops: a small pass): g_q, s_g;
2. W_down (D, I) NF4 -> W_down^T (I, D) int8 with one scale per input feature, the column maxima
   of the weight (`nf4_int8_t_kernel`, the NF4 decode of nf4.py, then round(v * 127 / colmax));
3. acc_h (I, M) int32 = W_down^T_q @ g_q^T (`torch._int_mm`, "TN");
4. `int8_swiglu_bwd_kernel`: dL/dhidden = acc_h * s_w[i] * s_g[m] in fp32, the SwiGLU derivative
   with the saved (gate|up), dL/d(gate|up) written over (gate|up) in half, and every token's absmax
   of it per group of features (no atomics);
5. dL/d(gate|up) (2I, M) -> (M, 2I) int8 with the per-token scales, the max over the groups
   (`quant_t_int8_kernel`);
6. W_gate, W_up (I, D) NF4 -> [W_gate^T | W_up^T] (D, 2I) int8, one scale per input feature shared
   by both (the contraction runs over gate and up together);
7. acc_x (M, D) int32 = dgu_q @ W^T_q^T (`torch._int_mm`), and the fp32 input gradient
   acc_x * s_g2[m] * s_w2[k] (torch ops).
The weights' column maxima are computed once per weight and run (from its fp16 rebuild; the rules
allow precomputed auxiliary tensors for the whole run) and kept on the module. The int8 weights
take the place of the fp16 rebuild in fused_proj's shared buffer (the (D, 2I) int8 copy is exactly
the size of one (I, D) fp16 weight), so no memory is added there; the int32 product of step 3 is
2x the half buffer it replaces (36 KB per token, transient).

Numerics: per-token x per-feature absmax scaling, round to nearest; the forward, the loss, the first
step and the eval do not change; the input gradient of the MLP gets int8 quantization noise
(measured against the fp16 backward in test_int8_bwd.py).
"""

import atexit
import os
import threading

import torch
import triton
import triton.language as tl

from . import mma_gemm, nf4


_STATE = {"on": os.environ.get("SOUP_SOLUTION_INT8_BWD", "1").strip().lower()
          not in ("0", "false", "no", ""), "thread": None, "calls": 0}
# SOUP_SOLUTION_MMA_INT8_BWD=1: steps 3 and 4 below in one tensor-core kernel of mma_gemm.py (B3:
# Triton's own Turing MMA lowering for our kernels; off until the organizers answer the wave-3
# question): bit-for-bit the same gradients, without the (I, tokens) int32 product in memory
_STATE["mma"] = os.environ.get("SOUP_SOLUTION_MMA_INT8_BWD", "0").strip().lower() not in ("0", "false", "no", "")
# The three kernels are compiled by a thread started at the first MLP backward, not by the warm-up
# (the first step waits for it: five kernels there put the first step 1.5 s later on the T4). The
# first START_CALLS MLP backwards of the run (four steps of the 28-layer model) take the fp16 path
# meanwhile; the next one waits for the thread if it is not done, so which backwards run int8 never
# depends on timing.
# with SOUP_SOLUTION_MMA_INT8_BWD six steps: the tensor-core kernels take longer to compile (soup-b3-abba3:
# the switch after four steps waited ~1 s for the thread)
START_CALLS = int(os.environ.get("SOUP_SOLUTION_INT8_START_CALLS", "168" if _STATE["mma"] else "112"))
_READY = threading.Event()
QBLOCK, ABLOCK = nf4.QBLOCK, nf4.ABLOCK
W_ROWS = 128  # weight rows per program of nf4_int8_t_kernel (128-byte rows of the transposed store)
# SwiGLU backward: (features, tokens) tile and feature groups. bench_int8_mlp.py on the T4: 0.44 ms
# per layer at 832 tokens and 0.83 ms at 1440 with (32, 64, 10); (32, 32, 10) 0.51 / 0.89; (16, 64,
# 20) 0.61 / 1.10; (8, 128, *) 1.1-1.8
TILE_I, TILE_M, GROUPS = 32, 64, 10
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
def int8_swiglu_bwd_kernel(acc_ptr, w_scale_ptr, g_scale_ptr, gate_up_ptr, pmax_ptr, inter, tokens,
                           per_group, TI: tl.constexpr, TM: tl.constexpr):
    # program (t, grp): tokens [t*TM, +TM) over the features [grp*per_group, +per_group), TI at a
    # time. dL/dhidden from the int32 product and its two scale vectors, the SwiGLU derivative over
    # (gate | up) stored as (2I, tokens), written back over it in half; the group's per-token absmax
    # of what was written goes to pmax[grp, token] (no atomics: quant_t_int8_kernel takes the max)
    tok = tl.program_id(0) * TM + tl.arange(0, TM)
    grp = tl.program_id(1)
    live = tok < tokens
    s_tok = tl.load(g_scale_ptr + tok, mask=live, other=0.0)
    peak = tl.zeros([TI, TM], dtype=tl.float32)  # elementwise over the loop, reduced once at the end
    up_shift = inter.to(tl.int64) * tokens
    for f0 in range(0, per_group, TI):
        feat = grp * per_group + f0 + tl.arange(0, TI)
        at = feat[:, None].to(tl.int64) * tokens + tok[None, :]
        dhid = tl.load(acc_ptr + at, mask=live[None, :], other=0).to(tl.float32)
        dhid = dhid * tl.load(w_scale_ptr + feat)[:, None] * s_tok[None, :]
        g = tl.load(gate_up_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
        u = tl.load(gate_up_ptr + up_shift + at, mask=live[None, :], other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu = g * sig
        d_u = (dhid * silu).to(gate_up_ptr.dtype.element_ty)
        d_g = (dhid * u * (sig + silu * (1.0 - sig))).to(gate_up_ptr.dtype.element_ty)
        tl.store(gate_up_ptr + at, d_g, mask=live[None, :])
        tl.store(gate_up_ptr + up_shift + at, d_u, mask=live[None, :])
        peak = tl.maximum(peak, tl.maximum(tl.abs(d_g.to(tl.float32)), tl.abs(d_u.to(tl.float32))))
    tl.store(pmax_ptr + grp * tokens + tok, tl.max(peak, axis=0), mask=live)


@triton.jit
def quant_t_int8_kernel(src_ptr, pmax_ptr, amax_ptr, q_ptr, n_rows, tokens,
                        GROUPS: tl.constexpr, TR: tl.constexpr, TM: tl.constexpr):
    # src (n_rows, tokens) half -> q (tokens, n_rows) int8; token t's scale is amax[t] / 127 with
    # amax[t] the max of its GROUPS partial maxima (written out once, by the first row tile)
    rows = tl.program_id(0) * TR + tl.arange(0, TR)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    amax = tl.zeros([TM], dtype=tl.float32)
    for grp in range(GROUPS):
        amax = tl.maximum(amax, tl.load(pmax_ptr + grp * tokens + tok, mask=live, other=0.0))
    if tl.program_id(0) == 0:
        tl.store(amax_ptr + tok, amax, mask=live)
    x = tl.load(src_ptr + rows[:, None].to(tl.int64) * tokens + tok[None, :], mask=live[None, :],
                other=0.0).to(tl.float32)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q8 = tl.minimum(tl.maximum(tl.floor(x * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    dest = tok[:, None].to(tl.int64) * n_rows + rows[None, :]
    tl.store(q_ptr + dest, tl.trans(q8), mask=live[:, None])


# ---------------------------------------------------------------------------------------------


def _quant_rows(x):
    """Per-row absmax int8 of a 2-D half tensor: (int8 rows, fp32 scales = absmax / 127); a row of
    zeros (the padding rows) gets scale 0 and zeros."""
    amax = x.abs().amax(dim=1).float()
    inv = torch.where(amax > 0, 127.0 / amax.clamp_min(1e-30), torch.zeros_like(amax))
    q = torch.floor(x.float().mul_(inv[:, None]).add_(0.5)).clamp_(-127, 127).to(torch.int8)
    return q, amax / 127.0


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


def _start_compile():
    """The thread that compiles the kernels (once); the GIL is mostly free while the training waits
    on the GPU, and the fp16 backward runs meanwhile."""
    if _STATE["thread"] is not None:
        return

    def work():
        try:
            if _warm() is None:
                _READY.set()
        except Exception as exc:  # noqa: BLE001 - stay on the fp16 backward
            print(f"[solution] int8 backward kernels not compiled ({type(exc).__name__}: {exc}); "
                  "the fp16 backward stays", flush=True)

    thread = threading.Thread(target=work, name="soup-int8-compile", daemon=True)
    _STATE["thread"] = thread
    thread.start()
    atexit.register(thread.join, 120.0)  # never leave the interpreter while it compiles


def compile_now():
    """Compiles the kernels in the calling thread, so that the next eligible call takes the int8
    path (tests and benchmarks)."""
    if not _READY.is_set() and _warm() is None:
        _READY.set()
    _STATE["calls"] = max(_STATE["calls"], START_CALLS)


def usable(plan, tokens):
    """Whether the int8 path takes this MLP backward: switched on, the weights are NF4 in nf4.py's
    layout, the shapes meet `_int_mm` (more than 16 rows, K and N multiples of 8) and the kernels'
    tiles, START_CALLS such backwards ran before (the kernels compile meanwhile)."""
    if not _STATE["on"] or tokens <= 16 or tokens % 8:
        return False
    for base in (plan.gate, plan.up, plan.down):
        if base.out_features % W_ROWS or base.in_features % QBLOCK:
            return False
        if not nf4._fast_path(base.weight.data, base.weight.quant_state, plan.half):
            return False
    inter = plan.gate.out_features
    if not (plan.down.in_features % 8 == 0 and plan.down.out_features % 8 == 0
            and inter % (GROUPS * TILE_I) == 0 and (2 * inter) % TQ_R == 0):
        return False
    _STATE["calls"] += 1
    if not _READY.is_set():
        _start_compile()
    if _STATE["calls"] <= START_CALLS:
        return False  # fp16 for the first START_CALLS, compiled or not: the switch point is fixed
    if not _READY.is_set():
        _STATE["thread"].join()
    return _READY.is_set()


def _mma_bwd(inter, d):
    """Whether steps 3-4 take mma_gemm.py's tensor-core kernel (switch, sm_75, its tile sizes)."""
    return (_STATE["mma"] and torch.cuda.get_device_capability() == (7, 5)
            and mma_gemm.int8_swiglu_bwd_usable(inter, d))


def mlp_backward(plan, gate_up, grad16, buffer):
    """The fused MLP's backward from dL/dout (tokens, D) half and the saved (gate|up) (2I, tokens):
    returns dL/dx (tokens, D) fp32; (gate|up) is overwritten with its gradient. `buffer` is
    fused_proj's flat half buffer for one (I, D) weight (taken as int8 bytes)."""
    tokens, d = grad16.shape
    inter = plan.gate.out_features
    device = grad16.device
    i8 = buffer.view(torch.int8)
    # 1. the incoming gradient per token (torch ops: a small pass, no kernel to compile)
    g_q, s_g = _quant_rows(grad16)
    # 2-3. dL/dhidden^T (I, tokens) in int32 through W_down^T int8
    inv_d, scale_d = _column_scales(plan.down, plan.half, buffer)
    w_down_t = i8[:inter * d].view(inter, d)
    _decode_t(plan.down, inv_d, w_down_t, d, 0)
    if _mma_bwd(inter, d):
        # 3-4 in one kernel (mma_gemm.py): the int32 product stays in registers; pmax per feature tile
        pmax = mma_gemm.int8_swiglu_bwd(w_down_t, g_q, scale_d, s_g, gate_up)
        # one partial maximum per token for the quantization below: it reads every partial once per
        # program, and the fused kernel leaves I / BLOCK_I of them (soup-b3-abba3: 140 of them cost
        # ~14 ms per step); the maximum is exact, so the scales are the same
        pmax = pmax.amax(0, keepdim=True)
    else:
        acc_h = torch._int_mm(w_down_t, g_q.t())
        # 4. the SwiGLU backward in place over (gate|up), per-token absmax per feature group
        pmax = torch.empty((GROUPS, tokens), dtype=torch.float32, device=device)
        int8_swiglu_bwd_kernel[(triton.cdiv(tokens, TILE_M), GROUPS)](
            acc_h, scale_d, s_g, gate_up, pmax, inter, tokens, inter // GROUPS, TI=TILE_I, TM=TILE_M)
        del acc_h
    # 5. dL/d(gate|up) per token, transposed to (tokens, 2I)
    amax = torch.empty(tokens, dtype=torch.float32, device=device)
    dgu_q = torch.empty((tokens, 2 * inter), dtype=torch.int8, device=device)
    quant_t_int8_kernel[(2 * inter // TQ_R, triton.cdiv(tokens, TQ_M))](
        gate_up, pmax, amax, dgu_q, 2 * inter, tokens, GROUPS=pmax.shape[0], TR=TQ_R, TM=TQ_M)
    # 6. [W_gate^T | W_up^T] (D, 2I) int8 with one scale per input feature for both
    inv_g, scale_g = _column_scales(plan.gate, plan.half, buffer)
    inv_u, scale_u = _column_scales(plan.up, plan.half, buffer)
    scale_gu = torch.maximum(scale_g, scale_u)
    inv_gu = _inverse(scale_gu * 127.0)
    w_gu_t = i8[:d * 2 * inter].view(d, 2 * inter)
    _decode_t(plan.gate, inv_gu, w_gu_t, 2 * inter, 0)
    _decode_t(plan.up, inv_gu, w_gu_t, 2 * inter, inter)
    # 7. dL/dx (tokens, D) through the int32 product and the two scale vectors (torch ops)
    if _mma_bwd(inter, d) and mma_gemm.int8_scaled_mm_usable(d, 2 * inter):
        # the product and both scalings in one kernel (mma_gemm.py), the same values
        return mma_gemm.int8_scaled_mm(dgu_q, w_gu_t, amax / 127.0, scale_gu)
    acc_x = torch._int_mm(dgu_q, w_gu_t.t())
    grad_x = acc_x.float().mul_((amax / 127.0)[:, None]).mul_(scale_gu[None, :])
    return grad_x


def _warm():
    """warmup.py registration of the three kernels in the classes training launches."""
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
    groups = GROUPS
    if not _mma_bwd(INTERMEDIATE, HIDDEN):
        compile_kernel(int8_swiglu_bwd_kernel, i32, f32, f32, half, f32, INTERMEDIATE, tokens,
                       INTERMEDIATE // GROUPS, TI=TILE_I, TM=TILE_M)
    else:
        mma_gemm.install()
        bi, bt, bk, warps, group = mma_gemm.SWIGLU_BWD_CFG
        compile_kernel(mma_gemm.tc_int8_swiglu_bwd_kernel, i8, i8, f32, f32, half, f32, INTERMEDIATE,
                       tokens, HIDDEN, BLOCK_I=bi, BLOCK_T=bt, BLOCK_K=bk, GROUP_I=group,
                       num_warps=warps, num_stages=1)
        groups = 1
        if mma_gemm.int8_scaled_mm_usable(HIDDEN, 2 * INTERMEDIATE):
            bm, bn, bk, warps, group = mma_gemm.SCALED_MM_CFG
            compile_kernel(mma_gemm.tc_int8_scaled_mm_kernel, i8, i8, f32, f32, f32, tokens, HIDDEN,
                           2 * INTERMEDIATE, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group,
                           num_warps=warps, num_stages=1)
    compile_kernel(quant_t_int8_kernel, half, f32, f32, i8, 2 * INTERMEDIATE, tokens, GROUPS=groups,
                   TR=TQ_R, TM=TQ_M)
    return None
