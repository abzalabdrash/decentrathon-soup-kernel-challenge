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

from . import int8_glue, nf4


_STATE = {"on": os.environ.get("SOUP_SOLUTION_INT8_BWD", "1").strip().lower()
          not in ("0", "false", "no", ""), "thread": None, "calls": 0}
# The three kernels are compiled by a thread started at the first MLP backward, not by the warm-up
# (the first step waits for it: five kernels there put the first step 1.5 s later on the T4). The
# first START_CALLS MLP backwards of the run (four steps of the 28-layer model) take the fp16 path
# meanwhile; the next one waits for the thread if it is not done, so which backwards run int8 never
# depends on timing.
START_CALLS = int(os.environ.get("SOUP_SOLUTION_INT8_START_CALLS", "112"))
_READY = threading.Event()
QBLOCK, ABLOCK = nf4.QBLOCK, nf4.ABLOCK
W_ROWS = 128  # weight rows per program of nf4_int8_t_kernel (128-byte rows of the transposed store)
# SOUP_SOLUTION_INT8_PAIR (CPU diet): W_gate^T and W_up^T decoded in one launch (nf4_int8_t2_kernel),
# and every decode from fused_proj's cached NF4 operands (no Params4bit attribute reads); =0: as before
_PAIR = {"on": os.environ.get("SOUP_SOLUTION_INT8_PAIR", "1").strip().lower()
         not in ("0", "false", "no", "")}
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
    _decode_t_block(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, inv_ptr,
                    out_ptr, blocks_per_row, out_stride, row_offset, tl.program_id(0),
                    ROWS=ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK)


@triton.jit
def nf4_int8_t2_kernel(w0, qa0, c0, s0, o0, l0, w1, qa1, c1, s1, o1, l1, inv_ptr, out_ptr,
                       blocks_per_row, out_stride, row_programs, ROWS: tl.constexpr,
                       QBLOCK: tl.constexpr, ABLOCK: tl.constexpr):
    # two weights of the same shape in one launch: programs [0, row_programs) decode the first
    # into columns [0, n_rows) of out, the next row_programs the second into [n_rows, 2 n_rows)
    # (W_gate^T | W_up^T of the MLP backward), with nf4_int8_t_kernel's body
    pid = tl.program_id(0)
    if pid < row_programs:
        _decode_t_block(w0, qa0, c0, s0, o0, l0, inv_ptr, out_ptr, blocks_per_row, out_stride, 0,
                        pid, ROWS=ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK)
    else:
        _decode_t_block(w1, qa1, c1, s1, o1, l1, inv_ptr, out_ptr, blocks_per_row, out_stride,
                        row_programs * ROWS, pid - row_programs, ROWS=ROWS, QBLOCK=QBLOCK,
                        ABLOCK=ABLOCK)


@triton.jit
def _decode_t_block(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, inv_ptr,
                    out_ptr, blocks_per_row, out_stride, row_offset, program, ROWS: tl.constexpr,
                    QBLOCK: tl.constexpr, ABLOCK: tl.constexpr):
    rows = program * ROWS + tl.arange(0, ROWS)
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
def int8_swiglu_bwd_kernel(acc_ptr, w_scale_ptr, g_scale_ptr, gate_up_ptr, q_ptr, qs_ptr, pmax_ptr,
                           inter, tokens, per_group, TI: tl.constexpr, TM: tl.constexpr,
                           QGROUP: tl.constexpr, INT8_IN: tl.constexpr):
    # program (t, grp): tokens [t*TM, +TM) over the features [grp*per_group, +per_group), TI at a
    # time. dL/dhidden from the int32 product and its two scale vectors, the SwiGLU derivative over
    # (gate | up) stored as (2I, tokens), written back over it in half; the group's per-token absmax
    # of what was written goes to pmax[grp, token] (no atomics: quant_t_int8_kernel takes the max).
    # INT8_IN: (gate | up) is read from offload_int8.py's int8 form instead (q (I, tokens) int16 holding
    # both halves' int8, qs (2I // QGROUP, tokens) fp32 scales; a TI-row tile lies in one group, so one
    # scale vector per tile), and the derivative goes over the int32 product itself: acc[i, t] becomes
    # the 32-bit word (d_up[i, t] << 16 | d_gate[i, t]) in half bits, stored by the program that read it
    # (quant_t_int8_il_kernel reads that layout); gate_up_ptr only gives the half dtype
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
        if INT8_IN:
            group = (grp * per_group + f0) // QGROUP
            sg = tl.load(qs_ptr + group * tokens + tok, mask=live, other=0.0)
            su = tl.load(qs_ptr + (inter // QGROUP + group) * tokens + tok, mask=live, other=0.0)
            # one int16 per element: the gate's int8 in the low byte, the up's in the high one;
            # rounded to half as offload_int8's rebuild pass writes them: the same values either way
            packed = tl.load(q_ptr + at, mask=live[None, :], other=0).to(tl.int32)
            half: tl.constexpr = gate_up_ptr.dtype.element_ty
            g = (((packed << 24) >> 24).to(tl.float32) * sg[None, :]).to(half).to(tl.float32)
            u = ((packed >> 8).to(tl.float32) * su[None, :]).to(half).to(tl.float32)
        else:
            g = tl.load(gate_up_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
            u = tl.load(gate_up_ptr + up_shift + at, mask=live[None, :], other=0.0).to(tl.float32)
        sig = tl.sigmoid(g)
        silu = g * sig
        d_u = (dhid * silu).to(gate_up_ptr.dtype.element_ty)
        d_g = (dhid * u * (sig + silu * (1.0 - sig))).to(gate_up_ptr.dtype.element_ty)
        if INT8_IN:
            word = ((d_g.to(tl.int16, bitcast=True).to(tl.int32) & 65535)
                    | (d_u.to(tl.int16, bitcast=True).to(tl.int32) << 16))
            tl.store(acc_ptr + at, word, mask=live[None, :])
        else:
            tl.store(gate_up_ptr + at, d_g, mask=live[None, :])
            tl.store(gate_up_ptr + up_shift + at, d_u, mask=live[None, :])
        peak = tl.maximum(peak, tl.maximum(tl.abs(d_g.to(tl.float32)), tl.abs(d_u.to(tl.float32))))
    tl.store(pmax_ptr + grp * tokens + tok, tl.max(peak, axis=0), mask=live)


@triton.jit
def quant_t_int8_il_kernel(src_ptr, pmax_ptr, amax_ptr, q_ptr, inter, tokens, GROUPS: tl.constexpr,
                           TR: tl.constexpr, TM: tl.constexpr, BF16: tl.constexpr):
    # quant_t_int8_kernel for int8_swiglu_bwd_kernel's INT8_IN layout: src (I, tokens) int32 words
    # (d_up << 16 | d_gate, half bits) -> q (tokens, 2I) int8, d_gate in columns [0, I), d_up in
    # [I, 2I): each word is read once, by the program that writes both of its columns
    rows = tl.program_id(0) * TR + tl.arange(0, TR)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    amax = tl.zeros([TM], dtype=tl.float32)
    for grp in range(GROUPS):
        amax = tl.maximum(amax, tl.load(pmax_ptr + grp * tokens + tok, mask=live, other=0.0))
    if tl.program_id(0) == 0:
        tl.store(amax_ptr + tok, amax, mask=live)
    word = tl.load(src_ptr + rows[:, None].to(tl.int64) * tokens + tok[None, :], mask=live[None, :],
                   other=0)
    half: tl.constexpr = tl.bfloat16 if BF16 else tl.float16
    d_g = word.to(tl.int16).to(half, bitcast=True).to(tl.float32)
    d_u = (word >> 16).to(tl.int16).to(half, bitcast=True).to(tl.float32)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q_g = tl.minimum(tl.maximum(tl.floor(d_g * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    q_u = tl.minimum(tl.maximum(tl.floor(d_u * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    dest = tok[:, None].to(tl.int64) * (2 * inter) + rows[None, :]
    tl.store(q_ptr + dest, tl.trans(q_g), mask=live[:, None])
    tl.store(q_ptr + dest + inter, tl.trans(q_u), mask=live[:, None])


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


def _ops(base, half):
    """fused_proj's cached NF4 operands of a 4-bit layer (packed words, absmax codes, state2.code,
    state2.absmax, offset, NF4 levels: no Params4bit attribute read per call), or None."""
    from .fused_proj import _nf4_args

    found = _nf4_args(base, half)
    return None if found is None else found[1][:6]


def _decode_t_pair(gate, up, inv, out_i8, out_stride, half):
    """_decode_t of gate into columns [0, I) and of up into [I, 2I) of out_i8 in one launch
    (SOUP_SOLUTION_INT8_PAIR, on by default; two launches when the cached operands are missing)."""
    ops = (_ops(gate, half), _ops(up, half)) if _PAIR["on"] else (None, None)
    if None in ops or gate.out_features != up.out_features or gate.in_features != up.in_features:
        _decode_t(gate, inv, out_i8, out_stride, 0, half)
        _decode_t(up, inv, out_i8, out_stride, gate.out_features, half)
        return
    n_rows, n_cols = gate.out_features, gate.in_features
    nf4_int8_t2_kernel[(2 * (n_rows // W_ROWS), n_cols // QBLOCK)](
        *ops[0], *ops[1], inv, out_i8, n_cols // QBLOCK, out_stride, n_rows // W_ROWS,
        ROWS=W_ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4, enable_fp_fusion=False)


def _decode_t(base, inv, out_i8, out_stride, row_offset, half=None):
    ops = _ops(base, half) if half is not None and _PAIR["on"] else None
    if ops is not None:
        n_rows, n_cols = base.out_features, base.in_features
        nf4_int8_t_kernel[(n_rows // W_ROWS, n_cols // QBLOCK)](
            *ops, inv, out_i8, n_cols // QBLOCK, out_stride, row_offset,
            ROWS=W_ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4, enable_fp_fusion=False)
        return
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


def mlp_backward(plan, gate_up, grad16, buffer, done=None, padded=None):
    """The fused MLP's backward from dL/dout (tokens, D) half and the saved (gate|up) (2I, tokens):
    returns dL/dx (tokens, D) fp32; (gate|up) is overwritten with its gradient. `buffer` is
    fused_proj's flat half buffer for one (I, D) weight (taken as int8 bytes). A uint8 `gate_up` is
    offload_int8.py's int8 form of it (offload_int8.raw): the SwiGLU backward reads that and writes
    the gradient over the int32 product of step 3 (interleaved per element: no new buffer)."""
    tokens, d = grad16.shape
    if padded is not None:  # grad16: the unpadded gradient in its own dtype (int8_glue pads and rounds)
        tokens = padded
    inter = plan.gate.out_features
    device = grad16.device
    i8 = buffer.view(torch.int8)
    q = qs = gate_up
    qgroup, int8_in = 1, gate_up.dtype == torch.uint8
    if int8_in:
        from . import offload_int8

        if gate_up.numel() != offload_int8.wire_bytes(2 * inter, tokens):
            raise ValueError(f"int8 (gate|up) of {gate_up.numel()} bytes for {tokens} tokens")
        qgroup = offload_int8.GROUP
        if qgroup % TILE_I or (inter // GROUPS) % TILE_I or inter % qgroup:  # a tile in one group
            gate_up = offload_int8.Codec((2 * inter, tokens), plan.half).decode(gate_up)
            q = qs = gate_up
            int8_in = False
        else:
            q, qs = offload_int8.views(gate_up, 2 * inter, tokens)
    # 1. the incoming gradient per token (int8_glue's kernel, or the torch ops)
    if padded is not None:
        g_q, s_g = int8_glue.quant_rows(grad16, rows=padded, half=plan.half)
    else:
        g_q, s_g = int8_glue.quant_rows(grad16) if int8_glue.usable(grad16) else _quant_rows(grad16)
    # 2-3. dL/dhidden^T (I, tokens) in int32 through W_down^T int8
    inv_d, scale_d = _column_scales(plan.down, plan.half, buffer)
    w_down_t = i8[:inter * d].view(inter, d)
    _decode_t(plan.down, inv_d, w_down_t, d, 0, plan.half)
    if done is not None:  # W_down's last read is queued (weight_stream: its slot may load the next)
        done(plan.down)
    acc_h = torch._int_mm(w_down_t, g_q.t())
    # 4. the SwiGLU backward in place over (gate|up), per-token absmax per feature group (from the
    # int8 form: over the int32 product, interleaved, no new buffer)
    pmax = torch.empty((GROUPS, tokens), dtype=torch.float32, device=device)
    dgu = acc_h if int8_in else gate_up
    int8_swiglu_bwd_kernel[(triton.cdiv(tokens, TILE_M), GROUPS)](
        acc_h, scale_d, s_g, acc_h.view(plan.half) if int8_in else gate_up, q, qs, pmax, inter,
        tokens, inter // GROUPS, TI=TILE_I, TM=TILE_M, QGROUP=qgroup, INT8_IN=int8_in)
    del acc_h, q, qs, gate_up
    # 5. dL/d(gate|up) per token, transposed to (tokens, 2I)
    amax = torch.empty(tokens, dtype=torch.float32, device=device)
    dgu_q = torch.empty((tokens, 2 * inter), dtype=torch.int8, device=device)
    if int8_in:
        quant_t_int8_il_kernel[(inter // TQ_R, triton.cdiv(tokens, TQ_M))](
            dgu, pmax, amax, dgu_q, inter, tokens, GROUPS=GROUPS, TR=TQ_R, TM=TQ_M,
            BF16=plan.half == torch.bfloat16)
    else:
        quant_t_int8_kernel[(2 * inter // TQ_R, triton.cdiv(tokens, TQ_M))](
            dgu, pmax, amax, dgu_q, 2 * inter, tokens, GROUPS=GROUPS, TR=TQ_R, TM=TQ_M)
    del dgu
    # 6. [W_gate^T | W_up^T] (D, 2I) int8 with one scale per input feature for both
    inv_g, scale_g = _column_scales(plan.gate, plan.half, buffer)
    inv_u, scale_u = _column_scales(plan.up, plan.half, buffer)
    if int8_glue.ON:  # kept on the layer: frozen weights
        scale_gu, inv_gu = int8_glue.gate_up_scales(plan.gate, scale_g, scale_u, _inverse)
    else:
        scale_gu = torch.maximum(scale_g, scale_u)
        inv_gu = _inverse(scale_gu * 127.0)
    w_gu_t = i8[:d * 2 * inter].view(d, 2 * inter)
    _decode_t_pair(plan.gate, plan.up, inv_gu, w_gu_t, 2 * inter, plan.half)
    if done is not None:
        done(plan.gate)
        done(plan.up)
    # 7. dL/dx (tokens, D) through the int32 product and the two scale vectors (torch ops)
    acc_x = torch._int_mm(dgu_q, w_gu_t.t())
    if int8_glue.ON:
        return int8_glue.dequant_rows_cols(acc_x, amax, scale_gu)
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
    compile_kernel(int8_swiglu_bwd_kernel, i32, f32, f32, half, half, half, f32, INTERMEDIATE, tokens,
                   INTERMEDIATE // GROUPS, TI=TILE_I, TM=TILE_M, QGROUP=1, INT8_IN=False)
    compile_kernel(quant_t_int8_kernel, half, f32, f32, i8, 2 * INTERMEDIATE, tokens, GROUPS=GROUPS,
                   TR=TQ_R, TM=TQ_M)
    if _PAIR["on"]:
        compile_kernel(nf4_int8_t2_kernel, i32, u8, f32, f32, f32, f32, i32, u8, f32, f32, f32, f32,
                       f32, i8, HIDDEN // QBLOCK, 2 * INTERMEDIATE, INTERMEDIATE // W_ROWS,
                       ROWS=W_ROWS, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=4,
                       enable_fp_fusion=False)
    int8_glue.warm(half, HIDDEN)
    return None


def warm_int8_in(group):
    """The INT8_IN variants (offload_int8.py's int8 form read directly), compiled by offload_int8's own
    thread: this module's thread compiles what main's does."""
    from .warmup import INTERMEDIATE, TOKENS, compile_kernel, training_half_dtype

    tokens = -(-TOKENS // 16) * 16
    half, f32, i8, i32 = training_half_dtype(), torch.float32, torch.int8, torch.int32
    compile_kernel(int8_swiglu_bwd_kernel, i32, f32, f32, half, torch.int16, f32, f32, INTERMEDIATE,
                   tokens, INTERMEDIATE // GROUPS, TI=TILE_I, TM=TILE_M, QGROUP=group, INT8_IN=True)
    compile_kernel(quant_t_int8_il_kernel, i32, f32, f32, i8, INTERMEDIATE, tokens, GROUPS=GROUPS,
                   TR=TQ_R, TM=TQ_M, BF16=half == torch.bfloat16)
