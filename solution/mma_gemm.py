"""Tensor-core GEMMs from our own @triton.jit kernels on the T4 (sm_75), with the NF4 decode inside
the tile (B3).

Why Triton's tl.dot runs on CUDA cores on a T4, and what this module changes
---------------------------------------------------------------------------
Triton 3.6 still carries the Turing lowering of the tensor-core dot: libtriton.so has
`callMmaTuringFp16` / `callMmaTuringInt8` (mma.sync.aligned.m16n8k8 f16 and m8n8k16 s8, chosen when
the MMA layout is `nvidia_mma` version 2.1, `NvidiaMmaEncodingAttr::isTuring`). Only the layout pass
refuses to build MMA layouts below compute capability 80 (AccelerateMatmul: "Dot op using MMA for
compute capability 75 has been deprecated. It falls back to the FMA path."), so every tl.dot on a T4
becomes FMA code, 5-10x slower than cuBLAS.

`install()` registers a compilation stage hook (`triton.knobs.runtime.add_stages_inspection_hook`,
Triton's public knob for this, called from `CUDABackend.add_stages`) that changes the TTGIR stage
of our own kernels only (entry function name starting with `tc_`): the TTIR goes through the
backend's own `make_ttgir` with capability 80, so AccelerateMatmul assigns the mma v2 layout; then
the module is marked back as a cuda:75 module and the layout as Turing (versionMinor 1), and the
rest of the pipeline (LLVM lowering, PTX, ptxas --gpu-name=sm_75) runs unchanged for sm_75. The
kernels must be launched with num_stages=1: the Ampere software pipeliner would emit cp.async,
which sm_75 does not have (ptxas would refuse it). Every other kernel of the run compiles exactly as
before. The result is the same tensor-core instruction the organizers allowed as inline PTX inside
an own @triton.jit kernel (answer 18), produced by Triton's own code generator.

The kernels here are ordinary Triton GEMMs (the tile loop and the grouped launch order follow the
Triton tutorial "Matrix Multiplication", triton-lang.org, tutorials/03-matrix-multiplication):
- `tc_matmul_kernel`: C = A @ B in fp16 with fp32 accumulation, any strides of B (x @ W.T for a
  (N, K) weight, or g @ W for the input gradient);
- `tc_nf4_matmul_kernel`: the same with B decoded from bitsandbytes' NF4 inside the tile (packed
  codes, 8-bit absmax codes, second-level scales and offset; the decoded fp16 values are bit-for-bit
  those of `nf4_dequant_kernel`, the rounding steps are the same); a forward (x @ W.T) and an
  input-gradient (g @ W) layout.
"""

import os
import re
import tempfile

import torch
import triton
import triton.language as tl

QBLOCK = 64  # NF4 elements per absmax (bitsandbytes blocksize)
ABLOCK = 256  # absmax blocks per second-level scale
PREFIX = "tc_"

_STATE = {"installed": False}


def _rewrite_ttgir(text):
    """The TTGIR of a kernel built with capability 80, as a cuda:75 module with Turing MMA layouts."""
    text = text.replace('"cuda:80"', '"cuda:75"')
    return re.sub(r"versionMajor = 2, versionMinor = 0", "versionMajor = 2, versionMinor = 1", text)


def install():
    """Registers the stage hook (once per process; an earlier hook keeps running first)."""
    if _STATE["installed"]:
        return
    from triton import knobs
    from triton.compiler.compiler import parse

    prev = knobs.runtime.add_stages_inspection_hook

    def hook(backend, stages, options, language, capability):
        if prev is not None:
            prev(backend, stages, options, language, capability)
        if capability != 75 or "ttgir" not in stages:
            return
        orig = stages["ttgir"]

        def ttgir(src, metadata):
            try:
                name = src.get_entry_func_name()
            except Exception:
                name = ""
            if not name.startswith(PREFIX):
                return orig(src, metadata)
            if options.num_stages != 1:
                raise ValueError(f"{name}: launch with num_stages=1 (no cp.async on sm_75)")
            mod = backend.make_ttgir(src, metadata, options, 80)
            fd, path = tempfile.mkstemp(suffix=".ttgir")
            try:
                with os.fdopen(fd, "w") as f:
                    f.write(_rewrite_ttgir(str(mod)))
                return parse(path, "ttgir", mod.context)
            finally:
                os.remove(path)

        stages["ttgir"] = ttgir

    knobs.runtime.add_stages_inspection_hook = hook
    _STATE["installed"] = True


@triton.jit
def tc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     GROUP_M: tl.constexpr):
    """C (M, N) fp16 = A (M, K) @ B (K, N), fp32 accumulation; K a multiple of BLOCK_K."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (rm % M)[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + (rn % N)[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def tc_matmul_i8_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                        stride_am, stride_bn, stride_cm,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        GROUP_M: tl.constexpr):
    """C (M, N) int32 = A (M, K) int8 @ B (K, N) int8, B column-major (a row-major (N, K) weight)."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (rm % M)[:, None] * stride_am + rk[None, :]
    b_ptrs = b_ptr + rk[:, None] + (rn % N)[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _nf4_absmax(qabsmax_ptr, code2_ptr, absmax2_ptr, offset, blk, ABLOCK: tl.constexpr):
    """bitsandbytes' double-quant absmax of blocks `blk`: two fp32 roundings, as in nf4_dequant_kernel."""
    q = tl.load(qabsmax_ptr + blk).to(tl.int32)
    return tl.load(code2_ptr + q) * tl.load(absmax2_ptr + blk // ABLOCK) + offset


@triton.jit
def tc_nf4_matmul_kernel(a_ptr, w_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr,
                         c_ptr, M, N, K, stride_am, stride_cm,
                         W_IS_NK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, QBLOCK: tl.constexpr,
                         ABLOCK: tl.constexpr):
    """C (M, N) fp16 = A (M, K) @ B, B decoded from an NF4 weight inside the tile.

    W_IS_NK: the weight is (N, K) row-major and B = W.T (forward, x @ W.T); otherwise it is (K, N)
    row-major and B = W (input gradient, g @ W). Packed byte i holds element 2i in the high nibble
    and 2i + 1 in the low one; absmax blocks of QBLOCK run along the weight's rows.
    """
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    size_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = a_ptr + (rm % M)[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :]
    offset = tl.load(offset_ptr)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kt in range(0, K // BLOCK_K):
        k0 = kt * BLOCK_K
        if W_IS_NK:
            # tile (BLOCK_N, BLOCK_K) of the (N, K) weight, rows rn, columns k0..k0 + BLOCK_K
            byte = tl.load(w_ptr + rn[:, None] * (K // 2) + (k0 // 2 + tl.arange(0, BLOCK_K // 2))[None, :])
            codes = tl.reshape(tl.join((byte >> 4) & 15, byte & 15), (BLOCK_N, BLOCK_K))
            blk = rn[:, None] * (K // QBLOCK) + (k0 // QBLOCK + tl.arange(0, BLOCK_K // QBLOCK))[None, :]
            amax = _nf4_absmax(qabsmax_ptr, code2_ptr, absmax2_ptr, offset, blk, ABLOCK)
            amax = tl.reshape(tl.broadcast_to(amax[:, :, None], (BLOCK_N, BLOCK_K // QBLOCK, QBLOCK)),
                              (BLOCK_N, BLOCK_K))
            w = (tl.load(nf4_ptr + codes.to(tl.int32)) * amax).to(tl.float16)
            b = tl.trans(w)
        else:
            # tile (BLOCK_K, BLOCK_N) of the (K, N) weight, rows k0.., columns rn
            rk = k0 + tl.arange(0, BLOCK_K)
            byte = tl.load(w_ptr + rk[:, None] * (N // 2) + (pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2))[None, :])
            codes = tl.reshape(tl.join((byte >> 4) & 15, byte & 15), (BLOCK_K, BLOCK_N))
            blk = rk[:, None] * (N // QBLOCK) + (pid_n * (BLOCK_N // QBLOCK) + tl.arange(0, BLOCK_N // QBLOCK))[None, :]
            amax = _nf4_absmax(qabsmax_ptr, code2_ptr, absmax2_ptr, offset, blk, ABLOCK)
            amax = tl.reshape(tl.broadcast_to(amax[:, :, None], (BLOCK_K, BLOCK_N // QBLOCK, QBLOCK)),
                              (BLOCK_K, BLOCK_N))
            b = (tl.load(nf4_ptr + codes.to(tl.int32)) * amax).to(tl.float16)
        acc = tl.dot(tl.load(a_ptrs), b, acc)
        a_ptrs += BLOCK_K
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :]
    tl.store(c_ptrs, acc.to(tl.float16), mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def tc_int8_swiglu_bwd_kernel(w_ptr, g_ptr, w_scale_ptr, g_scale_ptr, gate_up_ptr, pmax_ptr,
                              inter, tokens, D,
                              BLOCK_I: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
                              GROUP_I: tl.constexpr):
    """int8_bwd.py's steps 3 and 4 in one kernel: the int32 product dL/dhidden^T (I, tokens) =
    W_down^T_q (I, D) @ g_q^T (g_q (tokens, D) row-major) on the tensor cores, kept in registers,
    then int8_swiglu_bwd_kernel's epilogue on the tile: the same fp32 operations in the same order
    (scales, the SwiGLU derivative with the saved (gate|up), half results written over (gate|up)), and
    the tile's per-token absmax to pmax[feature tile, token]. The int32 product is exact, so every
    output is bit-for-bit the two-kernel path's; the (I, tokens) int32 buffer is never allocated."""
    pid = tl.program_id(0)
    num_i = inter // BLOCK_I
    num_t = tl.cdiv(tokens, BLOCK_T)
    group = GROUP_I * num_t
    first_i = (pid // group) * GROUP_I
    size_i = min(num_i - first_i, GROUP_I)
    pid_i = first_i + (pid % group) % size_i
    pid_t = (pid % group) // size_i
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = w_ptr + ri[:, None] * D + rk[None, :]
    b_ptrs = g_ptr + (rt % tokens)[None, :] * D + rk[:, None]
    acc = tl.zeros((BLOCK_I, BLOCK_T), dtype=tl.int32)
    for _ in range(0, D // BLOCK_K):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    live = rt < tokens
    s_tok = tl.load(g_scale_ptr + rt, mask=live, other=0.0)
    dhid = acc.to(tl.float32)
    dhid = dhid * tl.load(w_scale_ptr + ri)[:, None] * s_tok[None, :]
    at = ri[:, None].to(tl.int64) * tokens + rt[None, :]
    up_shift = inter.to(tl.int64) * tokens
    g = tl.load(gate_up_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + up_shift + at, mask=live[None, :], other=0.0).to(tl.float32)
    sig = tl.sigmoid(g)
    silu = g * sig
    d_u = (dhid * silu).to(gate_up_ptr.dtype.element_ty)
    d_g = (dhid * u * (sig + silu * (1.0 - sig))).to(gate_up_ptr.dtype.element_ty)
    tl.store(gate_up_ptr + at, d_g, mask=live[None, :])
    tl.store(gate_up_ptr + up_shift + at, d_u, mask=live[None, :])
    peak = tl.maximum(tl.abs(d_g.to(tl.float32)), tl.abs(d_u.to(tl.float32)))
    tl.store(pmax_ptr + pid_i * tokens + rt, tl.max(peak, axis=0), mask=live)


# (BLOCK_I, BLOCK_T, BLOCK_K, num_warps, GROUP_I) of tc_int8_swiglu_bwd_kernel
SWIGLU_BWD_CFG = tuple(int(v) for v in re.split(r"[,x]", os.environ.get("SOUP_SOLUTION_MMA_BWD_CFG", "64x128x64x4x8")))


def int8_swiglu_bwd(w_down_t, g_q, w_scale, g_scale, gate_up, cfg=SWIGLU_BWD_CFG):
    """Steps 3-4 of int8_bwd.mlp_backward: w_down_t (I, D) int8, g_q (tokens, D) int8, w_scale (I,)
    and g_scale (tokens,) fp32, gate_up (2I, tokens) half (overwritten with its gradient). Returns
    pmax (I / BLOCK_I, tokens) fp32, the per-tile token maxima."""
    install()
    bi, bt, bk, warps, group = cfg
    inter, d = w_down_t.shape
    tokens = g_q.shape[0]
    pmax = torch.empty((inter // bi, tokens), dtype=torch.float32, device=g_q.device)
    tc_int8_swiglu_bwd_kernel[(inter // bi * triton.cdiv(tokens, bt),)](
        w_down_t, g_q, w_scale, g_scale, gate_up, pmax, inter, tokens, d,
        BLOCK_I=bi, BLOCK_T=bt, BLOCK_K=bk, GROUP_I=group, num_warps=warps, num_stages=1)
    return pmax


def int8_swiglu_bwd_usable(inter, d, cfg=SWIGLU_BWD_CFG):
    return inter % cfg[0] == 0 and d % cfg[2] == 0


@triton.jit
def _silu_mul_tc(g, u):
    # silu(gate) * up in fp32 written as int8_fwd._silu_mul (and swiglu.py's forward) writes it
    return g * tl.sigmoid(g) * u


@triton.jit
def tc_int8_gate_up_kernel(w_ptr, x_ptr, w_scale_ptr, x_scale_ptr, gate_up_ptr, hidden_ptr,
                           xk_ptr, wk_ptr, inter, tokens, chunk, D,
                           BLOCK_F: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
                           GROUP_F: tl.constexpr, K: tl.constexpr):
    """int8_fwd.py's gate|up products and int8_gate_up_kernel (MODE "gu") in one kernel: for BLOCK_F
    features and BLOCK_T tokens, the int32 products of the gate rows and of the up rows of the int8
    weight w (2I, D), rows interleaved by chunk as int8_fwd._decode_gate_up writes them (gate rows of
    chunk c, then its up rows), with x_q (tokens, D) on the tensor cores, kept in registers; then the
    epilogue of int8_gate_up_kernel on the tile: the same fp32 operations in the same order (scales,
    the top-k channels' fp32 share wk @ xk^T when K > 0), (gate|up) (2I, tokens) and silu(gate) * up
    (I, tokens) in half. The int32 products are exact, so the outputs are bit-for-bit those of the
    chunked _int_mm path, without its int32 buffer. BLOCK_F divides chunk."""
    pid = tl.program_id(0)
    num_f = inter // BLOCK_F
    num_t = tl.cdiv(tokens, BLOCK_T)
    group = GROUP_F * num_t
    first_f = (pid // group) * GROUP_F
    size_f = min(num_f - first_f, GROUP_F)
    pid_f = first_f + (pid % group) % size_f
    pid_t = (pid % group) // size_f
    feat = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    # rows of w (and of w_scale, wk) of these features: gate, then up, within their chunk
    row_g = (feat // chunk) * (2 * chunk) + feat % chunk
    row_u = row_g + chunk
    rt = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    rk = tl.arange(0, BLOCK_K)
    g_ptrs = w_ptr + row_g[:, None].to(tl.int64) * D + rk[None, :]
    u_ptrs = w_ptr + row_u[:, None].to(tl.int64) * D + rk[None, :]
    x_ptrs = x_ptr + (rt % tokens)[None, :].to(tl.int64) * D + rk[:, None]
    acc_g = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.int32)
    acc_u = tl.zeros((BLOCK_F, BLOCK_T), dtype=tl.int32)
    for _ in range(0, D // BLOCK_K):
        xt = tl.load(x_ptrs)
        acc_g = tl.dot(tl.load(g_ptrs), xt, acc_g, out_dtype=tl.int32)
        acc_u = tl.dot(tl.load(u_ptrs), xt, acc_u, out_dtype=tl.int32)
        g_ptrs += BLOCK_K
        u_ptrs += BLOCK_K
        x_ptrs += BLOCK_K
    live = rt < tokens
    s_tok = tl.load(x_scale_ptr + rt, mask=live, other=0.0)
    g = acc_g.to(tl.float32)
    u = acc_u.to(tl.float32)
    g = g * tl.load(w_scale_ptr + row_g)[:, None] * s_tok[None, :]
    u = u * tl.load(w_scale_ptr + row_u)[:, None] * s_tok[None, :]
    if K:
        kk = tl.arange(0, K)
        xkt = tl.trans(tl.load(xk_ptr + rt[:, None].to(tl.int64) * K + kk[None, :],
                               mask=live[:, None], other=0.0))
        wg = tl.load(wk_ptr + row_g[:, None].to(tl.int64) * K + kk[None, :])
        wu = tl.load(wk_ptr + row_u[:, None].to(tl.int64) * K + kk[None, :])
        g = g + tl.dot(wg, xkt, input_precision="ieee")
        u = u + tl.dot(wu, xkt, input_precision="ieee")
    g = g.to(gate_up_ptr.dtype.element_ty)
    u = u.to(gate_up_ptr.dtype.element_ty)
    dest = feat[:, None].to(tl.int64) * tokens + rt[None, :]
    tl.store(gate_up_ptr + dest, g, mask=live[None, :])
    tl.store(gate_up_ptr + inter.to(tl.int64) * tokens + dest, u, mask=live[None, :])
    h = _silu_mul_tc(g.to(tl.float32), u.to(tl.float32))
    tl.store(hidden_ptr + dest, h.to(gate_up_ptr.dtype.element_ty), mask=live[None, :])


# (BLOCK_F, BLOCK_T, BLOCK_K, num_warps, GROUP_F) of tc_int8_gate_up_kernel
GATE_UP_CFG = tuple(int(v) for v in re.split(r"[,x]", os.environ.get("SOUP_SOLUTION_MMA_GU_CFG", "64x128x64x4x8")))


def int8_gate_up(w, s_w, x_q, s_x, inter, chunk, half, w_k=None, x_k=None, gate_up=None,
                 hidden=None, cfg=GATE_UP_CFG):
    """int8_fwd.gate_up_forward's products and epilogue (MODE "gu") in one launch: w (2I, D) int8
    interleaved by chunk, s_w (2I,), x_q (tokens, D) int8, s_x (tokens,), optional top-k w_k (2I, K)
    and x_k (tokens, K) fp32. Returns ((gate|up) (2I, tokens), silu(gate) * up (I, tokens)) in half."""
    install()
    bf, bt, bk, warps, group = cfg
    tokens, d = x_q.shape
    device = x_q.device
    if gate_up is None:
        gate_up = torch.empty((2 * inter, tokens), dtype=half, device=device)
    if hidden is None:
        hidden = torch.empty((inter, tokens), dtype=half, device=device)
    k = 0 if w_k is None else w_k.shape[1]
    dummy = s_x[:1]
    tc_int8_gate_up_kernel[(inter // bf * triton.cdiv(tokens, bt),)](
        w, x_q, s_w, s_x, gate_up, hidden, x_k if k else dummy, w_k if k else dummy, inter, tokens,
        chunk, d, BLOCK_F=bf, BLOCK_T=bt, BLOCK_K=bk, GROUP_F=group, K=k, num_warps=warps,
        num_stages=1)
    return gate_up, hidden


def int8_gate_up_usable(inter, d, chunk, cfg=GATE_UP_CFG):
    return chunk % cfg[0] == 0 and inter % chunk == 0 and d % cfg[2] == 0


def _grid(M, N, bm, bn):
    return (triton.cdiv(M, bm) * triton.cdiv(N, bn),)


def matmul(a, b, cfg, out=None):
    """a (M, K) fp16 row-major @ b (K, N) fp16 with any strides -> (M, N) fp16."""
    install()
    bm, bn, bk, warps, group = cfg
    M, K = a.shape
    N = b.shape[1]
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    return tc_matmul_kernel[_grid(M, N, bm, bn)](
        a, b, out, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group, num_warps=warps, num_stages=1), out


def matmul_i8(a, w, cfg, out=None):
    """a (M, K) int8 row-major @ w.T, w (N, K) int8 row-major -> (M, N) int32."""
    install()
    bm, bn, bk, warps, group = cfg
    M, K = a.shape
    N = w.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.int32, device=a.device)
    return tc_matmul_i8_kernel[_grid(M, N, bm, bn)](
        a, w, out, M, N, K, a.stride(0), w.stride(0), out.stride(0),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group, num_warps=warps, num_stages=1), out


def matmul_nf4(a, packed, qs, n_out, w_is_nk, cfg, out=None):
    """a (M, K) fp16 @ (W.T if w_is_nk else W) with W the NF4 weight of quant state `qs`."""
    install()
    bm, bn, bk, warps, group = cfg
    M, K = a.shape
    N = n_out
    if out is None:
        out = torch.empty((M, N), dtype=torch.float16, device=a.device)
    s2 = qs.state2
    return tc_nf4_matmul_kernel[_grid(M, N, bm, bn)](
        a, packed, qs.absmax, s2.code, s2.absmax, qs.offset, qs.code, out, M, N, K,
        a.stride(0), out.stride(0), W_IS_NK=w_is_nk, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
        GROUP_M=group, QBLOCK=QBLOCK, ABLOCK=ABLOCK, num_warps=warps, num_stages=1,
        enable_fp_fusion=False), out
