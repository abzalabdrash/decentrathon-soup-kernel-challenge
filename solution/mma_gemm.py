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
