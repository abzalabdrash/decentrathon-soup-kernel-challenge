"""NF4 weight dequantization in one Triton kernel, and bitsandbytes' 4-bit matmul running on it.

Every 4-bit linear layer of the base (7 per block, 28 blocks) rebuilds its fp16 weight on every forward
and every backward. With double quant (Soup's default), bitsandbytes 0.50.2 does it in three launches
plus two allocations: the 8-bit absmax codes are expanded to fp32 (`dequantize_blockwise`), the mean
offset is added (a separate torch add), then the 4-bit kernel writes the (N, K) weight. On a T4 the
training calls take this path: `gemm_4bit` sends M > 128 tokens to `_dequant_linear_fallback`
(dequantize, then `F.linear` in cuBLAS), and `MatMul4Bit.backward` always dequantizes.

Here one kernel reads the packed codes, the 8-bit absmax codes and the second-level scales, and writes
the fp16 weight in one pass. The layout it decodes (bitsandbytes 0.50.2, `quantize_4bit` with
`compress_statistics=True`):
- packed weight: uint8, (N * K + 1) // 2 bytes; byte i holds element 2i in the high nibble and
  element 2i + 1 in the low nibble of the row-major (N, K) weight;
- blocks of 64 elements (32 bytes): absmax of block b is `code2[q[b]] * absmax2[b // 256] + offset`,
  with q the uint8 `quant_state.absmax`, code2 = `state2.code` (256 fp32), absmax2 = `state2.absmax`
  (fp32, one per 256 blocks), offset = `quant_state.offset` (fp32 scalar);
- element value = `nf4[code] * absmax` in fp32, rounded to fp16, nf4 = `quant_state.code` (16 fp32).
The rounding steps are bitsandbytes' own (product rounded to fp32, plus offset rounded to fp32, product
rounded to fp32, then to fp16); the kernel is compiled without FMA contraction so that the multiply
and the add stay two roundings (with contraction 13 to 78 weights per matrix of this task come out one
ulp off). The result is bit-for-bit `bitsandbytes.functional.dequantize_4bit`.

The matmul stays in cuBLAS: on a T4 Triton 3.6 has no tensor-core path, so the kernel only replaces
the dequantization. The patched `MatMul4Bit` keeps only the packed weight and the quant state for the
backward, as the original does; the fp16 weight lives only inside each call. Where bitsandbytes would
run its own fused 4-bit GEMM (small M) or anything about the input is unusual, the original code runs.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

QBLOCK = 64  # elements per absmax (quant_state.blocksize)
ABLOCK = 256  # absmax blocks per second-level scale (state2.blocksize)
BLOCKS_PER_PROGRAM = 32  # 32 x 64 = 2048 weights per program


@triton.jit
def nf4_dequant_kernel(
    packed_ptr,  # uint8 (ceil(n / 2),): two 4-bit codes per byte, high nibble first
    qabsmax_ptr,  # uint8 (n_blocks,): 8-bit code of each block's absmax
    code2_ptr,  # fp32 (256,): state2.code, the map of those 8-bit codes
    absmax2_ptr,  # fp32 (ceil(n_blocks / ABLOCK),): state2.absmax
    offset_ptr,  # fp32 (): quant_state.offset
    nf4_ptr,  # fp32 (16,): quant_state.code, the NF4 levels
    out_ptr,  # fp16 (n,): the row-major (N, K) weight
    n,
    n_blocks,
    QBLOCK: tl.constexpr,
    ABLOCK: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    blk = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    live = blk < n_blocks
    # the absmax of each block, once per block: two roundings, as dequantize_blockwise + add
    q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
    scale = tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
    absmax = tl.load(code2_ptr + q) * scale
    absmax = absmax + tl.load(offset_ptr)

    half: tl.constexpr = QBLOCK // 2
    byte = blk[:, None] * half + tl.arange(0, half)[None, :]
    packed = tl.load(packed_ptr + byte, mask=byte * 2 < n, other=0)
    hi = tl.load(nf4_ptr + (packed >> 4).to(tl.int32)) * absmax[:, None]
    lo = tl.load(nf4_ptr + (packed & 15).to(tl.int32)) * absmax[:, None]
    vals = tl.interleave(hi, lo)  # (BLOCKS, QBLOCK): element 2j from the high nibble of byte j
    elem = blk[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    tl.store(out_ptr + elem, vals.to(out_ptr.dtype.element_ty), mask=elem < n)


def _fast_path(weight, quant_state, dtype):
    """True when the kernel handles this weight: NF4 with nested absmax in bitsandbytes' layout,
    blocksizes 64 / 256, fp16 output, everything on one CUDA device."""
    from bitsandbytes.functional import QuantState

    if not isinstance(quant_state, QuantState) or not torch.is_tensor(weight):
        return False
    if (quant_state.quant_type != "nf4" or not quant_state.nested or dtype != torch.float16
            or quant_state.blocksize != QBLOCK or quant_state.state2 is None
            or quant_state.state2.blocksize != ABLOCK):
        return False
    shape = quant_state.shape
    if shape is None or len(shape) != 2:
        return False
    n = int(shape[0]) * int(shape[1])
    state2 = quant_state.state2
    tensors = (weight, quant_state.absmax, quant_state.code, quant_state.offset, state2.absmax, state2.code)
    if not all(torch.is_tensor(t) and t.is_cuda and t.device == weight.device for t in tensors):
        return False
    if weight.device.index != torch.cuda.current_device():  # the kernel launches on the current device
        return False
    n_blocks = -(-n // QBLOCK)
    return (weight.dtype == torch.uint8 and weight.is_contiguous() and weight.numel() == (n + 1) // 2
            and 0 < n < 2 ** 31
            and quant_state.absmax.dtype == torch.uint8 and quant_state.absmax.numel() == n_blocks
            and quant_state.code.dtype == torch.float32 and quant_state.code.numel() == 16
            and quant_state.offset.dtype == torch.float32 and quant_state.offset.numel() == 1
            and state2.absmax.dtype == torch.float32 and state2.absmax.numel() == -(-n_blocks // ABLOCK)
            and state2.code.dtype == torch.float32 and state2.code.numel() == 256)


def _launch(weight, quant_state, dtype):
    """Runs the kernel; the caller has checked `_fast_path`."""
    n_rows, n_cols = (int(s) for s in quant_state.shape)
    n_blocks = quant_state.absmax.numel()
    out = torch.empty((n_rows, n_cols), dtype=dtype, device=weight.device)
    state2 = quant_state.state2
    nf4_dequant_kernel[(triton.cdiv(n_blocks, BLOCKS_PER_PROGRAM),)](
        weight, quant_state.absmax, state2.code, state2.absmax, quant_state.offset,
        quant_state.code, out, n_rows * n_cols, n_blocks,
        QBLOCK=QBLOCK, ABLOCK=ABLOCK, BLOCKS=BLOCKS_PER_PROGRAM,
        num_warps=4, enable_fp_fusion=False,
    )
    return out


def dequantize_nf4(weight, quant_state, dtype=None):
    """The (N, K) weight of a bitsandbytes 4-bit tensor, like `bitsandbytes.functional.dequantize_4bit`
    (same values, same dtype `quant_state.dtype` unless `dtype` is given). NF4 with double quant to
    fp16 runs the Triton kernel; anything else is bitsandbytes' own dequantization."""
    dtype = getattr(quant_state, "dtype", None) if dtype is None else dtype
    # a Params4bit answers every attribute through __torch_function__: check and launch on the plain
    # tensor (about 20 us less per call)
    plain = weight.data if torch.is_tensor(weight) and type(weight) is not torch.Tensor else weight
    if not _fast_path(plain, quant_state, dtype):
        import bitsandbytes.functional as bnb_functional

        out = bnb_functional.dequantize_4bit(weight, quant_state)
        return out if dtype is None or out.dtype == dtype else out.to(dtype)
    out = _launch(plain, quant_state, dtype)
    # bitsandbytes' shim for a packed weight passed in the transposed [1, n / 2] form
    if weight.dim() == 2 and weight.shape[0] == 1 and weight.numel() > 1:
        return out.t()
    return out


def _bnb_fused_gemm(A, quant_state):
    """Whether bitsandbytes' `gemm_4bit` would run its own fused 4-bit GEMM for this call (small M)
    instead of dequantize + F.linear. There the original stays; if the dispatch cannot be read, too."""
    try:
        from bitsandbytes.backends.cuda import ops
    except Exception:  # noqa: BLE001 - another bitsandbytes layout
        return True
    k = A.shape[-1]
    m = A.numel() // k
    n = int(quant_state.shape[0])
    custom_max_m = getattr(ops, "_gemm_4bit_custom_max_m", None)
    use_custom = getattr(ops, "_gemm_4bit_use_custom_fn", None)
    if not isinstance(custom_max_m, int) or not callable(use_custom):
        return True
    if m > custom_max_m or k % quant_state.blocksize != 0:
        return False
    try:
        return bool(use_custom(A.device.index, A.dtype, m, n, k))
    except Exception:  # noqa: BLE001
        return True


def _forward_takes_over(A, B, out, bias, quant_state):
    if not torch.is_tensor(A) or not A.is_cuda or A.dtype != torch.float16 or A.dim() < 1 \
            or A.numel() == 0 or not torch.is_tensor(B):
        return False
    if out is not None and not torch.is_tensor(out):
        return False
    if bias is not None and (not torch.is_tensor(bias) or bias.dtype != A.dtype or bias.dim() != 1
                             or bias.device != A.device):
        return False
    try:
        packed = B.view(-1, 1)
    except RuntimeError:
        return False
    if not _fast_path(packed, quant_state, A.dtype) or packed.device != A.device:
        return False
    if A.shape[-1] != int(quant_state.shape[1]):
        return False
    return not _bnb_fused_gemm(A, quant_state)


def patch_bnb_matmul4bit():
    """bitsandbytes' `MatMul4Bit` (what `bnb.matmul_4bit` applies when the input needs a gradient,
    i.e. every 4-bit linear layer in training) on the Triton dequantization: forward
    `F.linear(A, W, bias)`, backward `grad_A = grad_output @ W`, W rebuilt by `dequantize_nf4` each
    time from the packed weight. Same ctx fields as the original, so either backward can follow
    either forward."""
    from bitsandbytes.autograd import _functions as bnb_autograd

    cls = bnb_autograd.MatMul4Bit
    if getattr(cls, "_soup_nf4", False):
        return
    original_forward = cls.forward
    original_backward = cls.backward

    def forward(ctx, A, B, out=None, bias=None, quant_state=None):
        if not _forward_takes_over(A, B, out, bias, quant_state):
            return original_forward(ctx, A, B, out, bias, quant_state)
        ctx.is_empty = False
        B = B.view(-1, 1)
        weight = _launch(B, quant_state, A.dtype)
        output = F.linear(A, weight, bias)
        del weight
        if out is not None:
            out.copy_(output)
            output = out
        ctx.state = quant_state
        ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype
        ctx.tensors = (None, B) if any(ctx.needs_input_grad[:2]) else (None, None)
        return output

    def backward(ctx, grad_output):
        tensors = getattr(ctx, "tensors", None)
        state = getattr(ctx, "state", None)
        if (getattr(ctx, "is_empty", True) or not isinstance(tensors, tuple) or len(tensors) != 2
                or not torch.is_tensor(grad_output) or not grad_output.is_cuda
                or not _fast_path(tensors[1], state, getattr(state, "dtype", None))):
            return original_backward(ctx, grad_output)
        req_grad_a, _, _, req_grad_bias, _ = ctx.needs_input_grad
        grad_a = grad_bias = None
        if req_grad_bias:
            grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)
        if req_grad_a:
            grad_a = torch.matmul(grad_output, _launch(tensors[1], state, state.dtype).to(grad_output.dtype))
        return grad_a, None, None, grad_bias, None

    cls.forward = staticmethod(forward)
    cls.backward = staticmethod(backward)
    cls._soup_original_forward = original_forward
    cls._soup_original_backward = original_backward
    cls._soup_nf4 = True
