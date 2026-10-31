"""The torch glue of int8_bwd.py's MLP backward in own kernels (task H2, SOUP_SOLUTION_GLUE; on by default).

Per layer and step the int8 MLP backward made about 26 small torch launches around its GEMMs (T4
profile soup-m10-prof, per step: 28 floor, 85 clamp, 85 fp32 binary, 29 fp32 add, 56 half reductions,
casts): the per-token quantization of the incoming gradient (`int8_bwd._quant_rows`: abs, amax, cast,
compare, clamp, division, where, mul, add, floor, clamp, cast, division), the gate|up column scales
recombined every step from cached per-weight scales (maximum, mul, compare, clamp, division, where), and
the dequantization of the input gradient (`acc_x.float().mul_(s_tok).mul_(s_col)`). Here:

- `quant_rows_kernel` (own): one program per row; the same values as `_quant_rows` bit for bit: the absmax
  of the half row, inv = (1 / absmax) * 127 (torch's `127.0 / t`) or 0 for a zero row, q = floor(x * inv
  + 0.5) clamped to [-127, 127] (no fused multiply-add: the torch ops round the product first), scale =
  absmax * (1 / 127) (torch's division by a scalar). A row holding a nan gets a nan scale (torch's amax
  propagates it; Triton's max would not), an inf an inf scale, so the backward's products are not finite
  either and GradScaler skips the step as before;
- `dequant_rows_cols_kernel` (own): out = (float(acc) * (amax / 127)) * s_col, in the torch ops' order;
- `gate_up_scales`: max(scale_gate, scale_up) and its inverse kept on the layer next to the per-weight
  scales they come from (frozen weights: the same values every step).
"""

import os

import torch
import triton
import triton.language as tl

ON = os.environ.get("SOUP_SOLUTION_GLUE", "1").strip().lower() not in ("0", "false", "no", "")
BLOCK_COLS = 2048  # one row of the hidden size 1536 per program
COLS_TILE = 512  # columns per program of the dequantization


@triton.jit
def quant_rows_kernel(x_ptr, stride_x, q_ptr, stride_q, s_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < n_cols
    x = tl.load(x_ptr + row.to(tl.int64) * stride_x + offs, mask=m, other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    nan = tl.sum((x != x).to(tl.int32), axis=0) > 0
    amax = tl.where(nan, float("nan"), amax)
    # torch's order: 127.0 / t is t.reciprocal() * 127.0 (Tensor.__rtruediv__), t / 127.0 is t times
    # the fp32 reciprocal of 127 (the CUDA division by a scalar)
    one = tl.full([], 1.0, tl.float32)
    safe = tl.maximum(amax, 1e-30)
    inv = tl.where(amax > 0.0, tl.math.div_rn(one, safe) * 127.0, 0.0)
    y = tl.floor(x * inv + 0.5)
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    tl.store(q_ptr + row.to(tl.int64) * stride_q + offs, y.to(tl.int8), mask=m)
    tl.store(s_ptr + row, amax * tl.math.div_rn(one, tl.full([], 127.0, tl.float32)))


@triton.jit
def dequant_rows_cols_kernel(acc_ptr, stride_acc, amax_ptr, col_ptr, out_ptr, stride_out, n_cols,
                             TILE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * TILE + tl.arange(0, TILE)
    m = cols < n_cols
    a = tl.load(acc_ptr + row.to(tl.int64) * stride_acc + cols, mask=m, other=0).to(tl.float32)
    s_row = tl.load(amax_ptr + row) * tl.math.div_rn(tl.full([], 1.0, tl.float32), tl.full([], 127.0, tl.float32))
    y = a * s_row
    y = y * tl.load(col_ptr + cols, mask=m, other=0.0)
    tl.store(out_ptr + row.to(tl.int64) * stride_out + cols, y, mask=m)


def quant_rows(x):
    """int8_bwd._quant_rows of a 2-D half tensor (contiguous rows): (int8 rows, fp32 scales)."""
    n, d = x.shape
    q = torch.empty((n, d), dtype=torch.int8, device=x.device)
    s = torch.empty(n, dtype=torch.float32, device=x.device)
    if n:
        quant_rows_kernel[(n,)](x, x.stride(0), q, q.stride(0), s, d, BLOCK=triton.next_power_of_2(d),
                                num_warps=4, enable_fp_fusion=False)
    return q, s


def dequant_rows_cols(acc, amax, col):
    """acc.float().mul_((amax / 127)[:, None]).mul_(col[None, :]) for an int32 (n, d) product."""
    n, d = acc.shape
    out = torch.empty((n, d), dtype=torch.float32, device=acc.device)
    if n:
        dequant_rows_cols_kernel[(n, triton.cdiv(d, COLS_TILE))](
            acc, acc.stride(0), amax, col, out, out.stride(0), d, TILE=COLS_TILE, num_warps=4,
            enable_fp_fusion=False)
    return out


def gate_up_scales(gate, scale_g, scale_u, inverse):
    """(max(scale_g, scale_u), inverse(max * 127)) of a layer, kept on its gate module while the two
    per-weight scale tensors are the same objects (int8_bwd._column_scales keeps them per weight)."""
    cached = gate.__dict__.get("_soup_int8_gu")
    if cached is not None and cached[0] is scale_g and cached[1] is scale_u:
        return cached[2], cached[3]
    scale_gu = torch.maximum(scale_g, scale_u)
    inv_gu = inverse(scale_gu * 127.0)
    gate.__dict__["_soup_int8_gu"] = (scale_g, scale_u, scale_gu, inv_gu)
    return scale_gu, inv_gu


def usable(x):
    return ON and x.dim() == 2 and x.is_cuda and x.stride(1) == 1 and x.dtype in (torch.float16, torch.bfloat16)


def warm(half, hidden, f32=torch.float32):
    """The two kernels in the classes int8_bwd's backward launches (its compile thread calls this)."""
    from .warmup import compile_kernel

    if not ON:
        return
    compile_kernel(quant_rows_kernel, half, hidden, torch.int8, hidden, f32, hidden,
                   BLOCK=triton.next_power_of_2(hidden), num_warps=4, enable_fp_fusion=False)
    compile_kernel(dequant_rows_cols_kernel, torch.int32, hidden, f32, f32, f32, hidden, hidden,
                   TILE=COLS_TILE, num_warps=4, enable_fp_fusion=False)
