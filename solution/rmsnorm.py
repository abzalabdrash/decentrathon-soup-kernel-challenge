"""Qwen2RMSNorm in one Triton pass each way, and the decoder layer's `residual + attn_out` fused into
the post-attention norm.

Qwen2RMSNorm.forward in eager PyTorch is five kernels per call: pow, mean, add + rsqrt, the row
scale and the weight multiply, each reading and writing the whole (rows, 1536) fp32 tensor. With 57
norms per forward (two per layer and the final one) and their backward graphs that is a lot of
memory traffic for a few FLOPs. Here a norm is one kernel that reads the rows once and writes the
output once; it keeps one fp32 rstd per row for the backward (eager keeps rsqrt's (rows, 1) output
and the fp32 input, the same amount). The backward is one kernel as well: dx in fp32 by the
closed-form RMSNorm gradient, and dweight only when the weight trains (on this task the norm weights
are frozen, so the reduction over rows is skipped entirely).

In the decoder layer the residual add that comes right before `post_attention_layernorm` is fused
into the same kernel: it reads the residual (fp32) and the attention output (fp16 out of
bitsandbytes' o_proj under autocast), writes the new residual and the normed rows in one pass, and
its backward writes the gradient of both addends at once.

Inside the packed training step (unpad.varlen_model_forward) the layer-end `residual + mlp_out` is
fused the same way, into the next layer's input norm (and the final norm): the layers hand the two
addends on unsummed. And `norm_source` / `rebuild` let a consumer of a norm's output drop its own
half copy of it for the backward: the output is recomputed bit for bit from the input (or sum) and
rstd that the norm keeps anyway (fused_proj.py's q/k/v projection does this).

The post-attention norm can keep its two addends instead of their fp32 sum (`post_attention_addends`):
the residual is the layer's input, which the input norm keeps for its own backward anyway (no byte
more), and the attention output is half, so the layer keeps 2 bytes per element instead of 4; the
backward kernel (`rms_norm_rows_bwd_split`) adds them again exactly as the forward did (one fp32 add
of the same values: the same sum, bit for bit). With fused_proj.py's attention recompute the
attention output is not kept either: the backward gets it from the recompute (`addends.branch()`).
`SOUP_SOLUTION_SPLIT_SUM=1` turns the first on (fused_proj's SOUP_SOLUTION_RECOMPUTE_ATTN the second).

The numerics follow Qwen2RMSNorm step for step: fp32 math, the normed rows rounded back to the input
dtype before the fp32 weight multiply, the output in `promote_types(weight, input)`; in the backward,
dL/d(normed) is rounded to the input dtype where eager autograd rounds it. Only the order of the sum
of squares differs (rounding-level differences in fp32).
"""

import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

MAX_COLS = 8192  # a row is one register tile; longer rows take the original path
DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# rows per program and warps per program for the forward and the backward. bench_rmsnorm.py
# --sweep on rows of 1536 (RTX 3050 Ti): 1-8 rows and 2-16 warps all run at the memory bandwidth
# within noise; one row per program and 4 warps is never slower
FWD_ROWS, FWD_WARPS = 1, 4
BWD_ROWS, BWD_WARPS = 1, 4
DW_PROGRAMS_PER_SM = 4  # programs of the dweight path; each keeps a partial sum of its rows

# SOUP_SOLUTION_SPLIT_SUM=1: the post-attention norm keeps (residual, attention output) for its
# backward instead of their fp32 sum (see the module doc); read at import, like the other switches
_SPLIT = {"keep": os.environ.get("SOUP_SOLUTION_SPLIT_SUM", "0").strip().lower()
          not in ("0", "false", "no", "")}
KEEP_BRANCH = "keep-branch"  # AddRMSNormFunction's `addends`: save the branch itself


@triton.jit(do_not_specialize=["n_rows"])
def rms_norm_rows_fwd(
    x_ptr, stride_x,  # input rows (n_rows, n_cols), unit column stride
    add_ptr, stride_add,  # with ADD: rows added to x before the norm (the attention output)
    sum_ptr, stride_sum,  # with ADD: x + add stored in its dtype (the new residual)
    w_ptr,  # fp32 (n_cols,)
    y_ptr, stride_y,  # output rows
    rstd_ptr,  # fp32 (n_rows,): 1 / sqrt(mean(v * v) + eps) of every row, for the backward
    n_rows, n_cols, eps,
    ADD: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    row_ok = rows < n_rows
    col_ok = cols < n_cols
    tile_ok = row_ok[:, None] & col_ok[None, :]
    rows64 = rows.to(tl.int64)[:, None]
    v = tl.load(x_ptr + rows64 * stride_x + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
    if ADD:
        a = tl.load(add_ptr + rows64 * stride_add + cols[None, :], mask=tile_ok, other=0.0)
        # torch adds in fp32 and rounds the sum to the result dtype; the norm reads the rounded sum
        s = (v + a.to(tl.float32)).to(sum_ptr.dtype.element_ty)
        tl.store(sum_ptr + rows64 * stride_sum + cols[None, :], s, mask=tile_ok)
        v = s.to(tl.float32)
    mean_sq = tl.sum(v * v, axis=1) / n_cols
    rstd = tl.rsqrt(mean_sq + eps)
    tl.store(rstd_ptr + rows, rstd, mask=row_ok)
    normed = v * rstd[:, None]
    # Qwen2RMSNorm casts the normed rows back to the input dtype, then multiplies by the fp32 weight
    if ADD:
        normed = normed.to(sum_ptr.dtype.element_ty).to(tl.float32)
    else:
        normed = normed.to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
    out = normed * w[None, :]
    tl.store(y_ptr + rows64 * stride_y + cols[None, :], out.to(y_ptr.dtype.element_ty), mask=tile_ok)


@triton.jit(do_not_specialize=["n_rows"])
def rms_norm_rows_bwd(
    dy_ptr, stride_dy,  # gradient of the norm output
    x_ptr, stride_x,  # the norm input saved by the forward
    w_ptr,  # fp32 (n_cols,)
    rstd_ptr,  # fp32 (n_rows,) from the forward
    dsum_ptr, stride_dsum,  # with ADD_GRAD: gradient reaching the norm input by another path
    dx_ptr, stride_dx,  # gradient of the norm input, in its dtype (the residual's for the fused add)
    dx2_ptr, stride_dx2,  # with SECOND: the same gradient stored again in another dtype (the branch's)
    dw_ptr,  # with DW: fp32 (programs, n_cols), one partial sum of dy * normed per program
    n_rows, n_cols, tiles_per_program,
    ADD_GRAD: tl.constexpr,
    SECOND: tl.constexpr,
    DW: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    col_ok = cols < n_cols
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
    dw_acc = tl.zeros([BLOCK], dtype=tl.float32)
    for t in range(0, tiles_per_program):
        rows = (pid * tiles_per_program + t) * ROWS + tl.arange(0, ROWS)
        row_ok = rows < n_rows
        tile_ok = row_ok[:, None] & col_ok[None, :]
        rows64 = rows.to(tl.int64)[:, None]
        x = tl.load(x_ptr + rows64 * stride_x + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + rows64 * stride_dy + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_ok, other=0.0)
        normed = x * rstd[:, None]
        # eager autograd rounds dL/d(normed) to the input dtype at the backward of `.to(input_dtype)`
        g = (dy * w[None, :]).to(x_ptr.dtype.element_ty).to(tl.float32)
        # d/dx of x * rsqrt(mean(x^2) + eps): rstd * (g - normed * mean(g * normed))
        c = tl.sum(g * normed, axis=1) / n_cols
        dx = (g - normed * c[:, None]) * rstd[:, None]
        if ADD_GRAD:
            dx += tl.load(dsum_ptr + rows64 * stride_dsum + cols[None, :], mask=tile_ok,
                          other=0.0).to(tl.float32)
        tl.store(dx_ptr + rows64 * stride_dx + cols[None, :], dx.to(dx_ptr.dtype.element_ty),
                 mask=tile_ok)
        if SECOND:
            tl.store(dx2_ptr + rows64 * stride_dx2 + cols[None, :], dx.to(dx2_ptr.dtype.element_ty),
                     mask=tile_ok)
        if DW:
            # the weight multiplied the normed rows as rounded to the input dtype
            seen = normed.to(x_ptr.dtype.element_ty).to(tl.float32)
            dw_acc += tl.sum(dy * seen, axis=0)
    if DW:
        tl.store(dw_ptr + pid * n_cols + cols, dw_acc, mask=col_ok)


@triton.jit(do_not_specialize=["n_rows"])
def rms_norm_rows_rebuild(
    x_ptr, stride_x,  # the rows the forward normed: the norm input, or the fused add's sum
    rstd_ptr,  # fp32 (n_rows,): what the forward stored
    w_ptr,  # fp32 (n_cols,)
    y_ptr, stride_y,  # the forward's output rows again, rounded to y's dtype
    n_rows, n_cols,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    row_ok = rows < n_rows
    col_ok = cols < n_cols
    tile_ok = row_ok[:, None] & col_ok[None, :]
    rows64 = rows.to(tl.int64)[:, None]
    v = tl.load(x_ptr + rows64 * stride_x + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_ok, other=0.0)
    # rms_norm_rows_fwd after its reduction, the same operations in the same order: two products
    # with no sum between them, so nothing for the compiler to contract
    normed = (v * rstd[:, None]).to(x_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
    out = normed * w[None, :]
    tl.store(y_ptr + rows64 * stride_y + cols[None, :], out.to(y_ptr.dtype.element_ty), mask=tile_ok)


@triton.jit(do_not_specialize=["n_rows"])
def rms_norm_rows_bwd_split(
    dy_ptr, stride_dy,  # gradient of the norm output
    r_ptr, stride_r,  # the residual the forward added (fp32: the sum's dtype)
    b_ptr, stride_b,  # the branch the forward added to it (the attention output, half)
    w_ptr,  # fp32 (n_cols,)
    rstd_ptr,  # fp32 (n_rows,) from the forward
    dsum_ptr, stride_dsum,  # with ADD_GRAD: gradient reaching the sum by the residual path
    dx_ptr, stride_dx,  # gradient of the sum, in the residual's dtype
    dx2_ptr, stride_dx2,  # with SECOND: the same gradient stored again in the branch's dtype
    n_rows, n_cols,
    ADD_GRAD: tl.constexpr,
    SECOND: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # rms_norm_rows_bwd for a frozen weight (no dweight), reading the sum's two addends: the sum is
    # formed as rms_norm_rows_fwd formed it (one fp32 add, rounded to the residual's dtype, which
    # is the sum's), so it is the forward's sum bit for bit, and from there the same operations
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    row_ok = rows < n_rows
    col_ok = cols < n_cols
    tile_ok = row_ok[:, None] & col_ok[None, :]
    rows64 = rows.to(tl.int64)[:, None]
    r = tl.load(r_ptr + rows64 * stride_r + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + rows64 * stride_b + cols[None, :], mask=tile_ok, other=0.0)
    x = (r + b.to(tl.float32)).to(r_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)
    dy = tl.load(dy_ptr + rows64 * stride_dy + cols[None, :], mask=tile_ok, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + rows, mask=row_ok, other=0.0)
    normed = x * rstd[:, None]
    g = (dy * w[None, :]).to(r_ptr.dtype.element_ty).to(tl.float32)
    c = tl.sum(g * normed, axis=1) / n_cols
    dx = (g - normed * c[:, None]) * rstd[:, None]
    if ADD_GRAD:
        dx += tl.load(dsum_ptr + rows64 * stride_dsum + cols[None, :], mask=tile_ok,
                      other=0.0).to(tl.float32)
    tl.store(dx_ptr + rows64 * stride_dx + cols[None, :], dx.to(dx_ptr.dtype.element_ty),
             mask=tile_ok)
    if SECOND:
        tl.store(dx2_ptr + rows64 * stride_dx2 + cols[None, :], dx.to(dx2_ptr.dtype.element_ty),
                 mask=tile_ok)


def _rows(t):
    """(…, C) -> (rows, C) with a unit column stride, a view whenever possible."""
    t2 = t.reshape(-1, t.shape[-1])
    return t2 if t2.stride(-1) == 1 else t2.contiguous()


_SM_COUNT = {}


def _sm_count(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNT[index]


def _forward(x2, add2, sum2, weight, y2, rstd, eps):
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    has_add = add2 is not None
    add2 = add2 if has_add else x2
    sum2 = sum2 if has_add else x2
    rms_norm_rows_fwd[(triton.cdiv(n_rows, FWD_ROWS),)](
        x2, x2.stride(0), add2, add2.stride(0), sum2, sum2.stride(0), weight,
        y2, y2.stride(0), rstd, n_rows, n_cols, eps,
        ADD=has_add, ROWS=FWD_ROWS, BLOCK=block, num_warps=FWD_WARPS,
    )


def _backward(dy2, x2, weight, rstd, dsum2, dx_dtype, dx2_dtype, need_dw):
    """dx (in dx_dtype), the same values in dx2_dtype when it is given and differs (else None), and
    dweight when need_dw (else None)."""
    n_rows, n_cols = x2.shape
    block = triton.next_power_of_2(n_cols)
    dx = torch.empty((n_rows, n_cols), dtype=dx_dtype, device=x2.device)
    second = dx2_dtype is not None and dx2_dtype != dx_dtype
    dx2 = torch.empty((n_rows, n_cols), dtype=dx2_dtype, device=x2.device) if second else dx
    n_tiles = triton.cdiv(n_rows, BWD_ROWS)
    if need_dw:
        # deterministic: every program sums a fixed range of rows in a fixed order, then torch sums
        # the partials of all programs
        programs = min(n_tiles, DW_PROGRAMS_PER_SM * _sm_count(x2.device))
        tiles_per_program = triton.cdiv(n_tiles, programs)
        programs = triton.cdiv(n_tiles, tiles_per_program)
        partial = torch.empty((programs, n_cols), dtype=torch.float32, device=x2.device)
    else:
        programs, tiles_per_program, partial = n_tiles, 1, rstd
    has_dsum = dsum2 is not None
    dsum2 = dsum2 if has_dsum else dy2
    rms_norm_rows_bwd[(programs,)](
        dy2, dy2.stride(0), x2, x2.stride(0), weight, rstd, dsum2, dsum2.stride(0),
        dx, dx.stride(0), dx2, dx2.stride(0), partial, n_rows, n_cols, tiles_per_program,
        ADD_GRAD=has_dsum, SECOND=second, DW=need_dw, ROWS=BWD_ROWS, BLOCK=block,
        num_warps=BWD_WARPS,
    )
    dw = partial.sum(0).to(weight.dtype) if need_dw else None
    return dx, (dx2 if second else None), dw


def _backward_split(dy2, r2, b2, weight, rstd, dsum2, dx_dtype, dx2_dtype):
    """_backward for a frozen weight, the norm input given as its two addends r2 + b2."""
    n_rows, n_cols = r2.shape
    dx = torch.empty((n_rows, n_cols), dtype=dx_dtype, device=r2.device)
    second = dx2_dtype is not None and dx2_dtype != dx_dtype
    dx2 = torch.empty((n_rows, n_cols), dtype=dx2_dtype, device=r2.device) if second else dx
    has_dsum = dsum2 is not None
    dsum2 = dsum2 if has_dsum else dy2
    rms_norm_rows_bwd_split[(triton.cdiv(n_rows, BWD_ROWS),)](
        dy2, dy2.stride(0), r2, r2.stride(0), b2, b2.stride(0), weight, rstd, dsum2, dsum2.stride(0),
        dx, dx.stride(0), dx2, dx2.stride(0), n_rows, n_cols,
        ADD_GRAD=has_dsum, SECOND=second, ROWS=BWD_ROWS, BLOCK=triton.next_power_of_2(n_cols),
        num_warps=BWD_WARPS,
    )
    return dx, (dx2 if second else None)


def warm_bwd_split():
    """warmup.py registration of rms_norm_rows_bwd_split (registered by patch_qwen2_decoder_layer
    when the split sum or fused_proj's attention recompute is on): the class training launches,
    fp32 residual, half attention output, the residual's gradient added, dx again in half."""
    from . import _on
    from .warmup import HIDDEN, SKIPPED, TOKENS, compile_kernel, training_half_dtype

    if not _on("SOUP_SOLUTION_RMSNORM"):
        return SKIPPED
    f32, half = torch.float32, training_half_dtype()
    compile_kernel(rms_norm_rows_bwd_split, f32, HIDDEN, f32, HIDDEN, half, HIDDEN, f32, f32,
                   f32, HIDDEN, f32, HIDDEN, half, HIDDEN, TOKENS, HIDDEN,
                   ADD_GRAD=True, SECOND=True, ROWS=BWD_ROWS, BLOCK=triton.next_power_of_2(HIDDEN),
                   num_warps=BWD_WARPS)
    return None


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        x2 = _rows(x)
        y = torch.empty(x2.shape, dtype=torch.promote_types(weight.dtype, x.dtype), device=x.device)
        rstd = torch.empty(x2.shape[0], dtype=torch.float32, device=x.device)
        _forward(x2, None, None, weight, y, rstd, eps)
        ctx.save_for_backward(x2, weight, rstd)
        ctx.shape = x.shape
        return y.view(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x2, weight, rstd = ctx.saved_tensors
        dx, _, dw = _backward(_rows(dy), x2, weight, rstd, None, x2.dtype, None,
                              ctx.needs_input_grad[1])
        return dx.view(ctx.shape), dw, None


class AddRMSNormFunction(torch.autograd.Function):
    """(residual + branch, rmsnorm(residual + branch)) in one pass; the backward adds the gradient
    of the new residual to the norm's and gives it to both addends.

    `addends` (see `post_attention_addends`): None keeps the sum for the backward; KEEP_BRANCH keeps
    the residual and the branch instead (the residual is kept by the input norm anyway); an object
    with `branch()` / `branch_done()` (fused_proj's attention recompute) keeps the residual only and
    gets the branch from that object in the backward."""

    @staticmethod
    def forward(ctx, residual, branch, weight, eps, addends=None, offload=False):
        r2, b2 = _rows(residual), _rows(branch)
        sum_dtype = torch.promote_types(residual.dtype, branch.dtype)
        s = torch.empty(r2.shape, dtype=sum_dtype, device=residual.device)
        y = torch.empty(r2.shape, dtype=torch.promote_types(weight.dtype, sum_dtype),
                        device=residual.device)
        rstd = torch.empty(r2.shape[0], dtype=torch.float32, device=residual.device)
        _forward(r2, b2, s, weight, y, rstd, eps)
        if addends is not None and residual.dtype != sum_dtype:
            addends = None  # the split backward re-forms the sum in the residual's dtype
        ctx.addends = addends
        # with addends, the residual is read back in the backward from the input norm that keeps
        # it (residual_node: no second saved reference, offloaded once), else saved here
        ctx.residual_node = _input_norm_node(residual) if addends is not None else None
        kept = s if addends is None else (None if ctx.residual_node is not None else r2)
        extra = (b2,) if addends is KEEP_BRANCH else ()
        if offload and kept is not None:
            from .offload import mark_residual

            mark_residual(kept)  # taken by offload.py's pack hook when the forward returns
        if ctx.residual_node is not None:
            ctx.save_for_backward(weight, rstd, *extra)
        else:
            ctx.save_for_backward(kept, weight, rstd, *extra)
        ctx.shape = residual.shape
        ctx.dtypes = (residual.dtype, branch.dtype)
        return s.view(residual.shape), y.view(residual.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, ds, dy):
        residual_dtype, branch_dtype = ctx.dtypes
        addends = ctx.addends
        dsum = None if ds is None else _rows(ds)
        if addends is None:
            s, weight, rstd = ctx.saved_tensors
            d_res, d_branch, dw = _backward(_rows(dy), s, weight, rstd, dsum,
                                            residual_dtype, branch_dtype, ctx.needs_input_grad[2])
        else:
            if ctx.residual_node is not None:
                weight, rstd, *kept = ctx.saved_tensors
                r2 = ctx.residual_node.saved_tensors[0]
                ctx.residual_node = None
            else:
                r2, weight, rstd, *kept = ctx.saved_tensors
            b2 = kept[0] if addends is KEEP_BRANCH else _rows(addends.branch())
            if ctx.needs_input_grad[2]:  # a trained norm weight: the general kernel on the sum
                s = torch.add(r2, b2.to(r2.dtype))  # the forward's fp32 add of the same values
                d_res, d_branch, dw = _backward(_rows(dy), s, weight, rstd, dsum, residual_dtype,
                                                branch_dtype, True)
            else:
                d_res, d_branch = _backward_split(_rows(dy), r2, b2, weight, rstd, dsum,
                                                  residual_dtype, branch_dtype)
                dw = None
            if addends is not KEEP_BRANCH:
                addends.branch_done()
            ctx.addends = None
        d_res = d_res.view(ctx.shape)
        # as torch's add backward, both addends get the same tensor when their dtypes agree
        d_branch = d_res if d_branch is None else d_branch.view(ctx.shape)
        return d_res, d_branch, dw, None, None, None


def _input_norm_node(residual):
    """The input norm's node that keeps `residual` as its saved sum (an AddRMSNormFunction of the
    layer's input that kept the sum itself), else None."""
    node = getattr(residual, "grad_fn", None)
    if (isinstance(node, AddRMSNormFunction._backward_cls) and residual.output_nr == 0
            and getattr(node, "addends", 0) is None):
        return node
    return None


def rmsnorm(x, weight, eps):
    """Qwen2RMSNorm.forward(x) for a CUDA tensor x (…, C) and an fp32 weight (C,)."""
    return RMSNormFunction.apply(x, weight, float(eps))


def norm_source(y):
    """The autograd node of the norm whose output `y` is (RMSNormFunction's output, or the normed
    output of AddRMSNormFunction), when `rebuild` can recompute y from what that node keeps for its
    own backward; else None. The node's saved tensors live until its backward, which runs after the
    backward of anything that consumed y."""
    node = getattr(y, "grad_fn", None)
    if node is None or y.dtype != torch.float32:
        return None
    if isinstance(node, RMSNormFunction._backward_cls) and y.output_nr == 0:
        return node
    if (isinstance(node, AddRMSNormFunction._backward_cls) and y.output_nr == 1
            and getattr(node, "addends", None) is None):  # a split node does not keep the sum
        return node
    return None


def rebuild(node, dtype):
    """The output of the norm `node` (from `norm_source`) rounded to `dtype`, (rows, C), bit for bit
    what `y.to(dtype)` gave in the forward: the saved input (or sum) times the saved rstd, rounded to
    the input dtype, times the weight, the fp32 product rounded to `dtype`. One launch that reads
    the rows once and writes them in `dtype`; no reduction, so no dependence on the order of the
    sum of squares."""
    source, weight, rstd = node.saved_tensors
    n_rows, n_cols = source.shape
    out = torch.empty((n_rows, n_cols), dtype=dtype, device=source.device)
    rms_norm_rows_rebuild[(triton.cdiv(n_rows, FWD_ROWS),)](
        source, source.stride(0), rstd, weight, out, out.stride(0), n_rows, n_cols,
        ROWS=FWD_ROWS, BLOCK=triton.next_power_of_2(n_cols), num_warps=FWD_WARPS)
    return out


def warm_rebuild():
    """warmup.py registration for the rebuild kernel (registered by fused_proj.py's attention patch,
    its one user): the class training launches, fp32 rows of the residual width in the run's half
    dtype."""
    from . import _on
    from .warmup import HIDDEN, SKIPPED, TOKENS, compile_kernel, training_half_dtype

    if not _on("SOUP_SOLUTION_RMSNORM"):
        return SKIPPED
    compile_kernel(rms_norm_rows_rebuild, torch.float32, HIDDEN, torch.float32, torch.float32,
                   training_half_dtype(), HIDDEN, TOKENS, HIDDEN,
                   ROWS=FWD_ROWS, BLOCK=triton.next_power_of_2(HIDDEN), num_warps=FWD_WARPS)
    return None


def add_rmsnorm(residual, branch, weight, eps, addends=None, offload=False):
    """(r, y) with r = residual + branch and y = Qwen2RMSNorm(r), same shapes; `addends`: what the
    backward keeps (AddRMSNormFunction); `offload`: the fp32 tensor it keeps is a layer input that
    offload.py may move to host memory (SOUP_SOLUTION_OFFLOAD_RESIDUAL)."""
    return AddRMSNormFunction.apply(residual, branch, weight, float(eps), addends, offload)


def post_attention_addends(residual, branch):
    """What the post-attention norm keeps for its backward in place of the fp32 sum (None: the sum,
    as before). The attention output's recompute object when fused_proj.py attached one to it
    (SOUP_SOLUTION_RECOMPUTE_ATTN); else, with SOUP_SOLUTION_SPLIT_SUM=1, KEEP_BRANCH when the
    residual is a layer input the input norm keeps anyway (it requires grad: the input norm's node
    exists and saved it; the embedding output into layer 0 does not, there the sum stays cheaper)."""
    if (not torch.is_grad_enabled() or residual.dtype != torch.float32
            or branch.dtype not in (torch.float16, torch.bfloat16) or not residual.is_contiguous()):
        return None
    recompute = getattr(branch, "_soup_recompute", None)
    if recompute is not None and recompute.accepts(branch):
        return recompute
    if _SPLIT["keep"] and residual.requires_grad and branch.is_contiguous():
        return KEEP_BRANCH
    return None


def reference_rmsnorm(x, weight, eps):
    """Qwen2RMSNorm.forward of transformers 5.16.1, for tests and the benchmark."""
    input_dtype = x.dtype
    h = x.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + eps)
    return weight * h.to(input_dtype)


def fits(x, weight):
    """True when the kernels can take this input and weight; anything else takes the original."""
    return (torch.is_tensor(x) and torch.is_tensor(weight) and x.is_cuda and x.dim() in (2, 3)
            and x.dtype in DTYPES and weight.dtype == torch.float32 and weight.dim() == 1
            and weight.device == x.device and 0 < x.shape[-1] <= MAX_COLS
            and x.shape[-1] == weight.shape[0] and x.numel() > 0)


def _number(eps):
    return isinstance(eps, (int, float)) and not isinstance(eps, bool)


def patch_qwen2_rmsnorm():
    """Qwen2RMSNorm.forward through the Triton kernel; CPU, other dtypes and fake modules take the
    original."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2RMSNorm
    if getattr(cls, "_soup_rmsnorm", False):
        return
    original = getattr(cls, "_soup_original_forward", None) or cls.forward

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        eps = getattr(self, "variance_epsilon", None)
        if not _number(eps) or not fits(hidden_states, weight):
            return original(self, hidden_states)
        return rmsnorm(hidden_states, weight, eps)

    cls.forward = forward
    cls._soup_original_forward = original
    cls._soup_rmsnorm = True


def _plain_norm(norm, norm_cls):
    """A Qwen2RMSNorm called as the original would call it: no hooks to skip, no own forward."""
    return (isinstance(norm, norm_cls) and not norm._forward_hooks and not norm._forward_pre_hooks
            and "forward" not in norm.__dict__ and _number(getattr(norm, "variance_epsilon", None)))


def _addable(residual, branch):
    return (torch.is_tensor(branch) and branch.shape == residual.shape
            and branch.device == residual.device and branch.dtype in DTYPES)


def add_norm(norm, residual, branch, norm_cls, offload=False):
    """(residual + branch, norm(residual + branch)) for a Qwen2RMSNorm `norm`: one kernel when the
    norm is a plain one the kernel fits, else the add and the norm's own forward. The sum is the
    base's `residual + branch` (fp32 residual, half branch: summed and kept in fp32)."""
    if (_plain_norm(norm, norm_cls) and fits(residual, norm.weight) and _addable(residual, branch)):
        return add_rmsnorm(residual, branch, norm.weight, norm.variance_epsilon, offload=offload)
    total = residual + branch
    return total, norm(total)


def _offload_inputs():
    """Whether the layers' fp32 inputs go to host memory (offload.py, SOUP_SOLUTION_OFFLOAD_RESIDUAL;
    only an active Offloader takes them, i.e. inside Soup's training)."""
    from .offload import _RESIDUAL

    return _RESIDUAL["on"] and torch.is_grad_enabled()


def patch_qwen2_decoder_layer():
    """Qwen2DecoderLayer.forward with `residual + attn_out` and post_attention_layernorm in one
    kernel; the calls of input_layernorm, self_attn and mlp and their arguments are the original's.

    Inside unpad.varlen_model_forward the layer-end `residual + mlp_out` is carried to the next
    layer and summed in that layer's input norm kernel instead (one fp32 add per layer less each
    way): called with `_soup_carry=True` the layer returns (mlp_out, residual) unsummed, and called
    with `_soup_branch=mlp_out` of the previous layer it starts from `hidden_states + branch`. The
    branch comes first: weight_stream.py marks output[0] of a layer to bring the layer's weights in
    before its backward, and the branch's gradient is the one that leads into this layer's MLP (the
    residual's leads into the post-attention norm, which also waits for the MLP's)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2DecoderLayer
    if getattr(cls, "_soup_add_rmsnorm", False):
        return
    if _SPLIT["keep"] or os.environ.get("SOUP_SOLUTION_RECOMPUTE_ATTN", "0").strip().lower() not in (
            "0", "false", "no", "off", ""):
        from .warmup import register

        register(warm_bwd_split, name="rms_norm_rows_bwd_split")
    original = getattr(cls, "_soup_original_forward", None) or cls.forward
    norm_cls = modeling_qwen2.Qwen2RMSNorm

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=False, position_embeddings=None, _soup_branch=None, _soup_carry=False,
                **kwargs):
        norm = getattr(self, "post_attention_layernorm", None)
        if not _plain_norm(norm, norm_cls) or not fits(hidden_states, norm.weight):
            if _soup_branch is not None:
                hidden_states = hidden_states + _soup_branch
            return original(self, hidden_states, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, position_embeddings=position_embeddings, **kwargs)
        offload = _offload_inputs()
        if _soup_branch is not None:
            residual, hidden_states = add_norm(self.input_layernorm, hidden_states, _soup_branch,
                                               norm_cls, offload=offload)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if (torch.is_tensor(hidden_states) and hidden_states.shape == residual.shape
                and hidden_states.device == residual.device and hidden_states.dtype in DTYPES):
            residual, hidden_states = add_rmsnorm(residual, hidden_states, norm.weight,
                                                  norm.variance_epsilon,
                                                  post_attention_addends(residual, hidden_states),
                                                  offload=offload)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if _soup_carry and _addable(residual, hidden_states):
            return hidden_states, residual
        hidden_states = residual + hidden_states
        return hidden_states

    forward._soup_carries = True  # unpad.varlen_model_forward carries the residual through it
    cls.forward = forward
    cls._soup_original_forward = original
    cls._soup_add_rmsnorm = True
