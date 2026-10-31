"""Fused projections of a Qwen2 layer: q+k+v in one GEMM with the LoRA adapters added by addmm, the
attention in half from end to end, gate+up in one GEMM.

In the base, every 4-bit linear layer is its own chain: bitsandbytes casts the fp32 input to half,
rebuilds the weight, runs the GEMM and casts the result back to fp32; peft clones that result, runs
dropout on the fp32 input, autocast casts the dropped input and both adapter weights to half, two
small GEMMs, a multiply by `scaling` and an fp32 add. With unpad, rope and nf4 in place that is still
about 40 kernel launches per attention forward and more in the backward, and q, k, v live in fp32
until SDPA casts them back to half.

Attention here (inside `unpad.varlen`, the packed training step):
- the normed input is cast to half once (x16);
- the q, k and v weights are dequantized by `nf4.dequantize_nf4` into row slices of one shared
  (2048, 1536) buffer and one addmm with the concatenated bias gives qkv (N, 2048) in half;
- LoRA of q and v: the four adapter matrices go to half in one multi-tensor copy; dropout is drawn
  once for both adapters on x16 expanded to (2, N, D) (two independent masks, as the base has one
  dropout call per adapter), one bmm gives both h = dropout(x16) @ A.T, and
  `addmm_(qkv_slice, h, B.T, alpha=scaling)` adds the adapter output into the q and v columns of
  qkv with one rounding;
- the rotary embedding of q and k is one launch of rope.py's kernel (fp32 math, half out);
- q, k, v leave the function as (B, H, N, D) token-major views; attention is unpad.attention, one
  variable-length launch for all rows (and the alignment tail) with the kv heads expanded inside it
  and not saved, its token-major output goes straight to o_proj (dequant + GEMM).
Attention recompute (SOUP_SOLUTION_RECOMPUTE_ATTN=1|auto, `_AttnRecompute`): the attention keeps q,
the compact k and v, its output and the log-sum-exp for its backward (7 KB per token per layer),
and the post-attention norm keeps the fp32 sum residual + o_proj output (6 KB). In a recomputing
layer neither is kept: in the backward, the post-attention norm asks for the o_proj output again
and the layer's block runs once more with what is kept anyway (x16 rebuilt from the input norm, h,
the half adapters, the rotary tables, the cu_seqlens) and the weights the backward rebuilds anyway:
q|k|v GEMM + the adapters' addmm + rope, the attention kernel's forward, the o_proj GEMM. Every op
is the forward's, on the same operands and shapes (the same cuBLAS algorithm and the same kernel,
neither with atomics), so the rebuilt tensors are the forward's bit for bit and the gradients are
those of the kept path. About 0.5 ms per layer at 830 tokens; `auto` recomputes only as many layers
(from layer 0) as keep a step's attention memory at that of SOUP_SOLUTION_RECOMPUTE_TOKENS tokens
(640) without the recompute, so short steps pay nothing.

Only h (N x 32), the half adapter weights and the CUDA RNG state are kept for the backward; x16 is
rebuilt in the backward, bit for bit, from the input and rstd that the input norm (rmsnorm.py) keeps
for its own backward (`rmsnorm.rebuild`; kept as before when the input is not such a norm's
output), and the dropout masks are drawn again from that state (the same kernel on the same input
gives the same masks), so no mask, no x16 and no fp32 q/k/v is stored. The backward rebuilds the weights,
writes dq and dk through the rope kernel and dv straight into one (N, 2048) gradient, computes the
adapters' share of the input gradient in half (one bmm, then the two masks applied by one mul and
one addcmul into fp32), and adds the three projections' share to it with one in-place addmm in fp32
(`out_dtype=torch.float32`, the masks' 1 / (1 - p) as the addmm's beta).

MLP: x16 once, with zero rows appended up to a multiple of 8 tokens (with N % 8 != 0 cuBLAS runs
cutlass align1/align2 kernels on the T4, 2-3x slower than its aligned ones, for all five GEMMs of
the MLP; the zero rows stay zero and are cut off on the way out), gate and up rebuilt into one
(17920, 1536) buffer and one GEMM computes (gate|up) transposed, (17920, N padded), so that gate and
up are two contiguous row blocks for the kit's example SwiGLU kernel (it needs contiguous inputs)
without any copy; down_proj reads the transposed product directly. (gate|up) in half is kept for the
backward, as in mlp16.py (offload.py moves it to host memory), or, with
SOUP_SOLUTION_MLP_MEMORY=recompute (set_mlp_recompute), only x16, and the backward recomputes
(gate|up) by the same GEMM with the weights it rebuilds for the input gradient anyway. The MLP
returns half: the base's MLP output is the fp16 GEMM result cast to fp32, and the residual add
promotes half to fp32 with the same values.

Numerics: the GEMMs, their fp32 accumulation and the half outputs are the base's. What changes is the
rounding order around them: q/k/v are rounded to half once after the adapter is added (the base
adds an fp16 adapter output to an fp32 base output, rotates in fp32 and lets SDPA round), and the
input gradient is summed over the three projections inside one GEMM. Differences are at the level of
one half ulp. The dropout masks have the base's distribution but not its random stream (unpad.py
already changes the shapes the base draws on).

Guards: anything that is not this exact layer (no `unpad.varlen` context, a mask or a cache, CPU,
another autocast dtype, q/k/v/o or gate/up/down that are not bitsandbytes Linear4bit with an NF4
quant state, adapters other than one plain LoRA on q and on v, merged or disabled adapters, dropout
that is not nn.Dropout/Identity, module hooks, shapes where bitsandbytes would run its own small-M
4-bit GEMM) calls the forward that was in place before this patch.

Order in apply(): after unpad's `patch_qwen2_attention` and mlp16's `patch_qwen2_mlp_half`. Run later,
unpad's patch would wrap this one and call it only outside `varlen`, and mlp16's would replace the
forward from `_soup_original_forward` and drop this one.
"""

import math
import os
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from . import int8_bwd, nf4, rope, weight_stream
from .nf4 import dequantize_nf4
from .swiglu import BLOCK as SWIGLU_BLOCK
from .swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel
from .rmsnorm import norm_source, rebuild
from .unpad import (_CTX, _compact, _cu_seqlens, _efficient_backward, _expand_heads, kernel_operands,
                    sequences, varlen_forward)
from .unpad import attention as packed_attention

_STATE = {"enabled": True}
_HALVES = (torch.float16, torch.bfloat16)
_INPUTS = (torch.float32, torch.float16, torch.bfloat16)
_BUFFERS = {}  # (device, dtype) -> flat buffer that holds the dequantized weights of one GEMM


def _gate_up_split():
    """Gate and up as two GEMMs over one half-size weight buffer (the default: 26 MiB less of
    persistent buffer, one more launch each way); SOUP_SOLUTION_FUSED_GU_SPLIT=0: one GEMM over a
    full-size buffer. T4, offload mode, two runs of the same minute: peak 1991 -> 1971 MiB, steps 3+
    24.35 -> 23.58 s (noise: no dequantization more); recompute mode: -40 MiB, ~+4.5 ms per step
    (up is rebuilt twice in the backward)."""
    return os.environ.get("SOUP_SOLUTION_FUSED_GU_SPLIT", "1") not in ("0", "false", "no")


# ---------------------------------------------------------------------------------------------
# weights


# What the shared buffer holds (SOUP_SOLUTION_CPU_DIET): the key of the last dequantization into it
# and the views it returned. Every _buffer() call forgets it (whoever asks for the buffer writes
# into it); _weights / _attn_weights set it after their launch.
_HELD = {"key": None, "views": None}
_VIEWS = {}  # views of the shared buffers by (device, dtype, what), dropped when a buffer is replaced
_T_OF = {}  # id(view) -> (view, view.t()) for the weight views kept in _VIEWS


def _t(w):
    """w.t(): made once for a weight view the shared buffer keeps (_VIEWS), else now."""
    found = _T_OF.get(id(w))
    return found[1] if found is not None and found[0] is w else w.t()


def _buffer(device, dtype, numel):
    _HELD["key"] = _HELD["views"] = None
    key = (device, dtype)
    buf = _BUFFERS.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
        _BUFFERS[key] = buf
        _VIEWS.clear()
        _T_OF.clear()
    return buf


def _held(key):
    """The views of what the shared buffer holds when that is `key` (nothing written into it since
    the launch that made it), else None."""
    return _HELD["views"] if _HELD["key"] == key else None


def _weights(bases, dtype, device, offset=0):
    """The dequantized weights of `bases` stacked by rows, (sum of out, in), a view of the shared
    buffer starting at element `offset`. Valid until the next call on this device: every GEMM that
    reads it is queued on the same stream before the next dequantization overwrites it.

    With SOUP_SOLUTION_CPU_DIET (the default) three weights go in one launch; with
    SOUP_SOLUTION_DEQUANT_REUSE as well (the default) the kernel's operands come from _nf4_args and a
    call for exactly what the buffer holds (the same weights from the same offset, nothing written
    into the buffer since) returns it without a launch: the values in the buffer are those the
    launch would write again."""
    k = bases[0].in_features
    if _CPU["diet"] and not _CPU["reuse"]:  # the diet as before the reuse
        rows = sum(b.out_features for b in bases)
        flat = _buffer(device, dtype, offset + rows * k)
        w = flat[offset:offset + rows * k].view(rows, k)
        packed = [_nf4_packed(base, dtype) for base in bases]
        if all(p is not None for p in packed):
            if len(packed) == 3:
                _dequant3(packed, w)
            else:
                row = 0
                for base, (plain, state) in zip(bases, packed):
                    n = base.out_features
                    nf4._launch(plain, state, dtype, w[row:row + n])
                    row += n
            return w
    elif _CPU["diet"]:
        found = [_nf4_args(base, dtype) for base in bases]
        if None not in found:
            key = (device, dtype, offset, tuple(f[0] for f in found))
            views = _held(key)
            if views is not None:
                return views[0]
            shape = tuple(b.out_features for b in bases)
            rows = sum(shape)
            flat = _buffer(device, dtype, offset + rows * k)
            vkey = (device, dtype, offset, shape, k)
            views = _VIEWS.get(vkey)
            if views is None:
                w = flat[offset:offset + rows * k].view(rows, k)
                parts, row = [], 0
                for n in shape:
                    parts.append(w[row:row + n])
                    row += n
                views = _VIEWS[vkey] = (w, parts)
                _T_OF[id(w)] = (w, w.t())
            if len(found) == 3:
                _launch_many([f[1] for f in found], views[0])
            else:
                for part, (_, operands) in zip(views[1], found):
                    _launch_one(operands, part)
            _HELD["key"], _HELD["views"] = key, views
            return views[0]
    rows = sum(b.out_features for b in bases)
    flat = _buffer(device, dtype, offset + rows * k)
    w = flat[offset:offset + rows * k].view(rows, k)
    row = 0
    for base in bases:
        n = base.out_features
        dequantize_nf4(base.weight, base.weight.quant_state, dtype, out=w[row:row + n])
        row += n
    return w


def _attn_weights(plan, device):
    """(W_q|k|v (nq + nk + nv, D), W_o) of an attention plan's layer in the shared buffer, one after
    the other: one launch for the four weights, or none when the buffer still holds them. With the
    attention recompute a layer's backward then dequantizes them once (the recompute's launch; its
    o_proj, the o_proj backward and the q/k/v backward read the same buffer, nothing is written into
    it in between), and its forward once (q/k/v, then o_proj after the attention kernel). None
    without SOUP_SOLUTION_CPU_DIET and SOUP_SOLUTION_DEQUANT_REUSE or when a weight is not on nf4.py's
    kernel (then _weights)."""
    if not (_CPU["diet"] and _CPU["reuse"]):
        return None
    half = plan.half
    bases = plan.bases + (plan.o_base,)
    # the operands and key, kept on the plan (one layer's forward and backward of a step) while the
    # four Params4bit and their quant states are the same objects: _nf4_args' checks once per plan
    weights = tuple(base.weight for base in bases)
    memo = getattr(plan, "weights", None)
    if (memo is not None and memo[0] == device
            and all(a is b for a, b in zip(memo[1], weights))
            and all(w.quant_state is state for w, state in zip(weights, memo[2]))):
        key, operands = memo[3], memo[4]
    else:
        found = [_nf4_args(base, half) for base in bases]
        if None in found:
            return None
        key = (device, half, "attn", tuple(f[0] for f in found))
        operands = [f[1] for f in found]
        plan.weights = (device, weights, tuple(w.quant_state for w in weights), key, operands)
    views = _held(key)
    if views is not None:
        return views
    q, k, v, o = bases
    rows = q.out_features + k.out_features + v.out_features
    qkv, out = rows * q.in_features, o.out_features * o.in_features
    flat = _buffer(device, half, qkv + out)
    vkey = (device, half, "attn", rows, q.in_features, o.out_features, o.in_features)
    views = _VIEWS.get(vkey)
    if views is None:
        views = _VIEWS[vkey] = (flat[:qkv].view(rows, q.in_features),
                                flat[qkv:qkv + out].view(o.out_features, o.in_features))
        for w in views:
            _T_OF[id(w)] = (w, w.t())
    _launch_many(operands, views[0])
    _HELD["key"], _HELD["views"] = key, views
    return views


# CPU side of the step (SOUP_SOLUTION_CPU_DIET=0: every launch as before). Every Triton launch of
# solution/ costs about 50 us of CPU on the training path (Triton's launcher plus the harness's
# timing events; T4 cProfile, task C1: 681 launches per step), nf4._fast_path's checks 13 us more,
# and a layer made 14 dequantizations. With the diet: q, k, v and o in one launch, reused by the
# backward from the recompute's launch (_attn_weights), any other repeat of what the buffer holds
# skipped (_weights), the kernel's operands kept per packed weight (_nf4_args): 4 launches per layer
# and every Params4bit attribute read of the dequantizations less
_CPU = {"diet": os.environ.get("SOUP_SOLUTION_CPU_DIET", "1").strip().lower()
        not in ("0", "false", "no", ""),
        "reuse": os.environ.get("SOUP_SOLUTION_DEQUANT_REUSE", "1").strip().lower()
        not in ("0", "false", "no", "")}
_current_device = torch._C._cuda_getDevice


def _nf4_args(base, dtype):
    """(key, operands) of a 4-bit layer's weight for nf4.py's kernel, None when the kernel does not
    take it. operands: the packed weight as int32 words, the absmax codes, state2.code,
    state2.absmax, the offset, the NF4 levels, the number of blocks and the launch grid. Kept on the
    layer while the packed weight's storage, the quant state and the state's tensors stay the same
    objects, and the dtype and device (weight_stream.py points a layer's packed weight into its slot
    once), so a step reads no Params4bit attribute for it (each read is a Python __torch_function__
    call of bitsandbytes) and makes no view; `key` names what a launch dequantizes (_HELD). The
    cached operands keep the tensors whose ids are in the key alive: an id is not reused while it
    is in a key."""
    weight = base.weight
    state = weight.quant_state
    state2 = getattr(state, "state2", None)
    with torch._C.DisableTorchFunctionSubclass():
        ptr = weight.data_ptr()
    key = (ptr, id(state), id(state.absmax), id(state.code), id(state.offset), id(state2),
           id(getattr(state2, "absmax", None)), id(getattr(state2, "code", None)), dtype,
           _current_device())
    cached = base.__dict__.get("_soup_nf4_args")
    if cached is not None and cached[0] == key:
        return cached[1]
    plain = weight.data
    found = None
    if nf4._fast_path(plain, state, dtype):
        n_blocks = state.absmax.numel()
        found = (key, (plain.view(-1).view(torch.int32), state.absmax, state2.code, state2.absmax,
                       state.offset, state.code, n_blocks,
                       (triton.cdiv(n_blocks, nf4.BLOCKS_PER_PROGRAM),)))
    base.__dict__["_soup_nf4_args"] = (key, found)
    return found


def _nf4_packed(base, dtype):
    """(plain packed weight, quant state) of a 4-bit layer when nf4.py's kernel takes it, else None
    (the diet without SOUP_SOLUTION_DEQUANT_REUSE). nf4._fast_path's verdict is kept on the layer
    for its packed weight's storage, quant state, dtype and device."""
    weight = base.weight
    state = weight.quant_state
    plain = weight.data  # a Params4bit answers attributes through __torch_function__: read once
    key = (plain.data_ptr(), id(state), dtype, torch.cuda.current_device())
    cached = base.__dict__.get("_soup_nf4_packed")
    if cached is None or cached[0] != key:
        cached = (key, nf4._fast_path(plain, state, dtype))
        base.__dict__["_soup_nf4_packed"] = cached
    return (plain, state) if cached[1] else None


def _launch_one(operands, out):
    """nf4._launch with _nf4_args' operands: the same kernel on the same arguments."""
    nf4.nf4_dequant_kernel[operands[7]](
        *operands[:6], out, operands[6], QBLOCK=nf4.QBLOCK, ABLOCK=nf4.ABLOCK,
        BLOCKS=nf4.BLOCKS_PER_PROGRAM, num_warps=4, enable_fp_fusion=False)


def _launch_many(operands, out):
    """Three or four weights (_nf4_args' operands) into the contiguous elements of `out`, one after
    the other, in one launch: nf4_dequant3_kernel / nf4_dequant4_kernel."""
    blocks = nf4.BLOCKS_PER_PROGRAM
    programs = sum(op[7][0] for op in operands)
    args = [a for op in operands for a in op[:7]]
    kernel = nf4_dequant3_kernel if len(operands) == 3 else nf4_dequant4_kernel
    kernel[(programs,)](*args, out, QBLOCK=nf4.QBLOCK, ABLOCK=nf4.ABLOCK, BLOCKS=blocks,
                        num_warps=4, enable_fp_fusion=False)


@triton.jit
def _nf4_blocks(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, out_ptr,
                n_blocks, program, QBLOCK: tl.constexpr, ABLOCK: tl.constexpr,
                BLOCKS: tl.constexpr):
    # nf4.nf4_dequant_kernel's body for one weight, its programs numbered from 0 (same loads,
    # same products in the same order: the same values)
    blk = program * BLOCKS + tl.arange(0, BLOCKS)
    live = blk < n_blocks
    q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
    scale = tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
    absmax = tl.load(code2_ptr + q) * scale
    absmax = absmax + tl.load(offset_ptr)
    words: tl.constexpr = QBLOCK // 8
    w = tl.load(words_ptr + blk[:, None] * words + tl.arange(0, words)[None, :], mask=live[:, None],
                other=0)
    j = tl.arange(0, 8)
    codes = (w[:, :, None] >> ((j // 2) * 8 + (1 - j % 2) * 4)[None, None, :]) & 15
    levels = tl.reshape(tl.load(nf4_ptr + codes), (BLOCKS, QBLOCK))
    vals = levels * absmax[:, None]
    elem = blk[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    tl.store(out_ptr + elem, vals.to(out_ptr.dtype.element_ty), mask=live[:, None])


@triton.jit
def nf4_dequant3_kernel(
    w0, qa0, c0, s0, o0, l0, n0,  # weight 0: packed words, absmax codes, state2.code, state2.absmax,
    w1, qa1, c1, s1, o1, l1, n1,  # offset, NF4 levels, number of blocks (as nf4_dequant_kernel)
    w2, qa2, c2, s2, o2, l2, n2,
    out_ptr,  # the three weights stacked by rows, contiguous
    QBLOCK: tl.constexpr, ABLOCK: tl.constexpr, BLOCKS: tl.constexpr,
):
    # three weights in one launch: the first programs dequantize weight 0, the next weight 1, the
    # last weight 2, each into its rows of the output
    pid = tl.program_id(0)
    p0 = tl.cdiv(n0, BLOCKS)
    p1 = tl.cdiv(n1, BLOCKS)
    if pid < p0:
        _nf4_blocks(w0, qa0, c0, s0, o0, l0, out_ptr, n0, pid, QBLOCK, ABLOCK, BLOCKS)
    elif pid < p0 + p1:
        _nf4_blocks(w1, qa1, c1, s1, o1, l1, out_ptr + n0.to(tl.int64) * QBLOCK, n1, pid - p0,
                    QBLOCK, ABLOCK, BLOCKS)
    else:
        _nf4_blocks(w2, qa2, c2, s2, o2, l2, out_ptr + (n0 + n1).to(tl.int64) * QBLOCK, n2,
                    pid - p0 - p1, QBLOCK, ABLOCK, BLOCKS)


@triton.jit
def nf4_dequant4_kernel(
    w0, qa0, c0, s0, o0, l0, n0,  # as nf4_dequant3_kernel, one weight more: q, k, v and o of a layer
    w1, qa1, c1, s1, o1, l1, n1,
    w2, qa2, c2, s2, o2, l2, n2,
    w3, qa3, c3, s3, o3, l3, n3,
    out_ptr,  # the four weights one after the other, contiguous
    QBLOCK: tl.constexpr, ABLOCK: tl.constexpr, BLOCKS: tl.constexpr,
):
    pid = tl.program_id(0)
    p0 = tl.cdiv(n0, BLOCKS)
    p1 = tl.cdiv(n1, BLOCKS)
    p2 = tl.cdiv(n2, BLOCKS)
    if pid < p0:
        _nf4_blocks(w0, qa0, c0, s0, o0, l0, out_ptr, n0, pid, QBLOCK, ABLOCK, BLOCKS)
    elif pid < p0 + p1:
        _nf4_blocks(w1, qa1, c1, s1, o1, l1, out_ptr + n0.to(tl.int64) * QBLOCK, n1, pid - p0,
                    QBLOCK, ABLOCK, BLOCKS)
    elif pid < p0 + p1 + p2:
        _nf4_blocks(w2, qa2, c2, s2, o2, l2, out_ptr + (n0 + n1).to(tl.int64) * QBLOCK, n2,
                    pid - p0 - p1, QBLOCK, ABLOCK, BLOCKS)
    else:
        _nf4_blocks(w3, qa3, c3, s3, o3, l3, out_ptr + (n0 + n1 + n2).to(tl.int64) * QBLOCK, n3,
                    pid - p0 - p1 - p2, QBLOCK, ABLOCK, BLOCKS)


def _warm_dequant4():
    """warmup.py registration of nf4_dequant4_kernel: q, k, v and o of the run (every block count
    divisible by 16, so one class), in the run's half dtype."""
    from .warmup import HIDDEN, KV_HEADS, Q_HEADS, HEAD_DIM, compile_kernel, training_half_dtype

    per = []
    for rows, cols in ((Q_HEADS * HEAD_DIM, HIDDEN), (KV_HEADS * HEAD_DIM, HIDDEN),
                       (KV_HEADS * HEAD_DIM, HIDDEN), (HIDDEN, Q_HEADS * HEAD_DIM)):
        per += [torch.int32, torch.uint8, torch.float32, torch.float32, torch.float32, torch.float32,
                rows * cols // nf4.QBLOCK]
    compile_kernel(nf4_dequant4_kernel, *per, training_half_dtype(), QBLOCK=nf4.QBLOCK,
                   ABLOCK=nf4.ABLOCK, BLOCKS=nf4.BLOCKS_PER_PROGRAM, num_warps=4,
                   enable_fp_fusion=False)


def _dequant3_args(plain, state):
    state2 = state.state2
    return (plain.view(-1).view(torch.int32), state.absmax, state2.code, state2.absmax,
            state.offset, state.code, state.absmax.numel())


def _dequant3(packed, out):
    """The three weights of `packed` ((plain, quant state) each, nf4.py's fast path) into the
    contiguous rows of `out`, bit for bit nf4.dequantize_nf4 of each, in one launch."""
    args = [a for plain, state in packed for a in _dequant3_args(plain, state)]
    blocks = nf4.BLOCKS_PER_PROGRAM
    programs = sum(triton.cdiv(state.absmax.numel(), blocks) for _, state in packed)
    nf4_dequant3_kernel[(programs,)](*args, out, QBLOCK=nf4.QBLOCK, ABLOCK=nf4.ABLOCK, BLOCKS=blocks,
                                     num_warps=4, enable_fp_fusion=False)


def _warm_dequant3():
    """warmup.py registration of nf4_dequant3_kernel: q, k and v of the run (every block count
    divisible by 16, so one class), in the run's half dtype."""
    from .warmup import HIDDEN, KV_HEADS, Q_HEADS, HEAD_DIM, compile_kernel, training_half_dtype

    per = []
    for rows in (Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM):
        per += [torch.int32, torch.uint8, torch.float32, torch.float32, torch.float32, torch.float32,
                rows * HIDDEN // nf4.QBLOCK]
    compile_kernel(nf4_dequant3_kernel, *per, training_half_dtype(), QBLOCK=nf4.QBLOCK,
                   ABLOCK=nf4.ABLOCK, BLOCKS=nf4.BLOCKS_PER_PROGRAM, num_warps=4,
                   enable_fp_fusion=False)


def _cached_bias(owner, bases, dtype):
    """The biases of `bases` concatenated in `dtype` (zeros for a layer without one), cached on
    `owner` while the parameters keep their storage and version; None when no layer has a bias."""
    biases = [b.bias for b in bases]
    if all(b is None for b in biases):
        return None
    key = tuple(None if b is None else (b.data_ptr(), b._version, b.dtype) for b in biases)
    cached = owner.__dict__.get("_soup_fused_bias")
    if cached is not None and cached[0] == key and cached[1].dtype == dtype:
        return cached[1]
    parts = [torch.zeros(base.out_features, dtype=dtype, device=base.weight.device) if b is None
             else b.detach().to(dtype) for base, b in zip(bases, biases)]
    bias = torch.cat(parts)
    owner.__dict__["_soup_fused_bias"] = (key, bias)
    return bias


# ---------------------------------------------------------------------------------------------
# guards


def _no_hooks(*modules):
    return not any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks
                   or m._backward_pre_hooks for m in modules)


def _global_hooks():
    from torch.nn.modules import module as nn_module

    return bool(nn_module._global_forward_hooks or nn_module._global_forward_pre_hooks
                or nn_module._global_backward_hooks or nn_module._global_backward_pre_hooks)


_IMPORTED = {}  # name -> what _imported() found (or None when the import failed)
_FUSES = {}  # _bnb_would_fuse's verdicts by (m, k, n, blocksize, dtype, device index)


def _imported(name):
    """bitsandbytes' Linear4bit ("bnb") or Params4bit ("params4bit"), peft's LoRA Linear4bit
    ("lora") or bitsandbytes' CUDA ops ("ops"), imported once (an import statement in a guard costs
    a lookup through the import system on every call), None when the import fails."""
    if name not in _IMPORTED:
        try:
            if name in ("bnb", "params4bit"):
                import bitsandbytes as bnb

                found = bnb.nn.Linear4bit if name == "bnb" else bnb.nn.Params4bit
            elif name == "lora":
                from peft.tuners.lora.bnb import Linear4bit as found
            else:
                from bitsandbytes.backends.cuda import ops as found
        except Exception:  # noqa: BLE001 - not installed, another layout
            found = None
        _IMPORTED[name] = found
    return _IMPORTED[name]


def _flags(weight):
    """(requires_grad, is_cuda, device) of a weight. A Params4bit's getters are read with torch's
    function override off: bitsandbytes' __torch_function__ only forwards them (each read through
    it is a Python call), so the values are the same."""
    if type(weight) is _imported("params4bit"):
        with torch._C.DisableTorchFunctionSubclass():
            return weight.requires_grad, weight.is_cuda, weight.device
    return weight.requires_grad, weight.is_cuda, weight.device


def _bnb_would_fuse(m, base, dtype, device):
    """Whether bitsandbytes' gemm_4bit would run its own fused 4-bit GEMM for m rows (small M):
    there the base does not dequantize, so this module keeps the original path (as nf4.py does).
    The verdict depends only on the shapes, the dtype and the device: kept by those."""
    key = (m, base.in_features, base.out_features, base.weight.quant_state.blocksize, dtype,
           device.index)
    found = _FUSES.get(key)
    if found is None:
        found = _FUSES[key] = _bnb_would_fuse_uncached(m, base, dtype, device)
    return found


def _bnb_would_fuse_uncached(m, base, dtype, device):
    ops = _imported("ops")
    if ops is None:
        return True
    custom_max_m = getattr(ops, "_gemm_4bit_custom_max_m", None)
    use_custom = getattr(ops, "_gemm_4bit_use_custom_fn", None)
    if not isinstance(custom_max_m, int) or not callable(use_custom):
        return True
    k, n = base.in_features, base.out_features
    if m > custom_max_m or k % base.weight.quant_state.blocksize != 0:
        return False
    try:
        return bool(use_custom(device.index, dtype, m, n, k))
    except Exception:  # noqa: BLE001
        return True


def _plain_4bit(module):
    """`module` when it is a bitsandbytes Linear4bit with an NF4 quant state and frozen weights
    that this module can dequantize and multiply like its own forward does, else None."""
    linear4bit = _imported("bnb")
    if linear4bit is None:
        return None
    if not isinstance(module, linear4bit) or type(module).forward is not linear4bit.forward:
        return None
    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    if state is None or getattr(state, "quant_type", None) != "nf4":
        return None
    requires_grad, is_cuda, _ = _flags(weight)
    if requires_grad:
        return None
    shape = getattr(state, "shape", None)
    if shape is None or tuple(int(s) for s in shape) != (module.out_features, module.in_features):
        return None
    if module.bias is not None and module.bias.requires_grad:
        return None
    if module.compute_dtype not in _HALVES or not is_cuda or not _no_hooks(module):
        return None
    return module


def _lora(module):
    """(base, lora_A, lora_B, dropout p, scaling) of a peft LoRA Linear4bit with exactly one active
    plain LoRA adapter, else None."""
    lora_linear4bit = _imported("lora")
    if lora_linear4bit is None or type(module) is not lora_linear4bit or not _no_hooks(module):
        return None
    if module.disable_adapters or module.merged or getattr(module, "fan_in_fan_out", False):
        return None
    active = module.active_adapters
    if len(active) != 1:
        return None
    name = active[0]
    if name not in module.lora_A or name not in module.lora_B or name in module.lora_variant:
        return None
    lora_a, lora_b = module.lora_A[name], module.lora_B[name]
    if (type(lora_a) is not torch.nn.Linear or type(lora_b) is not torch.nn.Linear
            or lora_a.bias is not None or lora_b.bias is not None or not _no_hooks(lora_a, lora_b)):
        return None
    dropout = module.lora_dropout[name]
    if type(dropout) is torch.nn.Identity:
        p = 0.0
    elif type(dropout) is torch.nn.Dropout and not dropout.inplace and _no_hooks(dropout):
        p = float(dropout.p) if dropout.training else 0.0
    else:
        return None
    scaling = module.scaling.get(name)
    if not isinstance(scaling, (int, float)) or isinstance(scaling, bool) or not 0.0 <= p < 1.0:
        return None
    base = _plain_4bit(module.base_layer)
    if base is None or lora_a.weight.shape != (lora_a.out_features, base.in_features) \
            or lora_b.weight.shape != (base.out_features, lora_a.out_features):
        return None
    return base, lora_a, lora_b, p, float(scaling)


def _half_for(x, half):
    """True when the base would run this layer's attention in `half`: under CUDA autocast with that
    dtype, or with a half input and no autocast."""
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda") == half
    return x.dtype == half


class _AttnPlan:
    __slots__ = ("bases", "o_base", "lora_q", "lora_v", "p", "scaling", "half", "heads", "kv_heads",
                 "head_dim", "bias", "spans", "bounds", "n_tokens", "weights")


def _attention_plan(attn, x, position_embeddings, attention_mask, past_key_values):
    if not _STATE["enabled"]:
        return None
    spans = _CTX["spans"]
    if (spans is None or past_key_values is not None or attention_mask is not None
            or not torch.is_tensor(x) or not x.is_cuda or x.dim() != 3 or x.shape[0] != 1
            or x.dtype not in _INPUTS or x.numel() == 0 or torch.compiler.is_compiling()):
        return None
    if getattr(attn, "sliding_window", None) is not None or _global_hooks():
        return None
    lora_q = _lora(getattr(attn, "q_proj", None))
    lora_v = _lora(getattr(attn, "v_proj", None))
    k_base = _plain_4bit(getattr(attn, "k_proj", None))
    o_base = _plain_4bit(getattr(attn, "o_proj", None))
    if lora_q is None or lora_v is None or k_base is None or o_base is None:
        return None
    q_base, a_q, b_q, p, scaling = lora_q
    v_base, a_v, b_v, p_v, scaling_v = lora_v
    if p != p_v or scaling != scaling_v or a_q.out_features != a_v.out_features:
        return None
    half = q_base.compute_dtype
    bases = (q_base, k_base, v_base)
    if any(b.compute_dtype != half for b in bases + (o_base,)) or not _half_for(x, half):
        return None
    n_tokens, hidden = x.shape[1], x.shape[2]
    head_dim = getattr(attn, "head_dim", None)
    if not isinstance(head_dim, int) or head_dim <= 0 or any(b.in_features != hidden for b in bases):
        return None
    if q_base.out_features % head_dim or k_base.out_features % head_dim \
            or v_base.out_features != k_base.out_features:
        return None
    heads, kv_heads = q_base.out_features // head_dim, k_base.out_features // head_dim
    if kv_heads == 0 or heads % kv_heads or heads // kv_heads != getattr(attn, "num_key_value_groups", 1) \
            or o_base.in_features != heads * head_dim:
        return None
    if any(_flags(t)[2] != x.device for t in (q_base.weight, k_base.weight, v_base.weight,
                                               o_base.weight, a_q.weight, b_q.weight, a_v.weight,
                                               b_v.weight)):
        return None
    if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
        return None
    cos, sin = position_embeddings
    if not (torch.is_tensor(cos) and torch.is_tensor(sin) and cos.dim() == 3 and cos.shape[0] == 1
            and cos.shape[1:] == (n_tokens, head_dim) and sin.shape == cos.shape
            and cos.dtype in _INPUTS and sin.dtype == cos.dtype and cos.device == x.device
            and sin.device == x.device):
        return None
    half_dim = head_dim // 2
    if head_dim % 2 or half_dim < 8 or half_dim > 256 or half_dim & (half_dim - 1):
        return None
    if torch.is_grad_enabled() and (cos.requires_grad or sin.requires_grad):
        return None
    # the spans must follow each other from 0 (pack_batch's do); tokens after the last span are
    # the alignment tail of unpad.varlen_model_forward
    bounds = sequences(spans, n_tokens)
    if bounds is None:
        return None
    if any(_bnb_would_fuse(n_tokens, b, half, x.device) for b in bases + (o_base,)):
        return None
    plan = _AttnPlan()
    plan.bases, plan.o_base, plan.lora_q, plan.lora_v = bases, o_base, (a_q, b_q), (a_v, b_v)
    plan.p, plan.scaling, plan.half = p, scaling, half
    plan.heads, plan.kv_heads, plan.head_dim = heads, kv_heads, head_dim
    plan.bias = _cached_bias(attn, bases, half)
    plan.spans, plan.bounds, plan.n_tokens = spans, bounds, n_tokens
    return plan


# ---------------------------------------------------------------------------------------------
# attention


def _tables(cos, sin):
    if cos.stride() != sin.stride() or cos.stride(-1) != 1:
        cos, sin = cos.contiguous(), sin.contiguous()
    return cos, sin


def _heads(cols, n_tokens, heads, head_dim):
    """(N, heads * head_dim) columns -> (1, heads, N, head_dim) view."""
    return cols.view(1, n_tokens, heads, head_dim).transpose(1, 2)


def _last_dim_contiguous(t):
    return t if t.stride(-1) == 1 else t.contiguous()


# SOUP_SOLUTION_REBUILD_X16=0: _QKVRope keeps its half input for the backward instead of rebuilding
# it from the input norm's saved tensors
_ATTN = {"rebuild_x16": os.environ.get("SOUP_SOLUTION_REBUILD_X16", "1").strip().lower()
         not in ("0", "false", "no", "")}


MASKED_SUM_BLOCK = 1024


@triton.jit
def masked_pair_sum_kernel(g_ptr, m_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # out = g[0] * m[0] + g[1] * m[1] in fp32 for g (2, n) half and m (2, n) bool: the adapters'
    # share of the input gradient through their two dropout masks. The products are exact (a mask
    # is 0 or 1, the sign of a zero included), so the one rounding of the sum is torch's
    # mul-then-addcmul_ rounding, contracted or not
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < n_elements
    g0 = tl.load(g_ptr + offs, mask=ok, other=0.0).to(tl.float32)
    g1 = tl.load(g_ptr + n_elements + offs, mask=ok, other=0.0).to(tl.float32)
    m0 = tl.load(m_ptr + offs, mask=ok, other=0).to(tl.float32)
    m1 = tl.load(m_ptr + n_elements + offs, mask=ok, other=0).to(tl.float32)
    tl.store(out_ptr + offs, g0 * m0 + g1 * m1, mask=ok)


def _masked_pair_sum(g, mask):
    """(N, D) fp32 = g[0] * mask[0] + g[1] * mask[1] for g (2, N, D) half and mask (2, N, D) bool, one
    launch (torch: a mixed-dtype mul and addcmul_, neither vectorized)."""
    _, n, d = g.shape
    g, mask = g.contiguous(), mask.contiguous()
    out = torch.empty((n, d), dtype=torch.float32, device=g.device)
    count = n * d
    masked_pair_sum_kernel[(triton.cdiv(count, MASKED_SUM_BLOCK),)](
        g, mask, out, count, BLOCK=MASKED_SUM_BLOCK)
    return out


def _warm_masked_pair_sum():
    """warmup.py registration of masked_pair_sum_kernel: the class training launches (N x 1536
    elements, always divisible by 16), with the run's half dtype."""
    from .warmup import HIDDEN, TOKENS, compile_kernel, training_half_dtype

    compile_kernel(masked_pair_sum_kernel, training_half_dtype(), torch.bool, torch.float32,
                   TOKENS * HIDDEN, BLOCK=MASKED_SUM_BLOCK)


def _scaled_mm(a, b, alpha, out):
    """out = alpha * (a @ b), in out's dtype (fp32 out from half a and b accumulates in fp32)."""
    if out.dtype == a.dtype:
        return torch.addmm(out, a, b, beta=0.0, alpha=alpha, out=out)
    return torch.addmm(out, a, b, beta=0.0, alpha=alpha, out_dtype=out.dtype, out=out)


class _AdapterHalves:
    """Half copies of the four LoRA matrices (A_q, A_v stacked; B_q, B_v stacked) of every fused
    attention layer. The optimizer steps all of them between two forwards; the first layer that
    finds its matrices changed (a parameter's version or storage) refreshes every changed layer in
    one multi-tensor copy, the other layers find theirs current: one launch per step instead of
    one per layer, the same casts. A step's backward reads these buffers before the optimizer
    changes the parameters and the next forward refreshes them; anything that keeps a graph across
    that point gets autograd's in-place error (the buffers' version), not stale values."""

    def __init__(self):
        self._layers = weakref.WeakKeyDictionary()  # a layer's lora_A module of q -> entry

    @staticmethod
    def _stamp(params):
        return tuple((p.data_ptr(), p._version) for p in params)

    def halves(self, owner, params, half):
        """(a16, b16, their views: _adapter_views) for the layer `owner` with params (A_q, A_v,
        B_q, B_v)."""
        entry = self._layers.get(owner)
        shapes = tuple(tuple(p.shape) for p in params)
        if (entry is None or entry["shapes"] != shapes or entry["dtype"] != half
                or entry["device"] != params[0].device or any(
                    a is not b for a, b in zip(entry["params"], params))):
            (r, d), (nq, _), (nv, _) = shapes[0], shapes[2], shapes[3]
            a16 = torch.empty((2 * r, d), dtype=half, device=params[0].device)
            b16 = torch.empty((nq + nv, r), dtype=half, device=params[0].device)
            entry = {"params": params, "shapes": shapes, "dtype": half, "device": params[0].device,
                     "a16": a16, "b16": b16, "views": [a16[:r], a16[r:], b16[:nq], b16[nq:]],
                     "derived": _adapter_views(a16, b16, r, nq), "stamp": None}
            self._layers[owner] = entry
        if entry["stamp"] != self._stamp(params):
            self._refresh(entry)
        return entry["a16"], entry["b16"], entry["derived"]

    def _refresh(self, first):
        dst, src, done = [], [], []
        for entry in [first] + [e for e in self._layers.values() if e is not first]:
            if entry["dtype"] != first["dtype"] or entry["device"] != first["device"]:
                continue
            stamp = self._stamp(entry["params"])
            if entry["stamp"] != stamp:
                dst += entry["views"]
                src += entry["params"]
                done.append((entry, stamp))
        torch._foreach_copy_(dst, src)
        for entry, stamp in done:
            entry["stamp"] = stamp


_ADAPTERS = _AdapterHalves()


def _adapter_views(a16, b16, r, nq):
    """The views the LoRA products read of the half adapters: A as (2, r, D) and its transpose, A^T,
    B_q, B_v and their transposes (kept with the buffers of _AdapterHalves: made once per layer)."""
    a3 = a16.view(2, r, a16.shape[1])
    bq, bv = b16[:nq], b16[nq:]
    return a3, a3.transpose(1, 2), a16.t(), bq, bv, bq.t(), bv.t()


def _generator(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.default_generators[index]


def _get_rng_state(device):
    """torch.cuda.get_rng_state(device): with SOUP_SOLUTION_DEQUANT_REUSE the generator's own call,
    without the wrapper's device lookups (the same state)."""
    if _CPU["reuse"]:
        return _generator(device).get_state()
    return torch.cuda.get_rng_state(device)


def _set_rng_state(state, device):
    """torch.cuda.set_rng_state(state, device): with SOUP_SOLUTION_DEQUANT_REUSE the generator's own
    call, without the wrapper's defensive clone of `state` (set_state copies the values)."""
    if _CPU["reuse"]:
        _generator(device).set_state(state)
    else:
        torch.cuda.set_rng_state(state, device)


class _QKVRope(torch.autograd.Function):
    """x (1, N, D) -> q, k, v (1, H, N, Dh) in half: q and k rotated, LoRA added to q and v."""

    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute=None):
        q4, k4, v4, keep = _qkv_forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute)
        ctx.save_for_backward(*keep)
        return q4, k4, v4

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k, grad_v):
        return (*_qkv_backward(ctx, ctx.saved_tensors, grad_q, grad_k, grad_v), None, None, None, None)


def _rope_token_major(qkv, n, plan, cos, sin):
    """q and k of a q|k|v product (N, nq + nk + nv) rotated into buffers of their own, and the
    attention kernel's token-major (1, N, H, D) q, k, v (v a view of qkv): the same rope launch and
    the same three tensors (pointers, sizes, strides) as _heads + rope._launch +
    unpad.kernel_operands, without the head-major views in between."""
    half, dim = plan.half, plan.head_dim
    nq, nk = plan.bases[0].out_features, plan.bases[1].out_features
    row = qkv.stride(0)
    q_buf = torch.empty((n, nq), dtype=half, device=qkv.device)
    k_buf = torch.empty((n, nk), dtype=half, device=qkv.device)
    rope._launch_strided(qkv, (n * row, dim, row), qkv[:, nq:], (n * row, dim, row),
                         q_buf, (n * nq, dim, nq), k_buf, (n * nk, dim, nk), cos, sin, n,
                         plan.heads, plan.kv_heads, dim, backward=False)
    v = qkv.as_strided((1, n, plan.kv_heads, dim), (n * row, row, dim, 1),
                       qkv.storage_offset() + nq + nk)
    return q_buf.view(1, n, plan.heads, dim), k_buf.view(1, n, plan.kv_heads, dim), v


def _qkv_forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute, token_major=False):
    """_QKVRope's forward (also the first part of _AttnBlock's): (q4, k4, v4, the tensors to save);
    the rest of what the backward reads goes on `ctx`. `token_major` (_AttnBlock): q, k, v as the
    attention kernel takes them, (1, N, H, D) (_rope_token_major), instead of (1, H, N, D) views."""
    half, n, d = plan.half, plan.n_tokens, x.shape[-1]
    q_base, k_base, v_base = plan.bases
    nq, nk, nv = q_base.out_features, k_base.out_features, v_base.out_features
    r = a_q.shape[0]
    x16 = x.reshape(n, d)
    x16 = x16 if x16.dtype == half else x16.to(half)

    pair = _attn_weights(plan, x.device)  # q|k|v and o in one launch (o_proj reads it later)
    w = pair[0] if pair is not None else _weights(plan.bases, half, x.device)
    w_t = _t(w)
    qkv = torch.addmm(plan.bias, x16, w_t) if plan.bias is not None else torch.mm(x16, w_t)

    # the four adapter matrices in half: kept per layer and refreshed for every layer in one
    # multi-tensor launch after the optimizer step (else one launch per layer, as before)
    if _CPU["diet"]:
        a16, b16, derived = _ADAPTERS.halves(plan.lora_q[0], (a_q, a_v, b_q, b_v), half)
    else:
        a16 = torch.empty((2 * r, d), dtype=half, device=x.device)
        b16 = torch.empty((nq + nv, r), dtype=half, device=x.device)
        torch._foreach_copy_([a16[:r], a16[r:], b16[:nq], b16[nq:]], [a_q, a_v, b_q, b_v])
        derived = _adapter_views(a16, b16, r, nq)
    a3, a3_t, a_t, _, _, bq_t, bv_t = derived
    rng = None
    if plan.p > 0.0:
        rng = _get_rng_state(x.device)
        dropped, _ = torch.native_dropout(x16.expand(2, n, d), plan.p, True)
        h = torch.bmm(dropped, a3_t)  # (2, N, r)
        h_q, h_v = h[0], h[1]
        del dropped
    else:
        h = torch.mm(x16, a_t)  # (N, 2r)
        h_q, h_v = h[:, :r], h[:, r:]
    qkv[:, :nq].addmm_(h_q, bq_t, alpha=plan.scaling)
    qkv[:, nq + nk:].addmm_(h_v, bv_t, alpha=plan.scaling)

    # rotated q and k in their own token-major buffers: attention keeps q and k for the backward,
    # and a view of qkv would keep all of qkv alive with them; v leaves as a view of qkv and
    # attention saves a compact copy of it, so qkv dies after the attention forward
    cos, sin = _tables(cos, sin)
    if token_major:
        q4, k4, v4 = _rope_token_major(qkv, n, plan, cos, sin)
    elif _ATTN["views"]:  # the head-major views _heads made: transposes of the token-major ones
        q4, k4, v4 = (t.transpose(1, 2) for t in _rope_token_major(qkv, n, plan, cos, sin))
    else:
        q4 = _heads(torch.empty((n, nq), dtype=half, device=x.device), n, plan.heads, plan.head_dim)
        k4 = _heads(torch.empty((n, nk), dtype=half, device=x.device), n, plan.kv_heads,
                    plan.head_dim)
        rope._launch(_heads(qkv[:, :nq], n, plan.heads, plan.head_dim),
                     _heads(qkv[:, nq:nq + nk], n, plan.kv_heads, plan.head_dim), q4, k4, cos, sin,
                     backward=False)
        v4 = _heads(qkv[:, nq + nk:], n, plan.kv_heads, plan.head_dim)

    # x16 is x rounded to half, and x is the output of the input norm (rmsnorm.py), which keeps
    # its own input and rstd for its backward anyway: the backward rebuilds x16 from those, bit
    # for bit, instead of keeping it (N x D halves per layer)
    source = norm_source(x) if _ATTN["rebuild_x16"] and x.dtype != half else None
    keep = (h, a16, b16, cos, sin) if source is not None else (h, a16, b16, cos, sin, x16)
    ctx.source = source
    if recompute is not None:  # what the attention recompute starts from (kept here anyway)
        recompute.qkv_taken(source, x16, h, b16, cos, sin, derived)
    ctx.recompute, ctx.derived = recompute, derived
    ctx.plan, ctx.rng, ctx.x_shape, ctx.x_dtype = plan, rng, x.shape, x.dtype
    ctx.dims = (n, d, nq, nk, nv, r)
    return q4, k4, v4, keep


def _qkv_backward(ctx, saved, grad_q, grad_k, grad_v, token_major=False, box=None):
    """_QKVRope's backward (also the last part of _AttnBlock's and _QKVAttn's) from the tensors
    _qkv_forward saved: the gradients of x, A_q, B_q, A_v, B_v. `token_major` (_AttnBlock,
    _QKVAttn): the gradients of q, k, v as the attention backward gives them, (1, N, H, D), instead
    of (1, H, N, D) views of those. `box` (_QKVAttn): a list holding the only references to the
    three gradients, taken from it here, so they die once they are in `g`."""
    if box is not None:
        grad_q, grad_k, grad_v = box
        box.clear()
    h, a16, b16, cos, sin, *kept = saved
    plan = ctx.plan
    n, d, nq, nk, nv, r = ctx.dims
    half, s, p = plan.half, plan.scaling, plan.p
    recompute, ctx.recompute = ctx.recompute, None
    rebuilt = recompute.take_x16() if recompute is not None else None  # its rebuild, if it ran
    if kept:
        x16 = kept[0]
    else:
        x16 = (rebuilt if rebuilt is not None else rebuild(ctx.source, half)).view(n, d)
    if recompute is not None:
        recompute.release()
    device = x16.device

    g = torch.empty((n, nq + nk + nv), dtype=half, device=device)
    if token_major:  # rope._launch's arguments on the views below, without making them
        grad_q, grad_k = _last_dim_contiguous(grad_q), _last_dim_contiguous(grad_k)
        sq, sk, row, dim = grad_q.stride(), grad_k.stride(), g.stride(0), plan.head_dim
        rope._launch_strided(grad_q, (sq[0], sq[2], sq[1]), grad_k, (sk[0], sk[2], sk[1]),
                             g, (n * row, dim, row), g[:, nq:], (n * row, dim, row), cos, sin, n,
                             plan.heads, plan.kv_heads, dim, backward=True)
        g.as_strided((1, n, plan.kv_heads, dim), (n * row, row, dim, 1),
                     g.storage_offset() + nq + nk).copy_(grad_v)
    elif _ATTN["views"]:  # head-major gradients: their own strides, g's columns as _heads views them
        grad_q, grad_k = _last_dim_contiguous(grad_q), _last_dim_contiguous(grad_k)
        row, dim = g.stride(0), plan.head_dim
        rope._launch_strided(grad_q, grad_q.stride()[:3], grad_k, grad_k.stride()[:3],
                             g, (n * row, dim, row), g[:, nq:], (n * row, dim, row), cos, sin, n,
                             plan.heads, plan.kv_heads, dim, backward=True)
        g.as_strided((1, plan.kv_heads, n, dim), (n * row, dim, row, 1),
                     g.storage_offset() + nq + nk).copy_(grad_v)
    else:
        gq4 = _heads(g[:, :nq], n, plan.heads, plan.head_dim)
        gk4 = _heads(g[:, nq:nq + nk], n, plan.kv_heads, plan.head_dim)
        rope._launch(_last_dim_contiguous(grad_q), _last_dim_contiguous(grad_k), gq4, gk4, cos,
                     sin, backward=True)
        _heads(g[:, nq + nk:], n, plan.kv_heads, plan.head_dim).copy_(grad_v)
    del grad_q, grad_k, grad_v  # in g now (autograd still holds them in _QKVRope's backward)
    g_q, g_v = g[:, :nq], g[:, nq + nk:]

    need_x, need_a = ctx.needs_input_grad[0], ctx.needs_input_grad[1] or ctx.needs_input_grad[3]
    need_b = ctx.needs_input_grad[2] or ctx.needs_input_grad[4]
    # dL/dh = scaling * g @ B for both adapters; its dropout part keeps the masks' scale for the
    # addmm's beta at the end
    beta = 1.0
    a3, _, _, bq, bv, _, _ = ctx.derived  # views of a16 and b16 (saved: their versions checked)
    if p > 0.0:
        gh = torch.empty((2, n, r), dtype=half, device=device)
        _scaled_mm(g_q, bq, s, gh[0])
        _scaled_mm(g_v, bv, s, gh[1])
        current = _get_rng_state(device)
        _set_rng_state(ctx.rng, device)
        try:
            dropped, mask = torch.native_dropout(x16.expand(2, n, d), p, True)
        finally:
            _set_rng_state(current, device)
        grad_a = torch.bmm(gh.transpose(1, 2), dropped, out_dtype=torch.float32) if need_a else None
        del dropped
        if need_x:
            g_drop = torch.bmm(gh, a3)  # (2, N, D), half like the base's
            lora_x = _masked_pair_sum(g_drop, mask)
            beta = 1.0 / (1.0 - p)
            del g_drop
        grad_aq, grad_av = (grad_a[0], grad_a[1]) if need_a else (None, None)
        h_q, h_v = h[0], h[1]
    else:
        gh = torch.empty((n, 2 * r), dtype=half, device=device)
        _scaled_mm(g_q, bq, s, gh[:, :r])
        _scaled_mm(g_v, bv, s, gh[:, r:])
        grad_a = torch.mm(gh.t(), x16, out_dtype=torch.float32) if need_a else None
        if need_x:
            lora_x = torch.mm(gh, a16, out_dtype=torch.float32)
        grad_aq, grad_av = (grad_a[:r], grad_a[r:]) if need_a else (None, None)
        h_q, h_v = h[:, :r], h[:, r:]
    grad_bq = grad_bv = None
    if need_b:
        grad_bq = _scaled_mm(g_q.t(), h_q, s, torch.empty((nq, r), dtype=torch.float32,
                                                         device=device))
        grad_bv = _scaled_mm(g_v.t(), h_v, s, torch.empty((nv, r), dtype=torch.float32,
                                                         device=device))

    grad_x = None
    if need_x:
        pair = _attn_weights(plan, device)  # the recompute's or the o_proj backward's launch
        w = pair[0] if pair is not None else _weights(plan.bases, half, device)
        # one GEMM for the three projections, the adapters' share added as C (in place, fp32)
        grad_x = torch.addmm(lora_x, g, w, beta=beta, out_dtype=torch.float32, out=lora_x)
        grad_x = grad_x.view(ctx.x_shape)
        if ctx.x_dtype != torch.float32:
            grad_x = grad_x.to(ctx.x_dtype)
    grads = [grad_aq, grad_bq, grad_av, grad_bv]
    params = [m.weight for m in plan.lora_q + plan.lora_v]
    grads = [g_ if g_ is None or g_.dtype == p_.dtype else g_.to(p_.dtype)
             for g_, p_ in zip(grads, params)]
    return (grad_x, *grads)


# SOUP_SOLUTION_ATTN_ONE_FUNCTION=1: a recomputing layer's attention as one autograd node, _AttnBlock,
# instead of three (_QKVRope, unpad's variable-length attention, _Linear4bitHalf): two Python round
# trips into autograd less per layer and direction. Off by default: autograd keeps a node's incoming
# gradient until the node returns, so the o_proj output's gradient (N x D half) stays alive through
# the attention and q/k/v backward (T4, soup-c1-abba2: allocated peak 234.6 -> 238.8 MiB, 4.2 MiB =
# the 1,440-token step's N x 1536 halves, one 20 MiB segment of reserved more)
_ATTN["one_function"] = os.environ.get("SOUP_SOLUTION_ATTN_ONE_FUNCTION", "0").strip().lower() \
    not in ("0", "false", "no", "")
# SOUP_SOLUTION_FEWER_VIEWS=1 (task C1): rope launched from strides (rope._launch_strided, the same
# kernel arguments) and the recompute's q, k, v made in the kernel's token-major layout directly,
# without the head-major views that the next op turned back (~20 view ops less per layer and step)
_ATTN["views"] = os.environ.get("SOUP_SOLUTION_FEWER_VIEWS", "1").strip().lower() \
    not in ("0", "false", "no", "")


def _one_function_ok(plan, dropout):
    """Whether unpad.attention would take its variable-length kernel path for this plan's q, k, v
    (unpad._varlen_ok on the shapes and dtype _QKVRope gives them)."""
    from .unpad import _STATE as UNPAD

    if not UNPAD["varlen"] or dropout != 0.0 or torch.compiler.is_compiling():
        return False
    if plan.kv_heads == 0 or plan.heads % plan.kv_heads or plan.head_dim % 8 or plan.head_dim > 128:
        return False
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else plan.half
    if dtype == torch.bfloat16:
        if torch.cuda.get_device_capability(plan.bases[0].weight.device) < (8, 0):
            return False
    elif dtype != torch.float16:
        return False
    return dtype == plan.half and torch.backends.cuda.mem_efficient_sdp_enabled()


class _AttnBlock(torch.autograd.Function):
    """x (1, N, D) -> o_proj(attention(q, k, v)) (N, D) of a layer whose attention block is recomputed
    in its backward: _QKVRope, unpad's _VarlenAttention and _Linear4bitHalf as one node, the same
    operations on the same operands in the same order each way. Only for a recomputing layer: its
    attention keeps nothing but cu_seqlens, so holding the node's saved tensors to its end keeps
    nothing longer than the three nodes did."""

    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute, scale, o_bias):
        q, k, v, keep = _qkv_forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute,
                                     token_major=True)
        # unpad.attention and _VarlenAttention.forward (q, k, v as unpad.kernel_operands makes them)
        bounds = plan.bounds
        max_len = max(end - start for start, end in zip(bounds[:-1], bounds[1:]))
        cu = _cu_seqlens(bounds, q.device)
        groups = q.shape[2] // k.shape[2]
        out, lse, seed, offset = varlen_forward(q, k, v, cu, max_len, scale)
        recompute.attention_taken(cu, max_len, scale, q, k, v, out, lse)
        del q, k, v, lse
        ctx.attn_args = (seed, offset, max_len, scale, groups)
        ctx.qkv_count = len(keep)
        ctx.save_for_backward(*keep, cu)
        # _Linear4bitHalf.forward
        o = out.reshape(plan.n_tokens, plan.heads * plan.head_dim)
        del out
        pair = _attn_weights(plan, o.device) if o.dtype == plan.half else None
        w = pair[1] if pair is not None else _weights((plan.o_base,), o.dtype, o.device)
        w_t = _t(w)
        return torch.addmm(o_bias, o, w_t) if o_bias is not None else torch.mm(o, w_t)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        plan = ctx.plan
        n = plan.n_tokens
        # _Linear4bitHalf.backward
        half = plan.o_base.compute_dtype
        grad = grad if grad.dtype == half else grad.to(half)
        pair = _attn_weights(plan, grad.device) if half == plan.half else None
        w = pair[1] if pair is not None else _weights((plan.o_base,), half, grad.device)
        grad_out = torch.mm(grad, w).view(1, n, plan.heads, plan.head_dim)
        del grad, w, pair
        # _VarlenAttention.backward (the reshape's backward between them is this view)
        seed, offset, max_len, scale, groups = ctx.attn_args
        cu = saved[-1]
        q, k, v, out, lse, seed, offset = ctx.recompute.attention_tensors()
        grad_q, grad_k, grad_v, _ = _efficient_backward(
            grad_out, q, _expand_heads(k, groups), _expand_heads(v, groups), out, cu, max_len, lse,
            seed, offset, scale)
        if groups > 1:
            grad_k = grad_k.unflatten(2, (k.shape[2], groups)).sum(3)
            grad_v = grad_v.unflatten(2, (v.shape[2], groups)).sum(3)
        del grad_out, q, k, v, out, lse
        # _QKVRope.backward (it took the transposes of these: the gradients of its q4, k4, v4)
        grads = _qkv_backward(ctx, saved[:ctx.qkv_count], grad_q, grad_k, grad_v, token_major=True)
        return (*grads, None, None, None, None, None, None)


# SOUP_SOLUTION_ATTN_TWO_NODES=1 (task C1): a recomputing layer's _QKVRope and variable-length
# attention as one autograd node (_QKVAttn), the o_proj its own node (_Linear4bitHalf) as before. The
# o_proj backward still consumes the output's gradient before the attention backward runs; the node
# holds its incoming gradient (the attention output's, N x D half) through the q/k/v backward in place
# of the q/k/v gradients (N x (nq + nk + nv) half) that _QKVRope's node held there. One autograd node
# and the reshape's view node less per layer and direction. SOUP_SOLUTION_ATTN_ONE_FUNCTION=1 wins
_ATTN["two_nodes"] = os.environ.get("SOUP_SOLUTION_ATTN_TWO_NODES", "1").strip().lower() \
    not in ("0", "false", "no", "")


class _QKVAttn(torch.autograd.Function):
    """x (1, N, D) -> attention(q, k, v) as (N, Hq * Dh) of a layer whose attention block is
    recomputed in its backward: _QKVRope and unpad's _VarlenAttention as one node (the reshape to the
    o_proj's input is a view of its output), the same operations on the same operands in the same
    order each way (the q, k, v and their gradients in the kernel's layout, as _AttnBlock has them)."""

    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute, scale):
        q, k, v, keep = _qkv_forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan, recompute,
                                     token_major=True)
        # unpad.attention and _VarlenAttention.forward (q, k, v as unpad.kernel_operands makes them)
        bounds = plan.bounds
        max_len = max(end - start for start, end in zip(bounds[:-1], bounds[1:]))
        cu = _cu_seqlens(bounds, q.device)
        groups = q.shape[2] // k.shape[2]
        out, lse, seed, offset = varlen_forward(q, k, v, cu, max_len, scale)
        recompute.attention_taken(cu, max_len, scale, q, k, v, out, lse)
        del q, k, v, lse
        ctx.attn_args = (max_len, scale, groups)
        ctx.qkv_count = len(keep)
        ctx.save_for_backward(*keep, cu)
        return out.view(plan.n_tokens, plan.heads * plan.head_dim)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        plan = ctx.plan
        # _VarlenAttention.backward (the reshape's backward before it is this reshape)
        max_len, scale, groups = ctx.attn_args
        cu = saved[-1]
        q, k, v, out, lse, seed, offset = ctx.recompute.attention_tensors()
        grad_out = grad.reshape(1, plan.n_tokens, plan.heads, plan.head_dim)
        grad_out = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        grad_q, grad_k, grad_v, _ = _efficient_backward(
            grad_out, q, _expand_heads(k, groups), _expand_heads(v, groups), out, cu, max_len, lse,
            seed, offset, scale)
        del grad_out, q, out, lse
        if groups > 1:
            grad_k = grad_k.unflatten(2, (k.shape[2], groups)).sum(3)
            grad_v = grad_v.unflatten(2, (v.shape[2], groups)).sum(3)
        del k, v
        # _QKVRope.backward on the gradients of its q4, k4, v4 (whose transposes it took)
        box = [grad_q, grad_k, grad_v]
        del grad_q, grad_k, grad_v
        grads = _qkv_backward(ctx, saved[:ctx.qkv_count], None, None, None, token_major=True,
                              box=box)
        return (*grads, None, None, None, None, None)


def _recompute_mode():
    value = os.environ.get("SOUP_SOLUTION_RECOMPUTE_ATTN", "1").strip().lower()
    if value in ("1", "true", "yes", "on", "all"):
        return "all"
    return "auto" if value == "auto" else None


# SOUP_SOLUTION_RECOMPUTE_ATTN (see the module doc); `check` (tests): the recompute compares what it
# rebuilds with clones of the forward's tensors and appends any difference to `mismatches`
_RECOMPUTE = {"mode": _recompute_mode(),
              "tokens": int(os.environ.get("SOUP_SOLUTION_RECOMPUTE_TOKENS", "640") or "640"),
              "check": False, "mismatches": []}


def _kept_bytes(plan, d):
    """Bytes per token a fused attention layer keeps for its backward: (as before, with rmsnorm.py's
    split sum, with the recompute). Common to all three: the layer's fp32 input (the input norm
    keeps it) and h; then the post-attention norm's fp32 sum or its half branch, and the
    attention's q, compact k and v, output and log-sum-exp."""
    nq, nk = plan.bases[0].out_features, plan.bases[1].out_features
    r = plan.lora_q[0].weight.shape[0]
    attention = 2 * nq + 4 * nk + 2 * nq + 4 * plan.heads
    common = 4 * d + 4 * r
    return common + 4 * d + attention, common + 2 * d + attention, common


def recompute_layers(plan, d, layers):
    """How many layers, from layer 0, recompute their attention block in this step: none, all, or
    (auto) the fewest that keep the step's attention memory at what a step of
    SOUP_SOLUTION_RECOMPUTE_TOKENS tokens keeps without the recompute."""
    mode = _RECOMPUTE["mode"]
    if mode is None:
        return 0
    if mode == "all":
        return layers
    t0, n = _RECOMPUTE["tokens"], plan.n_tokens
    if n <= t0:
        return 0
    from .rmsnorm import _SPLIT

    plain, split, rebuilt = _kept_bytes(plan, d)
    base = split if _SPLIT["keep"] else plain
    return min(layers, math.ceil(layers * (n - t0) / (n * (1.0 - rebuilt / base))))


def _recompute_for(attn, plan, x):
    """An _AttnRecompute for this layer when it recomputes in this step, else None."""
    if _RECOMPUTE["mode"] is None or not torch.is_grad_enabled():
        return None
    index = getattr(attn, "layer_idx", None)
    layers = getattr(getattr(attn, "config", None), "num_hidden_layers", None)
    if not isinstance(index, int) or not isinstance(layers, int):
        return None
    return _AttnRecompute(plan) if index < recompute_layers(plan, x.shape[-1], layers) else None


class _AttnRecompute:
    """One layer's attention block rebuilt in its backward (see the module doc). The forward hands
    it what it starts from (_QKVRope: x16's source, h, the half adapters, the rotary tables; the
    variable-length attention: cu_seqlens, max length, scale; then the o_proj output it stands for).
    In the backward, the post-attention norm calls `branch()` (the o_proj output) and
    `branch_done()`, the attention `attention_tensors()`, _QKVRope `take_x16()` and `release()`;
    whichever comes first runs the rebuild (`_rebuild`). Holds the rebuilt tensors only from then
    until their last user."""

    def __init__(self, plan):
        self.plan = plan
        self.source = self.x16 = self.h = self.b16 = self.cos = self.sin = self.derived = None
        self.cu = self.max_len = self.scale = None
        self.armed = False  # the variable-length attention took it: it keeps nothing of its own
        self.out_key = None
        self.tensors = self.branch_out = None
        self.check = {} if _RECOMPUTE["check"] else None

    # forward
    def qkv_taken(self, source, x16, h, b16, cos, sin, derived=None):
        self.source, self.h, self.b16, self.cos, self.sin = source, h, b16, cos, sin
        self.derived = derived
        self.x16 = x16 if source is None else None  # without a source, _QKVRope keeps x16 itself
        if self.check is not None:
            self.check["x16"] = x16.clone()

    def attention_taken(self, cu, max_len, scale, q, k, v, out, lse):
        self.cu, self.max_len, self.scale, self.armed = cu, max_len, scale, True
        if self.check is not None:
            self.check.update(q=_compact(q).clone(), k=_compact(k).clone(), v=_compact(v).clone(),
                              out=out.clone(), lse=lse.clone())

    def output_taken(self, out):
        self.out_key = (out.data_ptr(), tuple(out.shape), out.dtype)
        if self.check is not None:
            self.check["branch"] = out.detach().reshape(-1, out.shape[-1]).clone()

    def accepts(self, branch):
        return self.armed and self.out_key == (branch.data_ptr(), tuple(branch.shape), branch.dtype)

    # backward
    def _rebuild(self, with_branch):
        plan = self.plan
        half, n = plan.half, plan.n_tokens
        q_base, k_base, _ = plan.bases
        nq, nk = q_base.out_features, k_base.out_features
        x16 = self.x16 if self.x16 is not None else rebuild(self.source, half)
        x16 = x16.view(n, -1)
        pair = _attn_weights(plan, x16.device)  # read again by the o_proj and q/k/v backward
        w = pair[0] if pair is not None else _weights(plan.bases, half, x16.device)
        w_t = _t(w)
        qkv = torch.addmm(plan.bias, x16, w_t) if plan.bias is not None else torch.mm(x16, w_t)
        h, b16, r = self.h, self.b16, plan.lora_q[0].weight.shape[0]
        h_q, h_v = (h[0], h[1]) if h.dim() == 3 else (h[:, :r], h[:, r:])
        bq_t, bv_t = (self.derived[5], self.derived[6]) if self.derived is not None else (
            b16[:nq].t(), b16[nq:].t())
        qkv[:, :nq].addmm_(h_q, bq_t, alpha=plan.scaling)
        qkv[:, nq + nk:].addmm_(h_v, bv_t, alpha=plan.scaling)
        if _ATTN["views"]:  # the same rope launch and kernel operands, fewer views
            q, k, v = _rope_token_major(qkv, n, plan, self.cos, self.sin)
        else:
            q4 = _heads(torch.empty((n, nq), dtype=half, device=x16.device), n, plan.heads,
                        plan.head_dim)
            k4 = _heads(torch.empty((n, nk), dtype=half, device=x16.device), n, plan.kv_heads,
                        plan.head_dim)
            rope._launch(_heads(qkv[:, :nq], n, plan.heads, plan.head_dim),
                         _heads(qkv[:, nq:nq + nk], n, plan.kv_heads, plan.head_dim), q4, k4,
                         self.cos, self.sin, backward=False)
            v4 = _heads(qkv[:, nq + nk:], n, plan.kv_heads, plan.head_dim)
            q, k, v = kernel_operands(q4, k4, v4, half)
        out, lse, seed, offset = varlen_forward(q, k, v, self.cu, self.max_len, self.scale)
        self.tensors = (_compact(q), _compact(k), _compact(v), out, lse, seed, offset)
        self.x16 = x16
        if with_branch:
            o = out.reshape(n, plan.heads * plan.head_dim)
            w = pair[1] if pair is not None else _weights((plan.o_base,), half, x16.device)
            bias = _cached_bias(plan.o_base, (plan.o_base,), half)
            w_t = _t(w)
            self.branch_out = torch.addmm(bias, o, w_t) if bias is not None else torch.mm(o, w_t)
        if self.check is not None:
            got = dict(zip(("q", "k", "v", "out", "lse"), self.tensors[:5]), x16=x16,
                       branch=self.branch_out)
            for name, value in got.items():
                want = self.check.get(name)
                if value is None or want is None:
                    continue
                if name == "lse" and value.dim() == 3 and value.shape[0] == len(plan.bounds) - 1:
                    # (sequences, heads, aligned length): the entries past a sequence are unwritten
                    same = all(torch.equal(value[i, :, :end - start], want[i, :, :end - start])
                               for i, (start, end) in enumerate(zip(plan.bounds[:-1], plan.bounds[1:])))
                elif name == "lse" and value.dim() == 3:  # one row of all tokens, aligned
                    same = torch.equal(value[..., :plan.n_tokens], want[..., :plan.n_tokens])
                else:
                    same = torch.equal(value, want.view_as(value))
                if not same:
                    _RECOMPUTE["mismatches"].append(name)

    def branch(self):
        if self.branch_out is None:
            self._rebuild(with_branch=True)
        return self.branch_out

    def branch_done(self):
        self.branch_out = None

    def attention_tensors(self):
        if self.tensors is None:
            self._rebuild(with_branch=False)
        tensors, self.tensors = self.tensors, None
        return tensors

    def take_x16(self):
        x16, self.x16 = self.x16, None
        return x16

    def release(self):
        self.source = self.x16 = self.h = self.b16 = self.cos = self.sin = self.cu = None
        self.derived = None
        self.tensors = self.branch_out = self.check = None


class _Linear4bitHalf(torch.autograd.Function):
    """x16 (M, K) @ W.T (+ bias) for a frozen 4-bit layer, W rebuilt in the shared buffer each way.
    `plan`: the attention plan whose o_proj this is: W from _attn_weights (the forward's q|k|v launch
    made it already; in the backward, the recompute's launch)."""

    @staticmethod
    def forward(ctx, x16, base, bias, plan=None):
        pair = _attn_weights(plan, x16.device) if plan is not None and x16.dtype == plan.half else None
        w = pair[1] if pair is not None else _weights((base,), x16.dtype, x16.device)
        w_t = _t(w)
        out = torch.addmm(bias, x16, w_t) if bias is not None else torch.mm(x16, w_t)
        ctx.base, ctx.plan = base, plan
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        half, plan = ctx.base.compute_dtype, ctx.plan
        grad = grad if grad.dtype == half else grad.to(half)
        pair = _attn_weights(plan, grad.device) if plan is not None and half == plan.half else None
        w = pair[1] if pair is not None else _weights((ctx.base,), half, grad.device)
        return torch.mm(grad, w), None, None, None


def _fused_attention(attn, x, position_embeddings, plan):
    a_q, b_q = plan.lora_q
    a_v, b_v = plan.lora_v
    cos, sin = position_embeddings
    recompute = _recompute_for(attn, plan, x)
    dropout = 0.0 if not attn.training else attn.attention_dropout
    n = plan.n_tokens
    bias = _cached_bias(plan.o_base, (plan.o_base,), plan.half)
    merged = recompute is not None and (_ATTN["one_function"] or _ATTN["two_nodes"]) \
        and _one_function_ok(plan, dropout)
    if merged and _ATTN["one_function"]:
        out = _AttnBlock.apply(x, a_q.weight, b_q.weight, a_v.weight, b_v.weight, cos, sin, plan,
                               recompute, attn.scaling, bias)
    elif merged:
        out = _QKVAttn.apply(x, a_q.weight, b_q.weight, a_v.weight, b_v.weight, cos, sin, plan,
                             recompute, attn.scaling)  # (N, Hq * Dh)
        out = _Linear4bitHalf.apply(out, plan.o_base, bias, plan)
    else:
        query, key, value = _QKVRope.apply(x, a_q.weight, b_q.weight, a_v.weight, b_v.weight, cos,
                                           sin, plan, recompute)
        out = packed_attention(query, key, value, plan.bounds, attn.scaling, dropout,
                               recompute)  # (1, N, H, Dh)
        out = out.reshape(n, plan.heads * plan.head_dim)
        out = _Linear4bitHalf.apply(out, plan.o_base, bias, plan)
    out = out.view(1, n, -1)
    if recompute is not None and recompute.armed:
        # the post-attention norm (rmsnorm.post_attention_addends) finds it here and keeps the
        # residual only; its backward gets this output again from the recompute
        recompute.output_taken(out)
        out._soup_recompute = recompute
    return out


# ---------------------------------------------------------------------------------------------
# MLP


_MLP_MEMORY = {"recompute": False,
               "inplace": os.environ.get("SOUP_SOLUTION_MLP_BWD_INPLACE", "1").strip().lower()
               not in ("0", "false", "no", ""),
               # SOUP_SOLUTION_MLP_ADDMM_INPLACE=1 (task C1): the input gradient's second GEMM adds
               # into the first one's fp32 result in place; out of place, torch's addmm copies that
               # result into a new tensor first and runs the same GEMM on the copy (one D2D copy
               # and one N x D fp32 buffer per layer more)
               "addmm_inplace": os.environ.get("SOUP_SOLUTION_MLP_ADDMM_INPLACE", "1").strip().lower()
               not in ("0", "false", "no", "")}


def _accumulate(result, a, b):
    """result + a @ b in fp32 (result fp32, a and b half), into result when that is on."""
    if _MLP_MEMORY["addmm_inplace"]:
        return torch.addmm(result, a, b, out_dtype=torch.float32, out=result)
    return torch.addmm(result, a, b, out_dtype=torch.float32)


def set_mlp_recompute(on):
    """What the fused MLP keeps for its backward: (gate|up) itself (False, the default: offload.py's
    saved-tensor hooks move it to host memory, or it stays on the GPU), or its half input x16
    (N x hidden) with (gate|up) recomputed by the same GEMM in the backward (True: no activation
    crosses PCIe, one GEMM more per layer). offload.patch_mlp_memory sets it from
    SOUP_SOLUTION_MLP_MEMORY."""
    _MLP_MEMORY["recompute"] = bool(on)


class _MLPPlan:
    __slots__ = ("gate", "up", "down", "half", "split", "recompute", "early", "ring")


def _mlp_plan(mlp, x):
    if not _STATE["enabled"]:
        return None
    act = getattr(getattr(mlp, "config", None), "hidden_act", "silu")
    if (act != "silu" or not torch.is_tensor(x) or not x.is_cuda or x.dtype not in _INPUTS
            or x.dim() < 2 or x.numel() == 0 or torch.compiler.is_compiling() or _global_hooks()):
        return None
    gate = _plain_4bit(getattr(mlp, "gate_proj", None))
    up = _plain_4bit(getattr(mlp, "up_proj", None))
    down = _plain_4bit(getattr(mlp, "down_proj", None))
    if gate is None or up is None or down is None:
        return None
    half = gate.compute_dtype
    if up.compute_dtype != half or down.compute_dtype != half:
        return None
    if torch.is_autocast_enabled("cuda") and torch.get_autocast_dtype("cuda") != half:
        return None
    hidden, inter = x.shape[-1], gate.out_features
    if (gate.in_features != hidden or up.in_features != hidden or up.out_features != inter
            or down.in_features != inter or down.out_features != hidden):
        return None
    if any(b.bias is not None for b in (gate, up, down)):
        return None
    if any(_flags(m.weight)[2] != x.device for m in (gate, up, down)):
        return None
    m = x.numel() // hidden
    if any(_bnb_would_fuse(m, b, half, x.device) for b in (gate, up, down)):
        return None
    ring = weight_stream.mlp_ring(gate, up, down)
    if ring is False:  # only some of the three in weight_stream.py's type ring
        return None
    plan = _MLPPlan()
    plan.gate, plan.up, plan.down, plan.half, plan.split = gate, up, down, half, _gate_up_split()
    plan.recompute = _MLP_MEMORY["recompute"]
    plan.early = False  # set by the patched forward (offload.EARLY)
    plan.ring = ring  # weight_stream.py's type ring holding the three matrices, or None
    return plan


def _ring_weights(plan, bases, device, step):
    """_weights of MLP matrices of `plan`. With weight_stream.py's type ring (SOUP_SOLUTION_WEIGHT_RING
    =type: one GPU slot per matrix type, shared by all layers) the compute stream first waits for
    each matrix's copy into its slot (acquire), and once the dequantization is queued the copy of
    the same matrices of the next layer in the pass (step +1: forward, -1: backward) starts
    (release): the GEMMs read the dequantized buffer, not the slot."""
    ring = plan.ring
    if ring is None:
        return _weights(bases, plan.half, device)
    for base in bases:
        ring.acquire(base, step)
    w = _weights(bases, plan.half, device)
    for base in bases:
        ring.release(base, step)
    return w


def _gate_up_matmul(plan, left, right_t, out):
    """out (2I, ·) = [W_gate; W_up] @ right_t (`left` False) or out (·, K) = left @ [W_gate; W_up]
    (`right_t` None), with the two weights in one buffer or one after the other (split)."""
    inter = plan.gate.out_features
    step = 1 if right_t is not None else -1  # the forward's product, or the backward's
    if not plan.split:
        w = _ring_weights(plan, (plan.gate, plan.up), out.device, step)
        if right_t is not None:
            return torch.mm(w, right_t, out=out)
        return torch.mm(left, w, out_dtype=torch.float32)
    w = _ring_weights(plan, (plan.gate,), out.device, step)
    if right_t is not None:
        torch.mm(w, right_t, out=out[:inter])
        w = _ring_weights(plan, (plan.up,), out.device, step)
        return torch.mm(w, right_t, out=out[inter:])
    result = torch.mm(left[:, :inter], w, out_dtype=torch.float32)
    w = _ring_weights(plan, (plan.up,), out.device, step)
    return _accumulate(result, left[:, inter:], w)


def _swiglu_bwd(grad_hidden, gate_up, grad_gate_up, inter, padded):
    """d(gate|up) of silu(gate) * up into grad_gate_up, which may be gate_up itself: every element is
    loaded before the values computed from it are stored, and no two programs share an element."""
    count = inter * padded
    example_swiglu_bwd_kernel[(triton.cdiv(count, SWIGLU_BLOCK),)](
        grad_hidden, gate_up[:inter], gate_up[inter:], grad_gate_up[:inter], grad_gate_up[inter:],
        count, BLOCK=SWIGLU_BLOCK)


def _recompute_backward(plan, x16, grad_hidden, padded):
    """grad_x (padded, D) in fp32 from the saved x16: (gate|up) recomputed by the forward's GEMMs on
    the same operands (the same values), the SwiGLU's gradient written over it in place, and its
    product with the gate|up weights. One rebuild of the weights serves both GEMMs: up is recomputed
    first with split, so that gate's weights are still in the buffer for the first gradient GEMM."""
    half, inter = plan.half, plan.gate.out_features
    gate_up = torch.empty((2 * inter, padded), dtype=half, device=x16.device)
    x_t = x16.t()
    if not plan.split:
        w = _ring_weights(plan, (plan.gate, plan.up), x16.device, -1)
        torch.mm(w, x_t, out=gate_up)
        _swiglu_bwd(grad_hidden, gate_up, gate_up, inter, padded)
        return torch.mm(gate_up.t(), w, out_dtype=torch.float32)
    ring = plan.ring
    if ring is not None:  # up is dequantized twice here: it keeps its slot until the second time
        ring.acquire(plan.up, -1)
    w = _weights((plan.up,), half, x16.device)
    torch.mm(w, x_t, out=gate_up[inter:])
    w = _ring_weights(plan, (plan.gate,), x16.device, -1)
    torch.mm(w, x_t, out=gate_up[:inter])
    _swiglu_bwd(grad_hidden, gate_up, gate_up, inter, padded)
    result = torch.mm(gate_up[:inter].t(), w, out_dtype=torch.float32)
    w = _weights((plan.up,), half, x16.device)
    if ring is not None:
        ring.release(plan.up, -1)
    return _accumulate(result, gate_up[inter:].t(), w)


def _padded_half(t, n, padded, half):
    """t (n, D) in `half` with zero rows appended up to `padded` rows (t itself, cast if needed, when
    n == padded). Every GEMM of the MLP then runs on a multiple of 8 tokens: with n % 8 != 0 cuBLAS
    takes cutlass align1/align2 kernels on the T4, 2-3x slower than its aligned ones."""
    if n == padded:
        return t if t.dtype == half else t.to(half)
    out = torch.empty((padded, t.shape[1]), dtype=half, device=t.device)
    out[:n].copy_(t)
    out[n:].zero_()
    return out


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        half = plan.half
        d, inter = x.shape[-1], plan.gate.out_features
        n = x.numel() // d
        padded = -(-n // 8) * 8
        # zero padding rows: their gate|up, SwiGLU and output rows are zeros, cut off below
        x16 = _padded_half(x.reshape(n, d), n, padded, half)
        gate_up = torch.empty((2 * inter, padded), dtype=half, device=x.device)
        _gate_up_matmul(plan, None, x16.t(), gate_up)
        if plan.early:
            from .offload import offload_early

            offload_early(gate_up)  # its copy to the host runs under SwiGLU and the down GEMM
        hidden = torch.empty((inter, padded), dtype=half, device=x.device)
        count = inter * padded
        example_swiglu_fwd_kernel[(triton.cdiv(count, SWIGLU_BLOCK),)](
            gate_up[:inter], gate_up[inter:], hidden, count, BLOCK=SWIGLU_BLOCK)
        w = _ring_weights(plan, (plan.down,), x.device, 1)
        out = torch.mm(hidden.t(), _t(w))  # (padded, D)
        # recompute: x16 (N x 1536 half) instead of (gate|up) (2 x 8960 x N half), 12x less, or not
        # even x16 when it is rebuilt from the post-attention norm's saved tensors (rmsnorm.rebuild)
        ctx.source = (norm_source(x) if plan.recompute and n == padded and _ATTN["rebuild_x16"]
                      else None)
        ctx.save_for_backward(*(() if ctx.source is not None else (x16 if plan.recompute else gate_up,)))
        ctx.plan, ctx.x_shape, ctx.x_dtype, ctx.n, ctx.padded = plan, x.shape, x.dtype, n, padded
        return out[:n].view(*x.shape[:-1], d)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors[0] if ctx.source is None else rebuild(ctx.source, ctx.plan.half)
        plan, n, padded = ctx.plan, ctx.n, ctx.padded
        half = plan.half
        inter = plan.gate.out_features
        grad16 = _padded_half(grad.reshape(n, grad.shape[-1]), n, padded, half)
        if not plan.recompute and int8_bwd.usable(plan, padded):
            # SOUP_SOLUTION_INT8_BWD=1: both input-gradient GEMMs on int8 tensor cores, the int8
            # weights in the shared buffer, (gate|up) overwritten with its gradient
            buffer = _buffer(grad.device, half, plan.down.out_features * plan.down.in_features)
            matrices = (plan.down, plan.gate, plan.up) if plan.ring is not None else ()
            for base in matrices:  # it reads the three (scales, int8 decode) in one go
                plan.ring.acquire(base, -1)
            grad_x = int8_bwd.mlp_backward(plan, saved, grad16, buffer)[:n]
            for base in matrices:
                plan.ring.release(base, -1)
            if ctx.x_dtype != torch.float32:
                grad_x = grad_x.to(ctx.x_dtype)
            return grad_x.view(ctx.x_shape), None
        w = _ring_weights(plan, (plan.down,), grad.device, -1)
        grad_hidden = torch.mm(_t(w), grad16.t())  # (I, padded)
        if plan.recompute:
            grad_x = _recompute_backward(plan, saved, grad_hidden, padded)
        else:
            gate_up = saved
            # in place (SOUP_SOLUTION_MLP_BWD_INPLACE=1): the saved (gate|up) has no other reader
            # after this backward (offload.py's unpack hands over a fresh copy and drops its own
            # reference), and _swiglu_bwd loads every element before storing what it computes
            # from it: one (2I, N) half buffer less at the start of the backward (36 KB per token)
            grad_gate_up = gate_up if _MLP_MEMORY["inplace"] else torch.empty_like(gate_up)
            _swiglu_bwd(grad_hidden, gate_up, grad_gate_up, inter, padded)
            del grad_hidden
            grad_x = _gate_up_matmul(plan, grad_gate_up.t(), None, grad_gate_up)
        grad_x = grad_x[:n]
        if ctx.x_dtype != torch.float32:
            grad_x = grad_x.to(ctx.x_dtype)
        return grad_x.view(ctx.x_shape), None


# ---------------------------------------------------------------------------------------------
# patches


def patch_qwen2_attention_fused():
    """Qwen2Attention.forward: inside `unpad.varlen`, the fused half path above; anything else goes
    to the forward that was in place when this patch ran."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2Attention
    _register_warmups()
    if getattr(cls, "_soup_fused_proj", False):
        return
    previous = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        plan = _attention_plan(self, hidden_states, position_embeddings, attention_mask,
                               past_key_values)
        if plan is None:
            return previous(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kwargs)
        return _fused_attention(self, hidden_states, position_embeddings, plan), None

    if "_soup_original_forward" not in vars(cls):
        cls._soup_original_forward = previous
    cls._soup_fused_proj_previous = previous
    cls.forward = forward
    cls._soup_fused_proj = True


def _register_warmups():
    """The attention's kernels outside the shared modules (rmsnorm.rms_norm_rows_rebuild for the x16
    rebuild, masked_pair_sum_kernel) into warmup.py's registry; by name, so registering again
    replaces the entries."""
    from .warmup import register

    if _ATTN["rebuild_x16"]:
        from .rmsnorm import warm_rebuild

        register(warm_rebuild, name="fused_proj_x16_rebuild")
    register(_warm_masked_pair_sum, name="fused_proj_masked_pair_sum")
    if _CPU["diet"]:
        if _CPU["reuse"]:
            register(_warm_dequant4, name="fused_proj_nf4_dequant4")
        else:
            register(_warm_dequant3, name="fused_proj_nf4_dequant3")


def patch_qwen2_mlp_fused():
    """Qwen2MLP.forward: gate+up in one GEMM and the SwiGLU on contiguous halves; anything else goes
    to the forward that was in place when this patch ran (mlp16's, when it is on)."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2MLP
    if getattr(cls, "_soup_fused_proj", False):
        return
    previous = cls.forward

    def forward(self, x):
        plan = _mlp_plan(self, x)
        if plan is None:
            weight_stream.mlp_unmanaged(self)  # a type ring loads this layer's matrices for it
            return previous(self, x)
        from .offload import DRAIN_LAST, EARLY, drain_inflight

        if torch.is_grad_enabled():
            plan.early = EARLY and not plan.recompute and x.requires_grad
            if plan.early or (DRAIN_LAST and self is _CTX["last_mlp"]):
                drain_inflight()  # the previous layer's (gate|up) leaves before this one is made
        return _MLP.apply(x, plan)

    forward._soup_weight_ring = True  # weight_stream.py: the matrices' reads go through acquire()
    if "_soup_original_forward" not in vars(cls):
        cls._soup_original_forward = previous
    cls._soup_fused_proj_previous = previous
    cls.forward = forward
    cls._soup_fused_proj = True


def unpatch_fused_proj():
    """Puts back the forwards these patches replaced (for tests)."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    for cls in (mq.Qwen2Attention, mq.Qwen2MLP):
        if getattr(cls, "_soup_fused_proj", False):
            cls.forward = cls._soup_fused_proj_previous
            del cls._soup_fused_proj_previous
            cls._soup_fused_proj = False
