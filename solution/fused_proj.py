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
- q, k, v leave the function as (B, H, N, D) token-major views; repeat_kv and per-row causal SDPA run as
  in unpad.py, the per-row outputs are concatenated token-major (one copy) and o_proj is dequant + GEMM.
Only x16, h (N x 32), the half adapter weights and the CUDA RNG state are kept for the backward; the
dropout masks are drawn again from that state in the backward (the same kernel on the same input
gives the same masks), so no mask and no fp32 q/k/v is stored. The backward rebuilds the weights,
writes dq and dk through the rope kernel and dv straight into one (N, 2048) gradient, computes the
adapters' share of the input gradient in half (one bmm, then the two masks applied by one mul and
one addcmul into fp32), and adds the three projections' share to it with one in-place addmm in fp32
(`out_dtype=torch.float32`, the masks' 1 / (1 - p) as the addmm's beta).

MLP: x16 once, gate and up rebuilt into one (17920, 1536) buffer and one GEMM computes (gate|up)
transposed, (17920, N) with the token axis padded to a multiple of 8, so that gate and up are two
contiguous row blocks for the kit's example SwiGLU kernel (it needs contiguous inputs) without any
copy; down_proj reads the transposed product directly. Only (gate|up) in half is kept for the
backward, as in mlp16.py. The MLP returns half: the base's MLP output is the fp16 GEMM result cast
to fp32, and the residual add promotes half to fp32 with the same values.

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

import os

import torch
import torch.nn.functional as F
import triton

from . import rope
from .nf4 import dequantize_nf4
from .swiglu import BLOCK as SWIGLU_BLOCK
from .swiglu import example_swiglu_bwd_kernel, example_swiglu_fwd_kernel
from .unpad import _CTX

_STATE = {"enabled": True}
_HALVES = (torch.float16, torch.bfloat16)
_INPUTS = (torch.float32, torch.float16, torch.bfloat16)
_BUFFERS = {}  # (device, dtype) -> flat buffer that holds the dequantized weights of one GEMM


def _gate_up_split():
    """SOUP_SOLUTION_FUSED_GU_SPLIT=1: gate and up as two GEMMs over one half-size weight buffer
    (27.5 MB less of transient weight memory, one more launch each way)."""
    return os.environ.get("SOUP_SOLUTION_FUSED_GU_SPLIT", "0") not in ("0", "false", "no")


# ---------------------------------------------------------------------------------------------
# weights


def _buffer(device, dtype, numel):
    key = (device, dtype)
    buf = _BUFFERS.get(key)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(numel, dtype=dtype, device=device)
        _BUFFERS[key] = buf
    return buf


def _weights(bases, dtype, device, offset=0):
    """The dequantized weights of `bases` stacked by rows, (sum of out, in), a view of the shared
    buffer starting at element `offset`. Valid until the next call on this device: every GEMM that
    reads it is queued on the same stream before the next dequantization overwrites it."""
    k = bases[0].in_features
    rows = sum(b.out_features for b in bases)
    flat = _buffer(device, dtype, offset + rows * k)
    w = flat[offset:offset + rows * k].view(rows, k)
    row = 0
    for base in bases:
        n = base.out_features
        dequantize_nf4(base.weight, base.weight.quant_state, dtype, out=w[row:row + n])
        row += n
    return w


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


def _bnb_would_fuse(m, base, dtype, device):
    """Whether bitsandbytes' gemm_4bit would run its own fused 4-bit GEMM for m rows (small M):
    there the base does not dequantize, so this module keeps the original path (as nf4.py does)."""
    try:
        from bitsandbytes.backends.cuda import ops
    except Exception:  # noqa: BLE001 - another bitsandbytes layout
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
    try:
        import bitsandbytes as bnb
    except Exception:  # noqa: BLE001
        return None
    linear4bit = bnb.nn.Linear4bit
    if not isinstance(module, linear4bit) or type(module).forward is not linear4bit.forward:
        return None
    weight = getattr(module, "weight", None)
    state = getattr(weight, "quant_state", None)
    if state is None or getattr(state, "quant_type", None) != "nf4" or weight.requires_grad:
        return None
    shape = getattr(state, "shape", None)
    if shape is None or tuple(int(s) for s in shape) != (module.out_features, module.in_features):
        return None
    if module.bias is not None and module.bias.requires_grad:
        return None
    if module.compute_dtype not in _HALVES or not weight.is_cuda or not _no_hooks(module):
        return None
    return module


def _lora(module):
    """(base, lora_A, lora_B, dropout p, scaling) of a peft LoRA Linear4bit with exactly one active
    plain LoRA adapter, else None."""
    try:
        from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit
    except Exception:  # noqa: BLE001
        return None
    if type(module) is not LoraLinear4bit or not _no_hooks(module):
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
                 "head_dim", "bias", "spans", "n_tokens")


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
    if any(t.device != x.device for t in (q_base.weight, k_base.weight, v_base.weight, o_base.weight,
                                           a_q.weight, b_q.weight, a_v.weight, b_v.weight)):
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
    # the spans must tile the packed sequence in order (pack_batch's spans do)
    position = 0
    for start, end in spans:
        if start != position or end <= start:
            return None
        position = end
    if position != n_tokens:
        return None
    if any(_bnb_would_fuse(n_tokens, b, half, x.device) for b in bases + (o_base,)):
        return None
    plan = _AttnPlan()
    plan.bases, plan.o_base, plan.lora_q, plan.lora_v = bases, o_base, (a_q, b_q), (a_v, b_v)
    plan.p, plan.scaling, plan.half = p, scaling, half
    plan.heads, plan.kv_heads, plan.head_dim = heads, kv_heads, head_dim
    plan.bias = _cached_bias(attn, bases, half)
    plan.spans, plan.n_tokens = spans, n_tokens
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


def _scaled_mm(a, b, alpha, out):
    """out = alpha * (a @ b), in out's dtype (fp32 out from half a and b accumulates in fp32)."""
    if out.dtype == a.dtype:
        return torch.addmm(out, a, b, beta=0.0, alpha=alpha, out=out)
    return torch.addmm(out, a, b, beta=0.0, alpha=alpha, out_dtype=out.dtype, out=out)


class _QKVRope(torch.autograd.Function):
    """x (1, N, D) -> q, k, v (1, H, N, Dh) in half: q and k rotated, LoRA added to q and v."""

    @staticmethod
    def forward(ctx, x, a_q, b_q, a_v, b_v, cos, sin, plan):
        half, n, d = plan.half, plan.n_tokens, x.shape[-1]
        q_base, k_base, v_base = plan.bases
        nq, nk, nv = q_base.out_features, k_base.out_features, v_base.out_features
        r = a_q.shape[0]
        x16 = x.reshape(n, d)
        x16 = x16 if x16.dtype == half else x16.to(half)

        w = _weights(plan.bases, half, x.device)
        qkv = torch.addmm(plan.bias, x16, w.t()) if plan.bias is not None else torch.mm(x16, w.t())

        # the four adapter matrices cast to half in one multi-tensor launch
        a16 = torch.empty((2 * r, d), dtype=half, device=x.device)
        b16 = torch.empty((nq + nv, r), dtype=half, device=x.device)
        torch._foreach_copy_([a16[:r], a16[r:], b16[:nq], b16[nq:]], [a_q, a_v, b_q, b_v])
        rng = None
        if plan.p > 0.0:
            rng = torch.cuda.get_rng_state(x.device)
            dropped, _ = torch.native_dropout(x16.expand(2, n, d), plan.p, True)
            h = torch.bmm(dropped, a16.view(2, r, d).transpose(1, 2))  # (2, N, r)
            h_q, h_v = h[0], h[1]
            del dropped
        else:
            h = torch.mm(x16, a16.t())  # (N, 2r)
            h_q, h_v = h[:, :r], h[:, r:]
        qkv[:, :nq].addmm_(h_q, b16[:nq].t(), alpha=plan.scaling)
        qkv[:, nq + nk:].addmm_(h_v, b16[nq:].t(), alpha=plan.scaling)

        # rotated q and k in their own token-major buffers: SDPA keeps q for the backward, and a view
        # of qkv would keep k and v alive with it; qkv itself dies once repeat_kv has copied v
        cos, sin = _tables(cos, sin)
        q4 = _heads(torch.empty((n, nq), dtype=half, device=x.device), n, plan.heads, plan.head_dim)
        k4 = _heads(torch.empty((n, nk), dtype=half, device=x.device), n, plan.kv_heads,
                    plan.head_dim)
        rope._launch(_heads(qkv[:, :nq], n, plan.heads, plan.head_dim),
                     _heads(qkv[:, nq:nq + nk], n, plan.kv_heads, plan.head_dim), q4, k4, cos, sin,
                     backward=False)
        v4 = _heads(qkv[:, nq + nk:], n, plan.kv_heads, plan.head_dim)

        ctx.save_for_backward(x16, h, a16, b16, cos, sin)
        ctx.plan, ctx.rng, ctx.x_shape, ctx.x_dtype = plan, rng, x.shape, x.dtype
        ctx.dims = (n, d, nq, nk, nv, r)
        return q4, k4, v4

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_q, grad_k, grad_v):
        x16, h, a16, b16, cos, sin = ctx.saved_tensors
        plan = ctx.plan
        n, d, nq, nk, nv, r = ctx.dims
        half, s, p = plan.half, plan.scaling, plan.p
        device = x16.device

        g = torch.empty((n, nq + nk + nv), dtype=half, device=device)
        gq4 = _heads(g[:, :nq], n, plan.heads, plan.head_dim)
        gk4 = _heads(g[:, nq:nq + nk], n, plan.kv_heads, plan.head_dim)
        rope._launch(_last_dim_contiguous(grad_q), _last_dim_contiguous(grad_k), gq4, gk4, cos, sin,
                     backward=True)
        _heads(g[:, nq + nk:], n, plan.kv_heads, plan.head_dim).copy_(grad_v)
        g_q, g_v = g[:, :nq], g[:, nq + nk:]

        need_x, need_a = ctx.needs_input_grad[0], ctx.needs_input_grad[1] or ctx.needs_input_grad[3]
        need_b = ctx.needs_input_grad[2] or ctx.needs_input_grad[4]
        # dL/dh = scaling * g @ B for both adapters; its dropout part keeps the masks' scale for the
        # addmm's beta at the end
        beta = 1.0
        if p > 0.0:
            gh = torch.empty((2, n, r), dtype=half, device=device)
            _scaled_mm(g_q, b16[:nq], s, gh[0])
            _scaled_mm(g_v, b16[nq:], s, gh[1])
            current = torch.cuda.get_rng_state(device)
            torch.cuda.set_rng_state(ctx.rng, device)
            try:
                dropped, mask = torch.native_dropout(x16.expand(2, n, d), p, True)
            finally:
                torch.cuda.set_rng_state(current, device)
            grad_a = torch.bmm(gh.transpose(1, 2), dropped, out_dtype=torch.float32) if need_a else None
            del dropped
            if need_x:
                g_drop = torch.bmm(gh, a16.view(2, r, d))  # (2, N, D), half like the base's
                lora_x = torch.mul(g_drop[0], mask[0],
                                   out=torch.empty((n, d), dtype=torch.float32, device=device))
                lora_x.addcmul_(g_drop[1], mask[1])
                beta = 1.0 / (1.0 - p)
                del g_drop
            grad_aq, grad_av = (grad_a[0], grad_a[1]) if need_a else (None, None)
            h_q, h_v = h[0], h[1]
        else:
            gh = torch.empty((n, 2 * r), dtype=half, device=device)
            _scaled_mm(g_q, b16[:nq], s, gh[:, :r])
            _scaled_mm(g_v, b16[nq:], s, gh[:, r:])
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
            w = _weights(plan.bases, half, device)
            # one GEMM for the three projections, the adapters' share added as C (in place, fp32)
            grad_x = torch.addmm(lora_x, g, w, beta=beta, out_dtype=torch.float32, out=lora_x)
            grad_x = grad_x.view(ctx.x_shape)
            if ctx.x_dtype != torch.float32:
                grad_x = grad_x.to(ctx.x_dtype)
        grads = [grad_aq, grad_bq, grad_av, grad_bv]
        params = [m.weight for m in plan.lora_q + plan.lora_v]
        grads = [None if g_ is None else g_.to(p_.dtype) for g_, p_ in zip(grads, params)]
        return (grad_x, *grads, None, None, None)


class _Linear4bitHalf(torch.autograd.Function):
    """x16 (M, K) @ W.T (+ bias) for a frozen 4-bit layer, W rebuilt in the shared buffer each way."""

    @staticmethod
    def forward(ctx, x16, base, bias):
        w = _weights((base,), x16.dtype, x16.device)
        out = torch.addmm(bias, x16, w.t()) if bias is not None else torch.mm(x16, w.t())
        ctx.base = base
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        half = ctx.base.compute_dtype
        grad = grad if grad.dtype == half else grad.to(half)
        w = _weights((ctx.base,), half, grad.device)
        return torch.mm(grad, w), None, None


def _fused_attention(attn, x, position_embeddings, plan):
    import transformers.models.qwen2.modeling_qwen2 as mq

    a_q, b_q = plan.lora_q
    a_v, b_v = plan.lora_v
    cos, sin = position_embeddings
    query, key, value = _QKVRope.apply(x, a_q.weight, b_q.weight, a_v.weight, b_v.weight, cos, sin,
                                       plan)
    groups = plan.heads // plan.kv_heads
    if groups > 1:
        key = mq.repeat_kv(key, groups)
        value = mq.repeat_kv(value, groups)
    dropout = 0.0 if not attn.training else attn.attention_dropout
    n = plan.n_tokens
    if len(plan.spans) == 1:
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout,
                                             scale=attn.scaling, is_causal=n > 1).transpose(1, 2)
    else:
        lengths = [end - start for start, end in plan.spans]
        pieces = [F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, scale=attn.scaling,
                                                 is_causal=length > 1).transpose(1, 2)
                  for q, k, v, length in zip(query.split(lengths, dim=2), key.split(lengths, dim=2),
                                             value.split(lengths, dim=2), lengths)]
        out = torch.cat(pieces, dim=1)  # (1, N, H, Dh), token-major
    out = out.reshape(n, plan.heads * plan.head_dim)
    out = _Linear4bitHalf.apply(out, plan.o_base, _cached_bias(plan.o_base, (plan.o_base,), plan.half))
    return out.view(1, n, -1)


# ---------------------------------------------------------------------------------------------
# MLP


class _MLPPlan:
    __slots__ = ("gate", "up", "down", "half", "split")


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
    if any(m.weight.device != x.device for m in (gate, up, down)):
        return None
    m = x.numel() // hidden
    if any(_bnb_would_fuse(m, b, half, x.device) for b in (gate, up, down)):
        return None
    plan = _MLPPlan()
    plan.gate, plan.up, plan.down, plan.half, plan.split = gate, up, down, half, _gate_up_split()
    return plan


def _gate_up_matmul(plan, left, right_t, out):
    """out (2I, ·) = [W_gate; W_up] @ right_t (`left` False) or out (·, K) = left @ [W_gate; W_up]
    (`right_t` None), with the two weights in one buffer or one after the other (split)."""
    inter = plan.gate.out_features
    if not plan.split:
        w = _weights((plan.gate, plan.up), plan.half, out.device)
        if right_t is not None:
            return torch.mm(w, right_t, out=out)
        return torch.mm(left, w, out_dtype=torch.float32)
    w = _weights((plan.gate,), plan.half, out.device)
    if right_t is not None:
        torch.mm(w, right_t, out=out[:inter])
        w = _weights((plan.up,), plan.half, out.device)
        return torch.mm(w, right_t, out=out[inter:])
    result = torch.mm(left[:, :inter], w, out_dtype=torch.float32)
    w = _weights((plan.up,), plan.half, out.device)
    return torch.addmm(result, left[:, inter:], w, out_dtype=torch.float32)


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan):
        half = plan.half
        d, inter = x.shape[-1], plan.gate.out_features
        n = x.numel() // d
        padded = -(-n // 8) * 8  # leading dimension of the transposed products, 16-byte aligned
        x16 = x.reshape(n, d)
        x16 = x16 if x16.dtype == half else x16.to(half)
        gate_up = torch.empty((2 * inter, padded), dtype=half, device=x.device)
        _gate_up_matmul(plan, None, x16.t(), gate_up[:, :n])
        hidden = torch.empty((inter, padded), dtype=half, device=x.device)
        count = inter * padded
        # the padding columns hold garbage that no GEMM reads
        example_swiglu_fwd_kernel[(triton.cdiv(count, SWIGLU_BLOCK),)](
            gate_up[:inter], gate_up[inter:], hidden, count, BLOCK=SWIGLU_BLOCK)
        w = _weights((plan.down,), half, x.device)
        out = torch.mm(hidden[:, :n].t(), w.t())  # (N, D)
        ctx.save_for_backward(gate_up)
        ctx.plan, ctx.x_shape, ctx.x_dtype, ctx.n = plan, x.shape, x.dtype, n
        return out.view(*x.shape[:-1], d)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (gate_up,) = ctx.saved_tensors
        plan, n = ctx.plan, ctx.n
        half = plan.half
        inter, padded = plan.gate.out_features, gate_up.shape[1]
        grad16 = grad.reshape(n, grad.shape[-1])
        grad16 = grad16 if grad16.dtype == half else grad16.to(half)
        w = _weights((plan.down,), half, grad.device)
        grad_hidden = torch.empty((inter, padded), dtype=half, device=grad.device)
        torch.mm(w.t(), grad16.t(), out=grad_hidden[:, :n])
        grad_gate_up = torch.empty_like(gate_up)
        count = inter * padded
        example_swiglu_bwd_kernel[(triton.cdiv(count, SWIGLU_BLOCK),)](
            grad_hidden, gate_up[:inter], gate_up[inter:], grad_gate_up[:inter], grad_gate_up[inter:],
            count, BLOCK=SWIGLU_BLOCK)
        del grad_hidden
        grad_x = _gate_up_matmul(plan, grad_gate_up[:, :n].t(), None, grad_gate_up)
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
            return previous(self, x)
        return _MLP.apply(x, plan)

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
