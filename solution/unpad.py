"""Padding removal for the training step (allowed by the rules: "removing padding inside a batch in
your own code is fine; mixing rows across batches is not").

The collator pads the 4 rows of a batch to the longest one; on the task data that makes 43% of the
positions padding, and the base runs every layer, the head and the loss on them. Here the real
tokens of the batch are packed into one sequence of N tokens (rows stay in order, each row keeps its
own positions 0..len-1 for the rotary embedding), the decoder layers run on (1, N, D), and attention
is causal within each row's slice: exactly the tokens a real token attends to in the base (a padded
row with a causal + padding mask). The fused head (fused_ce.py) scores the packed tokens and scatters
the per-token values back to (B, T) for the trainer's metrics and Soup's rewind log, so the logged
numbers keep their base meaning.

Alignment tail: N is rarely a multiple of 8 (6 of the 64 steps), and N is the contiguous extent of
several GEMM operands in the layers (the transposed gate|up product and its gradient, the LoRA and
weight-gradient products), where cuBLAS then falls back to its align1/align2 kernels at about half
the speed. `varlen_model_forward` appends a few dummy tokens (the last real id again, positions
0..pad-1, so a short row of its own) to make the layers' token count a multiple of
`SOUP_SOLUTION_PAD_TOKENS` (16; 0 or 1 turns it off) and cuts them off before the final norm, so the
head, the loss, the metrics and the rewind values only ever see the N real tokens. The tail belongs
to none of the spans; attention treats it as one more sequence (`sequences`), so its rows hold real,
finite values: no real token attends to them, they get no loss, so their gradient is exactly zero
and they add exactly nothing to the LoRA weight gradients (sums over all rows).

Residual carry: with rmsnorm.py's decoder layer, each layer hands its MLP output and its residual
on unsummed and the next layer's input norm adds them in its kernel (the final norm for the last
layer), with the base's rounding: one fp32 add less per layer in the forward, and its gradient
accumulation and half cast less in the backward. `SOUP_SOLUTION_CARRY_RESIDUAL=0` turns it off.

Attention (`attention`): one launch of the CUTLASS memory-efficient kernel each way per layer for
all the sequences at once (`aten::_efficient_attention_forward/backward` with `cu_seqlens`, the
variable-length form of the kernel SDPA's mem-efficient backend runs; the per-row SDPA calls it
replaces ran the same kernel once per row and direction, plus split/cat copies). The kernel wants
as many k/v heads as q heads: k and v are expanded inside the autograd function for the launch and
again in the backward, so only the compact k and v (2 heads) are kept for the backward, not the
6x copies that `repeat_kv` used to leave saved. The causal mask is applied from the top left of
each sequence, as `is_causal` does per row, and the blocking starts at each sequence's first token,
so the forward is bitwise the per-row SDPA result; the backward matches it to rounding (dq).
`SOUP_SOLUTION_VARLEN_ATTN=0` keeps the per-row SDPA loop, which also serves any input the kernel
path does not take (CPU, fp32 without autocast, bf16 before sm_80, attention dropout, head sizes
other than multiples of 8 up to 128, SDPA's mem-efficient backend switched off).

Under autocast the base's SDPA also runs in fp16 (HF repeats the kv heads and passes the mask, no
GQA fast path), so the causal attention here computes the same thing.
"""

import contextlib
import os

import torch
import torch.nn.functional as F


def _env_flag(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


_STATE = {
    "pad": int(os.environ.get("SOUP_SOLUTION_PAD_TOKENS", "16") or "0"),
    "varlen": _env_flag("SOUP_SOLUTION_VARLEN_ATTN", "1"),
    "carry": _env_flag("SOUP_SOLUTION_CARRY_RESIDUAL", "1"),
    # the attention backward's key splits (aten's num_splits_key; 0 = the kernel's own choice). One
    # split: each key block's dK/dV is summed by one program (no fp32 accumulation workspace per
    # split: the step's peak sat in this backward after the early gate|up offload, 78.8 MiB in the
    # op at ~1,400 tokens) and dQ is added in a fixed order, so it is bit-reproducible
    "splits_key": int(os.environ.get("SOUP_SOLUTION_ATTN_SPLITS_KEY", "0") or "0"),
    "tail_id": None,  # the tail's token id; None: the last real id (tests vary it)
}
_CTX = {"spans": None, "cu": None, "last_mlp": None}
_CAUSAL_FROM_TOP_LEFT = 1  # aten's custom_mask_type for SDPA's is_causal


@contextlib.contextmanager
def varlen(spans):
    """While active, Qwen2Attention runs causal attention within each of these (start, end) spans
    of the single packed sequence (and within the padding tail after the last span)."""
    previous = _CTX["spans"], _CTX["cu"], _CTX["last_mlp"]
    _CTX["spans"] = spans
    try:
        yield
    finally:
        _CTX["spans"], _CTX["cu"], _CTX["last_mlp"] = previous


def pack_batch(input_ids, attention_mask):
    """(B, T) ids and 0/1 mask -> packed ids (1, N), per-row positions (1, N), the flat indices of
    the kept tokens into B*T, and the (start, end) span of every non-empty row in the packed order."""
    mask = attention_mask.to(torch.bool)
    lengths = mask.sum(dim=1)
    idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
    ids = input_ids.flatten()[idx].unsqueeze(0)
    pos = (torch.cumsum(mask.to(torch.int64), dim=1) - 1).flatten()[idx].unsqueeze(0)
    ends = torch.cumsum(lengths, dim=0).tolist()
    spans = [(end - length, end) for end, length in zip(ends, lengths.tolist()) if length > 0]
    return ids, pos, idx, spans


def scatter_rows(values, idx, shape):
    """(N,) values of the packed tokens back to shape (B, T), zeros at padding."""
    full = torch.zeros(shape[0] * shape[1], dtype=values.dtype, device=values.device)
    full[idx] = values
    return full.view(shape)


def sequences(spans, n_tokens):
    """The boundaries (0, e1, e2, ..., n_tokens) of the sequences attention runs over in a packed
    sequence of n_tokens: the spans, which must follow each other from 0, then the tokens after the
    last span (the padding tail) as one more sequence. None for spans that do not fit that."""
    bounds = [0]
    for start, end in spans:
        if start != bounds[-1] or end <= start:
            return None
        bounds.append(end)
    if bounds[-1] > n_tokens:
        return None
    if bounds[-1] < n_tokens:
        bounds.append(n_tokens)
    return tuple(bounds) if len(bounds) > 1 else None


def _pad_count(n_tokens):
    multiple = _STATE["pad"]
    return (-n_tokens) % multiple if multiple > 1 and n_tokens > 0 else 0


def varlen_model_forward(model, input_ids, position_ids, spans):
    """Qwen2Model.forward for the packed training step: embeddings, rotary tables from the per-row
    positions, the decoder layers (attention within each row through `varlen`) on the tokens plus
    the alignment tail, final norm on the real tokens. Mirrors the original forward with no cache
    and no mask tensor; returns (1, N, D) for the N tokens of `input_ids`."""
    n = input_ids.shape[1]
    pad = _pad_count(n)
    if pad:
        fill = input_ids[:, -1:]
        if _STATE["tail_id"] is not None:
            fill = torch.full_like(fill, _STATE["tail_id"])
        input_ids = torch.cat([input_ids, fill.expand(1, pad)], dim=1)
        tail = torch.arange(pad, dtype=position_ids.dtype, device=position_ids.device)
        position_ids = torch.cat([position_ids, tail.unsqueeze(0)], dim=1)
    hidden = model.embed_tokens(input_ids)
    position_embeddings = model.rotary_emb(hidden, position_ids)
    layers = model.layers[: model.config.num_hidden_layers]
    carry = _carries(layers)
    branch = None  # the previous layer's MLP output, not yet added to `hidden` (see rmsnorm.py)
    with varlen(spans):
        bounds = sequences(spans, n + pad)
        if bounds is not None and hidden.is_cuda:
            _cu_seqlens(bounds, hidden.device)  # here, before any layer's work is queued
        _CTX["last_mlp"] = getattr(layers[-1], "mlp", None) if len(layers) else None
        for layer in layers:
            extra = {"_soup_branch": branch, "_soup_carry": True} if carry else {}
            out = layer(hidden, attention_mask=None, position_ids=position_ids,
                        past_key_values=None, use_cache=False,
                        position_embeddings=position_embeddings, **extra)
            if carry and isinstance(out, tuple):
                branch, hidden = out
            else:
                hidden, branch = out, None
    if branch is None:
        hidden = model.norm(hidden)
    else:
        from transformers.models.qwen2 import modeling_qwen2 as mq

        from .rmsnorm import add_norm

        hidden = add_norm(model.norm, hidden, branch, mq.Qwen2RMSNorm)[1]
    # the norm works row by row: the real rows are the same with or without the tail
    return hidden[:, :n] if pad else hidden


def _carries(layers):
    """Whether the layers' forward is rmsnorm.py's, which carries `residual + mlp_out` into the next
    layer's input norm (`SOUP_SOLUTION_CARRY_RESIDUAL=0` turns it off)."""
    if not _STATE["carry"] or len(layers) == 0:
        return False
    forward = type(layers[0]).forward
    return (getattr(forward, "_soup_carries", False)
            and all(type(layer).forward is forward and "forward" not in vars(layer)
                    and not getattr(layer, "gradient_checkpointing", False) for layer in layers))


# ---------------------------------------------------------------------------------------------
# attention


def _cu_seqlens(bounds, device):
    """`bounds` as the kernel's int32 cu_seqlens on `device`, built once per step (the copy from
    host memory waits for the stream, so it is made before the layers queue their work)."""
    cached = _CTX["cu"]
    if cached is not None and cached[0] == bounds and cached[1] == device:
        return cached[2]
    cu = torch.tensor(bounds, dtype=torch.int32, device=device)
    if _CTX["spans"] is not None:  # kept for the step's other layers, dropped when `varlen` ends
        _CTX["cu"] = (bounds, device, cu)
    return cu


def _expand_heads(t, groups):
    """(1, N, Hkv, D) -> (1, N, Hkv * groups, D), head h reading kv head h // groups (repeat_kv)."""
    if groups == 1:
        return t
    b, n, h, d = t.shape
    return t.unsqueeze(3).expand(b, n, h, groups, d).reshape(b, n, h * groups, d)


def _expanded(t, q, groups):
    """k or v as the kernel takes them for q: expanded, unless they already have q's heads (the
    recompute's tensors under fused_proj.py's SOUP_SOLUTION_ATTN_GLUE)."""
    return t if t.shape[2] == q.shape[2] else _expand_heads(t, groups)


def _compact(t):
    """t, or a copy of it when it is a view into a larger buffer (a saved view would keep the whole
    buffer alive until the backward, e.g. v as a column slice of the fused qkv product)."""
    if t.is_contiguous() and t.untyped_storage().nbytes() == t.numel() * t.element_size():
        return t
    return t.clone(memory_format=torch.contiguous_format)


def varlen_forward(q, k, v, cu, max_len, scale):
    """(out, lse, seed, offset) of the kernel's forward for `_VarlenAttention`'s operands; also the
    bit-for-bit recompute of fused_proj.py's attention recompute (the kernel's forward has no
    atomics: the same operands give the same output and log-sum-exp)."""
    groups = q.shape[2] // k.shape[2]
    out, lse, seed, offset, _, _ = torch.ops.aten._efficient_attention_forward(
        q, _expand_heads(k, groups), _expand_heads(v, groups), None, cu, cu, max_len, max_len,
        0.0, _CAUSAL_FROM_TOP_LEFT, True, scale=scale)
    return out, lse, seed, offset


class _VarlenAttention(torch.autograd.Function):
    """q (1, N, Hq, D), k and v (1, N, Hkv, D), last dim contiguous -> causal attention within each
    sequence of `cu` (int32 boundaries), (1, N, Hq, D) contiguous. Saved: q, the compact k and v,
    the output and the log-sum-exp (with dropout 0 the backward needs nothing else); or, given a
    `recompute` object (fused_proj.py's attention recompute), nothing but `cu`: the backward asks
    that object for the five tensors, which it rebuilds bit for bit."""

    @staticmethod
    def forward(ctx, q, k, v, cu, max_len, scale, recompute=None):
        groups = q.shape[2] // k.shape[2]
        out, lse, seed, offset = varlen_forward(q, k, v, cu, max_len, scale)
        ctx.recompute = recompute
        if recompute is None:
            kept = (_compact(q), _compact(k), _compact(v), out, lse)
            from .offload import ATTN, mark_offload

            if ATTN:  # SOUP_SOLUTION_OFFLOAD_ATTN: to host memory instead of recomputed
                for t in kept:
                    mark_offload(t)
            ctx.save_for_backward(*kept, cu)
        else:
            recompute.attention_taken(cu, max_len, scale, q, k, v, out, lse)
            ctx.save_for_backward(cu)
        ctx.args = (seed, offset, max_len, scale, groups)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        seed, offset, max_len, scale, groups = ctx.args
        if ctx.recompute is None:
            q, k, v, out, lse, cu = ctx.saved_tensors
        else:
            (cu,) = ctx.saved_tensors
            q, k, v, out, lse, seed, offset = ctx.recompute.attention_tensors()
            ctx.recompute = None
        grad_out = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
        grad_q, grad_k, grad_v, _ = _efficient_backward(
            grad_out, q, _expanded(k, q, groups), _expanded(v, q, groups), out, cu, max_len, lse,
            seed, offset, scale)
        if groups > 1:  # the expansion's backward: the sum over each kv head's q heads
            grad_k = grad_k.unflatten(2, (q.shape[2] // groups, groups)).sum(3)
            grad_v = grad_v.unflatten(2, (q.shape[2] // groups, groups)).sum(3)
        return grad_q, grad_k, grad_v, None, None, None, None


def _efficient_backward(grad_out, q, k, v, out, cu, max_len, lse, seed, offset, scale):
    """aten's variable-length memory-efficient backward, with SOUP_SOLUTION_ATTN_SPLITS_KEY when set
    (a torch without the keyword takes the kernel's own choice from then on)."""
    args = (grad_out, q, k, v, None, out, cu, cu, max_len, max_len, lse, 0.0, seed, offset,
            _CAUSAL_FROM_TOP_LEFT, False)
    splits = _STATE["splits_key"]
    if splits > 0:
        try:
            return torch.ops.aten._efficient_attention_backward(*args, scale=scale,
                                                                num_splits_key=splits)
        except (TypeError, RuntimeError):
            _STATE["splits_key"] = 0
    return torch.ops.aten._efficient_attention_backward(*args, scale=scale)


def _sdpa_dtype(t):
    """The dtype SDPA computes in for input `t`: autocast's on CUDA when it is on, else t's."""
    if t.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return t.dtype


def _varlen_ok(query, key, value, dropout):
    if not _STATE["varlen"] or dropout != 0.0 or torch.compiler.is_compiling():
        return False
    if not (query.is_cuda and query.dim() == 4 and key.dim() == 4 and key.shape == value.shape):
        return False
    if key.device != query.device or value.device != query.device:
        return False
    b, heads, n, d = query.shape
    if b != 1 or key.shape[0] != 1 or key.shape[2] != n or key.shape[3] != d or n == 0:
        return False
    if key.shape[1] == 0 or heads % key.shape[1] or d % 8 or d > 128:
        return False
    dtype = _sdpa_dtype(query)
    if dtype == torch.bfloat16:
        if torch.cuda.get_device_capability(query.device) < (8, 0):
            return False
    elif dtype != torch.float16:
        return False
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return False
    return torch.backends.cuda.mem_efficient_sdp_enabled()


def _per_row(query, key, value, bounds, scale, dropout):
    """The per-row SDPA loop (`is_causal` on each sequence's slice, kv heads repeated)."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = mq.repeat_kv(key, groups)
        value = mq.repeat_kv(value, groups)
    pieces = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        pieces.append(F.scaled_dot_product_attention(
            query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
            dropout_p=dropout, scale=scale, is_causal=end - start > 1).transpose(1, 2))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)


def kernel_operands(query, key, value, dtype):
    """The kernel's token-major q, k, v from the projections' (1, H, N, D) views."""
    q, k, v = (t.transpose(1, 2).to(dtype) for t in (query, key, value))
    return tuple(t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))


def attention(query, key, value, bounds, scale, dropout=0.0, recompute=None):
    """Causal attention within each packed sequence of `bounds` (see `sequences`): query
    (1, Hq, N, D), key and value (1, Hkv, N, D) as the projections leave them (any strides with
    the head dim contiguous) -> (1, N, Hq, D) token-major, in the dtype SDPA would return.
    `recompute`: fused_proj.py's attention recompute object; the variable-length path then keeps
    nothing for the backward but asks it (the per-row loop ignores it and keeps its tensors)."""
    if not _varlen_ok(query, key, value, dropout):
        return _per_row(query, key, value, bounds, scale, dropout)
    q, k, v = kernel_operands(query, key, value, _sdpa_dtype(query))
    max_len = max(end - start for start, end in zip(bounds[:-1], bounds[1:]))
    return _VarlenAttention.apply(q, k, v, _cu_seqlens(bounds, q.device), max_len, scale, recompute)


def patch_qwen2_attention():
    """Qwen2Attention.forward: inside `varlen`, causal attention within each row of the packed
    sequence."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2Attention
    if getattr(cls, "_soup_varlen", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        spans = _CTX["spans"]
        bounds = None
        if (spans is not None and past_key_values is None and attention_mask is None
                and hidden_states.dim() == 3 and hidden_states.shape[0] == 1):
            bounds = sequences(spans, hidden_states.shape[1])
        if bounds is None:
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = mq.apply_rotary_pos_emb(query, key, cos, sin)
        dropout = 0.0 if not self.training else self.attention_dropout
        attn = attention(query, key, value, bounds, self.scaling, dropout)
        attn = attn.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn), None

    cls.forward = forward
    cls._soup_original_forward = original
    cls._soup_varlen = True
