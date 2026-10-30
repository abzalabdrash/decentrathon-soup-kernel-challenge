"""Padding removal for the training step (allowed by the rules: "removing padding inside a batch in
your own code is fine; mixing rows across batches is not").

The collator pads the 4 rows of a batch to the longest one; on the task data that makes 43% of the
positions padding, and the base runs every layer, the head and the loss on them. Here the real
tokens of the batch are packed into one sequence of N tokens (rows stay in order, each row keeps its
own positions 0..len-1 for the rotary embedding), the decoder layers run on (1, N, D), and attention
runs per row with `is_causal=True` on that row's slice: exactly the tokens a real token attends to in
the base (a padded row with a causal + padding mask). The fused head (fused_ce.py) scores the packed
tokens and scatters the per-token values back to (B, T) for the trainer's metrics and Soup's rewind
log, so the logged numbers keep their base meaning.

Under autocast the base's SDPA also runs in fp16 (HF repeats the kv heads and passes the mask, no
GQA fast path), so the per-row causal SDPA here computes the same thing, kernel aside.
"""

import contextlib

import torch
import torch.nn.functional as F

_CTX = {"spans": None}


@contextlib.contextmanager
def varlen(spans):
    """While active, Qwen2Attention runs per-row causal attention over these (start, end) spans of
    the single packed sequence."""
    previous = _CTX["spans"]
    _CTX["spans"] = spans
    try:
        yield
    finally:
        _CTX["spans"] = previous


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


def varlen_model_forward(model, input_ids, position_ids, spans):
    """Qwen2Model.forward for the packed training step: embeddings, rotary tables from the per-row
    positions, the decoder layers (attention per row through `varlen`), final norm. Mirrors the
    original forward with no cache and no mask tensor."""
    hidden = model.embed_tokens(input_ids)
    position_embeddings = model.rotary_emb(hidden, position_ids)
    with varlen(spans):
        for layer in model.layers[: model.config.num_hidden_layers]:
            hidden = layer(hidden, attention_mask=None, position_ids=position_ids,
                           past_key_values=None, use_cache=False,
                           position_embeddings=position_embeddings)
    return model.norm(hidden)


def patch_qwen2_attention():
    """Qwen2Attention.forward: inside `varlen`, attention per row over the packed sequence."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    cls = mq.Qwen2Attention
    if getattr(cls, "_soup_varlen", False):
        return
    original = cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None,
                **kwargs):
        spans = _CTX["spans"]
        if (spans is None or past_key_values is not None or attention_mask is not None
                or hidden_states.dim() != 3 or hidden_states.shape[0] != 1):
            return original(self, hidden_states, position_embeddings, attention_mask,
                            past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = mq.apply_rotary_pos_emb(query, key, cos, sin)
        groups = getattr(self, "num_key_value_groups", 1)
        if groups > 1:
            key = mq.repeat_kv(key, groups)
            value = mq.repeat_kv(value, groups)
        dropout = 0.0 if not self.training else self.attention_dropout
        pieces = []
        for start, end in spans:
            pieces.append(F.scaled_dot_product_attention(
                query[:, :, start:end], key[:, :, start:end], value[:, :, start:end],
                dropout_p=dropout, scale=self.scaling, is_causal=end - start > 1))
        attn = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn), None

    cls.forward = forward
    cls._soup_original_forward = original
    cls._soup_varlen = True
