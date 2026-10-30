"""Packing and attention of the training step (unpad.py): the alignment tail of the packed sequence,
the sequences attention runs over, and the variable-length attention against the per-row SDPA it
replaces and a plain masked reference, forward and backward; the tail's isolation, what is saved
for the backward, launches per call, and the guards that keep the per-row path."""

import pytest
import torch

from solution import unpad

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

HEADS, KV_HEADS, HEAD_DIM = 12, 2, 128
SCALE = HEAD_DIM ** -0.5
PATTERNS = {
    "354 tokens in 4 rows": [100, 90, 84, 80],
    "1427 tokens in 4 rows, one long": [512, 401, 311, 203],
    "rows of length 1": [1, 37, 1, 64, 1],
    "one row": [300],
}


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _spans(lengths):
    ends = torch.tensor(lengths).cumsum(0).tolist()
    return [(end - length, end) for end, length in zip(ends, lengths)]


def _qkv(n, seed=0, dtype=torch.float16, device="cuda", heads=HEADS, kv_heads=KV_HEADS,
         head_dim=HEAD_DIM):
    """A leaf (N, (Hq + 2 Hkv) D) buffer and q, k, v as the fused projection leaves them: (1, H, N, D)
    transposed views of its column blocks (token-major, v a view into the wider buffer)."""
    g = torch.Generator(device).manual_seed(seed)
    width = (heads + 2 * kv_heads) * head_dim
    qkv = torch.randn(n, width, generator=g, device=device).to(dtype).requires_grad_(True)
    nq, nk = heads * head_dim, kv_heads * head_dim

    def view(cols, h):
        return cols.view(1, n, h, head_dim).transpose(1, 2)

    return qkv, view(qkv[:, :nq], heads), view(qkv[:, nq:nq + nk], kv_heads), \
        view(qkv[:, nq + nk:], kv_heads)


def _per_row_sdpa(q, k, v, bounds, scale=SCALE):
    """What unpad.py ran before: SDPA per row on the memory-efficient backend (the T4's)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        return unpad._per_row(q, k, v, bounds, scale, 0.0)


def _masked_reference(q, k, v, bounds, scale=SCALE):
    """fp32 attention over the whole packed sequence with a block-diagonal causal mask."""
    n, groups = q.shape[2], q.shape[1] // k.shape[1]
    seq = torch.zeros(n, dtype=torch.long, device=q.device)
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        seq[start:end] = i
    pos = torch.arange(n, device=q.device)
    allowed = (seq[:, None] == seq[None, :]) & (pos[None, :] <= pos[:, None])
    k = k.float().repeat_interleave(groups, dim=1)
    v = v.float().repeat_interleave(groups, dim=1)
    scores = (q.float() @ k.transpose(-1, -2)) * scale
    probs = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    return (probs @ v).transpose(1, 2)  # (1, N, H, D)


def _run(fn, qkv, q, k, v, bounds, grad):
    qkv.grad = None
    out = fn(q, k, v, bounds)
    out.backward(grad.to(out.dtype))
    return out.detach(), qkv.grad.clone()


@pytest.fixture(autouse=True)
def _defaults():
    saved = dict(unpad._STATE)
    unpad._STATE.update(pad=16, varlen=True, carry=True)
    yield
    unpad._STATE.update(saved)


# ------------------------------------------------------------------------------------------------
# sequences and the tail


def test_sequences():
    assert unpad.sequences([(0, 10), (10, 13), (13, 22)], 22) == (0, 10, 13, 22)
    assert unpad.sequences([(0, 10), (10, 13), (13, 22)], 32) == (0, 10, 13, 22, 32)  # the tail
    assert unpad.sequences([], 16) == (0, 16)  # nothing but tail
    assert unpad.sequences([(0, 10), (12, 20)], 20) is None  # a gap
    assert unpad.sequences([(1, 10)], 10) is None  # not from 0
    assert unpad.sequences([(0, 10), (10, 10)], 10) is None  # an empty span
    assert unpad.sequences([(0, 10), (10, 30)], 20) is None  # past the tokens
    assert unpad.sequences([], 0) is None


@pytest.mark.parametrize("n,pad", [(826, 6), (832, 0), (1, 15), (0, 0), (17, 15)])
def test_pad_count(n, pad):
    assert unpad._pad_count(n) == pad
    unpad._STATE["pad"] = 0
    assert unpad._pad_count(n) == 0


# ------------------------------------------------------------------------------------------------
# varlen attention against the per-row SDPA and the masked reference


@pytest.mark.parametrize("tail", [0, 14])
@pytest.mark.parametrize("pattern", list(PATTERNS))
def test_varlen_matches_per_row_and_reference(pattern, tail):
    lengths = PATTERNS[pattern]
    n = sum(lengths) + tail
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n, seed=len(lengths) + tail)
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda",
                       generator=torch.Generator("cuda").manual_seed(1)).half()

    calls = []
    apply = unpad._VarlenAttention.apply

    def attention(*args):
        calls.append(1)
        return apply(*args)

    unpad._VarlenAttention.apply = attention
    try:
        out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    finally:
        unpad._VarlenAttention.apply = apply
    assert calls == [1]
    ref_out, ref_dqkv = _run(_per_row_sdpa, qkv, q, k, v, bounds, grad)
    assert out.shape == (1, n, HEADS, HEAD_DIM) and out.dtype == torch.float16 and out.is_contiguous()
    # the same kernel with the same blocking per sequence: the forward is bitwise per-row SDPA's
    assert torch.equal(out, ref_out)
    assert torch.isfinite(dqkv).all()
    assert _rel(dqkv, ref_dqkv) < 2e-3

    exact_out, exact_dqkv = _run(_masked_reference, qkv, q, k, v, bounds, grad.float())
    assert _rel(out, exact_out) < 2e-3
    assert _rel(dqkv, exact_dqkv) < 1e-2
    assert _rel(ref_dqkv, exact_dqkv) < 1e-2


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
                    reason="bf16 memory-efficient attention needs sm_80")
def test_varlen_bf16():
    lengths = PATTERNS["354 tokens in 4 rows"]
    n = sum(lengths) + 14
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n, dtype=torch.bfloat16)
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda").bfloat16()
    out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    exact_out, exact_dqkv = _run(_masked_reference, qkv, q, k, v, bounds, grad.float())
    assert out.dtype == torch.bfloat16
    assert _rel(out, exact_out) < 1e-2 and _rel(dqkv, exact_dqkv) < 3e-2


def test_autocast_casts_like_sdpa():
    """fp32 q, k, v under fp16 autocast (unpad's own attention path): half attention, fp32 grads."""
    lengths = [60, 33, 40]
    n = sum(lengths) + 11
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n, dtype=torch.float32)
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda").half()
    with torch.autocast("cuda", dtype=torch.float16):
        out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
        ref_out, ref_dqkv = _run(_per_row_sdpa, qkv, q, k, v, bounds, grad)
    assert out.dtype == ref_out.dtype == torch.float16 and dqkv.dtype == torch.float32
    assert torch.equal(out, ref_out)
    assert _rel(dqkv, ref_dqkv) < 2e-3


# ------------------------------------------------------------------------------------------------
# the alignment tail is isolated from the real rows


def test_tail_rows_are_finite_and_isolated():
    lengths = PATTERNS["354 tokens in 4 rows"]
    real, tail = sum(lengths), 14
    qkv, q, k, v = _qkv(real + tail, seed=3)
    bounds = unpad.sequences(_spans(lengths), real + tail)
    grad = torch.randn(1, real + tail, HEADS, HEAD_DIM, device="cuda").half()
    grad[:, real:] = 0  # the tail gets no loss
    out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    assert torch.isfinite(out).all()
    assert torch.all(dqkv[real:] == 0)  # exactly: nothing flows from the tail to the LoRA sums

    # the real rows do not see the tail: same output as without it, same gradients
    small, sq, sk, sv = _qkv(real, seed=3)
    with torch.no_grad():
        small.copy_(qkv[:real])
    small_out, small_dqkv = _run(lambda *a: unpad.attention(*a, SCALE), small, sq, sk, sv,
                                 unpad.sequences(_spans(lengths), real), grad[:, :real])
    assert torch.equal(out[:, :real], small_out)
    assert _rel(dqkv[:real], small_dqkv) < 1e-3


# ------------------------------------------------------------------------------------------------
# what is saved for the backward, launches


def _saved_bytes(fn):
    shapes = []

    def pack(t):
        shapes.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, shapes


def test_saved_tensors_hold_no_expanded_kv():
    lengths = PATTERNS["1427 tokens in 4 rows, one long"]
    n = sum(lengths) + 5
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n)
    out, saved = _saved_bytes(lambda: unpad.attention(q, k, v, bounds, SCALE))
    full = (1, n, HEADS, HEAD_DIM)
    compact = (1, n, KV_HEADS, HEAD_DIM)
    assert [s for s, _, _ in saved].count(full) == 2  # q and the output
    assert [s for s, _, _ in saved].count(compact) == 2  # k and v at 2 heads
    total = sum(b for _, _, b in saved)
    half = 2 * n * HEAD_DIM
    lse = 4 * len(lengths + [5]) * HEADS * (max(lengths) + 64)
    assert total <= half * (2 * HEADS + 2 * KV_HEADS) + lse + 64
    # the per-row path keeps the expanded k and v of every row
    unpad._STATE["varlen"] = False
    _, per_row = _saved_bytes(lambda: unpad.attention(q, k, v, bounds, SCALE))
    assert sum(b for _, _, b in per_row) > total + 2 * half * (HEADS - KV_HEADS) * 0.9
    out.backward(torch.ones_like(out))


def test_saved_views_do_not_keep_the_qkv_buffer():
    """q, k, v as column views of one qkv buffer: the saved ones are compact copies, so the buffer
    is not kept until the backward."""
    n = 300
    _, q, k, v = _qkv(n)
    storages = []

    def pack(t):
        storages.append(t.untyped_storage().nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        unpad.attention(q, k, v, (0, n), SCALE)
    assert storages[:3] == [2 * n * HEADS * HEAD_DIM] + [2 * n * KV_HEADS * HEAD_DIM] * 2


def _fmha_launches(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return (sum("fmha" in name and "cutlassF" in name for name in names),
            sum("fmha" in name and "cutlassB" in name for name in names))


@pytest.mark.parametrize("varlen", [True, False])
def test_one_launch_each_way(varlen):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    unpad._STATE["varlen"] = varlen
    lengths = PATTERNS["354 tokens in 4 rows"]
    n = sum(lengths) + 14
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n)

    def forward():
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):  # the T4's backend for the per-row path
            return unpad.attention(q, k, v, bounds, SCALE)

    for _ in range(2):  # warm up
        forward().sum().backward()
    holder = {}
    fwd, _ = _fmha_launches(lambda: holder.update(out=forward()))
    _, bwd = _fmha_launches(lambda: holder["out"].sum().backward())
    rows = len(bounds) - 1
    assert (fwd, bwd) == ((1, 1) if varlen else (rows, rows))


# ------------------------------------------------------------------------------------------------
# guards: the per-row SDPA loop


def _takes_varlen(q, k, v, bounds, dropout=0.0):
    calls = []
    apply = unpad._VarlenAttention.apply

    def attention(*args):
        calls.append(1)
        return apply(*args)

    unpad._VarlenAttention.apply = attention
    try:
        out = unpad.attention(q, k, v, bounds, SCALE, dropout)
    finally:
        unpad._VarlenAttention.apply = apply
    return bool(calls), out


def test_guards_keep_the_per_row_path():
    lengths = [40, 31, 17]
    n = sum(lengths) + 8
    bounds = unpad.sequences(_spans(lengths), n)
    _, q, k, v = _qkv(n)
    assert _takes_varlen(q, k, v, bounds)[0]
    assert not _takes_varlen(q, k, v, bounds, dropout=0.1)[0]
    _, q32, k32, v32 = _qkv(n, dtype=torch.float32)
    taken, out = _takes_varlen(q32, k32, v32, bounds)
    assert not taken and out.dtype == torch.float32  # fp32 without autocast: SDPA in fp32
    with torch.autocast("cuda", dtype=torch.float16):
        assert _takes_varlen(q32, k32, v32, bounds)[0]
    _, qc, kc, vc = _qkv(n, dtype=torch.float32, device="cpu")
    taken, out = _takes_varlen(qc, kc, vc, bounds)
    assert not taken and out.shape == (1, n, HEADS, HEAD_DIM)
    _, q12, k12, v12 = _qkv(n, head_dim=12)
    assert not _takes_varlen(q12, k12, v12, bounds)[0]
    _, q4, k4, v4 = _qkv(n, kv_heads=HEADS)  # no grouping
    assert _takes_varlen(q4, k4, v4, bounds)[0]
    unpad._STATE["varlen"] = False
    assert not _takes_varlen(q, k, v, bounds)[0]
    unpad._STATE["varlen"] = True
    enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    try:
        assert not _takes_varlen(q, k, v, bounds)[0]
    finally:
        torch.backends.cuda.enable_mem_efficient_sdp(enabled)
    two = q.expand(2, -1, -1, -1)
    assert not unpad._varlen_ok(two, k.expand(2, -1, -1, -1), v.expand(2, -1, -1, -1), 0.0)


def test_per_row_fallback_matches_varlen():
    lengths = PATTERNS["rows of length 1"]
    n = sum(lengths) + 3
    bounds = unpad.sequences(_spans(lengths), n)
    qkv, q, k, v = _qkv(n, seed=9)
    grad = torch.randn(1, n, HEADS, HEAD_DIM, device="cuda").half()
    out, dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    unpad._STATE["varlen"] = False
    ref_out, ref_dqkv = _run(lambda *a: unpad.attention(*a, SCALE), qkv, q, k, v, bounds, grad)
    assert _rel(out, ref_out) < 1e-3 and _rel(dqkv, ref_dqkv) < 2e-3


# ------------------------------------------------------------------------------------------------
# the packed model forward with the tail


_MISSING = object()


@pytest.fixture()
def attention_patch():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    names = ("forward", "_soup_original_forward", "_soup_varlen")
    saved = [(name, vars(mq.Qwen2Attention).get(name, _MISSING)) for name in names]
    unpad.patch_qwen2_attention()
    yield
    for name, value in saved:
        if value is _MISSING:
            if name in vars(mq.Qwen2Attention):
                delattr(mq.Qwen2Attention, name)
        else:
            setattr(mq.Qwen2Attention, name, value)


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                         num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=1024,
                         attn_implementation="sdpa")
    return Qwen2ForCausalLM(config).cuda().train()


def _model_step(model, lengths, seen):
    ids = torch.randint(0, 256, (len(lengths), max(lengths)), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(5))
    mask = torch.zeros_like(ids)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
    packed, pos, idx, spans = unpad.pack_batch(ids, mask)
    model.zero_grad(set_to_none=True)
    hooks = [layer.register_forward_pre_hook(lambda m, args, kw: seen.append(args[0].shape[1]),
                                             with_kwargs=True)
             for layer in model.model.layers]
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            hidden = unpad.varlen_model_forward(model.model, packed, pos, spans)
            logits = model.lm_head(hidden).float()
    finally:
        for hook in hooks:
            hook.remove()
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], packed[0, 1:])
    (loss * 1024.0).backward()
    return hidden.detach(), [p.grad.clone() / 1024.0 for p in model.parameters()]


@pytest.mark.parametrize("lengths", [[40, 31, 17], [100, 1, 64, 7], [16, 16]])
def test_varlen_model_forward_tail(attention_patch, lengths):
    model = _tiny_model()
    n = sum(lengths)
    unpad._STATE["pad"] = 0
    seen = []
    ref_hidden, ref_grads = _model_step(model, lengths, seen)
    assert seen == [n, n]
    unpad._STATE["pad"] = 16
    seen.clear()
    hidden, grads = _model_step(model, lengths, seen)
    padded = -(-n // 16) * 16
    assert seen == [padded, padded]  # the layers run on a multiple of 16 tokens
    assert hidden.shape == ref_hidden.shape == (1, n, 256)
    assert unpad._CTX["spans"] is None and unpad._CTX["cu"] is None
    assert torch.isfinite(hidden).all()
    assert _rel(hidden, ref_hidden) < 5e-3
    for g, ref in zip(grads, ref_grads):
        assert torch.isfinite(g).all()
        if ref.abs().max() > 0:
            assert _rel(g, ref) < 1e-2


def test_patched_attention_outside_varlen_is_original(attention_patch):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    model = _tiny_model()
    attn = model.model.layers[0].self_attn
    x = torch.randn(1, 20, 256, device="cuda")
    pos = torch.arange(20, device="cuda").unsqueeze(0)
    pe = model.model.rotary_emb(x, pos)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ref, _ = mq.Qwen2Attention._soup_original_forward(attn, x, pe, None)
        with unpad.varlen([(0, 20)]):
            out, _ = attn(x, pe, None)
        with unpad.varlen([(0, 10), (12, 20)]):  # spans that do not follow each other
            bad, _ = attn(x, pe, None)
    assert _rel(out, ref) < 2e-3
    assert torch.equal(bad, ref)  # took the original forward


# ------------------------------------------------------------------------------------------------
# the residual carried into the next layer's input norm (rmsnorm.py's decoder layer)


@pytest.fixture()
def layer_patches():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import rmsnorm

    names = ("forward", "_soup_original_forward", "_soup_varlen", "_soup_rmsnorm",
             "_soup_add_rmsnorm")
    owners = (mq.Qwen2Attention, mq.Qwen2RMSNorm, mq.Qwen2DecoderLayer)
    saved = [(cls, name, vars(cls).get(name, _MISSING)) for cls in owners for name in names]
    unpad.patch_qwen2_attention()
    rmsnorm.patch_qwen2_rmsnorm()
    rmsnorm.patch_qwen2_decoder_layer()
    yield
    for cls, name, value in saved:
        if value is _MISSING:
            if name in vars(cls):
                delattr(cls, name)
        else:
            setattr(cls, name, value)


def _launches(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    return sum(e.device_type == torch.autograd.DeviceType.CUDA for e in prof.events()), result


@pytest.mark.parametrize("lengths", [[40, 31, 17], [100, 1, 64, 7]])
def test_residual_carry_matches_the_plain_loop(layer_patches, lengths):
    model = _tiny_model()
    assert unpad._carries(model.model.layers)
    results = []
    for carry in (False, True, False, True):  # the second pair after warm-up, for the counts
        unpad._STATE["carry"] = carry
        results.append(_launches(lambda: _model_step(model, lengths, [])))
    (n_plain, (ref_hidden, ref_grads)), (n_carry, (hidden, grads)) = results[2], results[3]
    # the forward: the same sums in the same fp32 rounding, bitwise the plain loop's. The backward
    # adds the residual's gradient inside the norm kernel (an fma, as the post-attention fusion
    # does), and the attention backward's dq is not bit-reproducible anyway
    assert torch.equal(hidden, ref_hidden)
    for g, ref in zip(grads, ref_grads):
        assert _rel(g, ref) < 1e-3
    # per layer: the forward add, and the half cast of the branch's gradient when the MLP returns
    # half (it does unless another test module left an MLP patch in place); per boundary between
    # layers: the sum of the residual's two gradients; one fill more (the final norm's unused sum
    # output gets a zero gradient, which keeps the backward on its warmed-up kernel variant)
    layers = len(model.model.layers)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        half_branch = model.model.layers[0].mlp(torch.randn(1, 4, 256, device="cuda")).dtype != \
            torch.float32
    expected = (3 if half_branch else 2) * layers - 2
    assert n_plain - n_carry == expected, (n_plain, n_carry, half_branch)


def test_residual_carry_falls_back_per_layer(layer_patches):
    """A hook on a layer's input norm: that layer adds the carried branch itself and runs its norm as
    the original would; the results stay those of the plain loop."""
    model = _tiny_model()
    lengths = [40, 31, 17]
    unpad._STATE["carry"] = False
    ref_hidden, ref_grads = _model_step(model, lengths, [])
    seen = []
    handle = model.model.layers[1].input_layernorm.register_forward_hook(
        lambda *a: seen.append(1))
    try:
        unpad._STATE["carry"] = True
        hidden, grads = _model_step(model, lengths, [])
    finally:
        handle.remove()
    assert seen == [1]
    assert torch.equal(hidden, ref_hidden)
    for g, ref in zip(grads, ref_grads):
        assert _rel(g, ref) < 1e-3
    model.model.layers[0].gradient_checkpointing = True  # never carried through a checkpoint
    assert not unpad._carries(model.model.layers)
