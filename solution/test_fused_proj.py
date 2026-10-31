"""Fused q+k+v / gate+up projections against the path they replace (main's unpad attention, rope,
nf4 and mlp16 patches): a tiny 4-bit Qwen2 with peft LoRA through the packed model forward, the
layers at the 1.5B shapes, LoRA dropout with the masks drawn again in the backward, the guards,
plus kernel launches and saved memory per layer (printed with -s)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

_MISSING = object()


def _owners():
    from bitsandbytes.autograd import _functions as bnb_autograd
    from transformers.models.qwen2 import modeling_qwen2 as mq

    flags = ("forward", "_soup_original_forward", "_soup_varlen", "_soup_mlp16", "_soup_add_rmsnorm",
             "_soup_rmsnorm", "_soup_fused_proj", "_soup_fused_proj_previous")
    return [(mq, ("apply_rotary_pos_emb", "_soup_rope", "_soup_original_apply_rotary_pos_emb")),
            (mq.Qwen2Attention, flags), (mq.Qwen2MLP, flags), (mq.Qwen2DecoderLayer, flags),
            (mq.Qwen2RMSNorm, flags),
            (bnb_autograd.MatMul4Bit, ("forward", "backward", "_soup_nf4", "_soup_original_forward",
                                       "_soup_original_backward"))]


@pytest.fixture(scope="module", autouse=True)
def patched():
    """main's patches of the layer, then the fused ones on top; all undone after the module."""
    from solution import fused_proj
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope
    from solution.unpad import patch_qwen2_attention

    saved = [(owner, name, vars(owner).get(name, _MISSING)) for owner, names in _owners()
             for name in names]
    patch_qwen2_attention()
    patch_qwen2_rope()
    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    fused_proj.patch_qwen2_attention_fused()
    fused_proj.patch_qwen2_mlp_fused()
    recompute = fused_proj._MLP_MEMORY["recompute"]
    yield
    fused_proj._STATE["enabled"] = True
    fused_proj.set_mlp_recompute(recompute)
    for owner, name, value in saved:
        if value is _MISSING:
            if name in vars(owner):
                delattr(owner, name)
        else:
            setattr(owner, name, value)


def _fused(on):
    from solution import fused_proj

    fused_proj._STATE["enabled"] = on


def _linear4bit(k, n, seed, bias=False):
    """A bnb Linear4bit (NF4, double quant, fp16 compute) with a random weight and an fp32 bias, as
    prepare_model_for_kbit_training leaves it."""
    import bitsandbytes as bnb

    g = torch.Generator().manual_seed(seed)
    layer = bnb.nn.Linear4bit(k, n, bias=bias, compute_dtype=torch.float16, quant_type="nf4",
                              compress_statistics=True)
    weight = (torch.randn(n, k, generator=g) * 0.02).half()
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4",
                                     compress_statistics=True, module=layer)
    if bias:
        layer.bias = torch.nn.Parameter(torch.randn(n, generator=g) * 0.1, requires_grad=False)
    return layer.cuda()


def _lora(base, seed, dropout=0.0, r=16, alpha=32):
    """peft's LoRA Linear4bit around `base` with random A and B (B is zero right after init)."""
    from peft import LoraConfig
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=["q_proj", "v_proj"])
    layer = LoraLinear4bit(base, "default", config=config, r=r, lora_alpha=alpha)
    # fp32 adapters, as get_peft_model leaves them (autocast_adapter_dtype)
    layer.lora_A.float()
    layer.lora_B.float()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for w in (layer.lora_A["default"].weight, layer.lora_B["default"].weight):
            w.copy_(torch.randn(w.shape, generator=g) * 0.05)
    return layer


def _config(hidden=1536, inter=8960, heads=12, kv=2, layers=1):
    from transformers import Qwen2Config

    return Qwen2Config(vocab_size=512, hidden_size=hidden, intermediate_size=inter,
                       num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
                       max_position_embeddings=4096, rope_theta=1_000_000.0,
                       attn_implementation="sdpa")


def _quantize_attention(attn, seed, dropout=0.0):
    hidden = attn.q_proj.in_features
    attn.q_proj = _lora(_linear4bit(hidden, attn.q_proj.out_features, seed, bias=True), seed + 1,
                        dropout)
    attn.k_proj = _linear4bit(hidden, attn.k_proj.out_features, seed + 2, bias=True)
    attn.v_proj = _lora(_linear4bit(hidden, attn.v_proj.out_features, seed + 3, bias=True), seed + 4,
                        dropout)
    attn.o_proj = _linear4bit(attn.o_proj.in_features, hidden, seed + 5)
    return attn


def _quantize_mlp(mlp, seed):
    hidden, inter = mlp.gate_proj.in_features, mlp.gate_proj.out_features
    mlp.gate_proj = _linear4bit(hidden, inter, seed)
    mlp.up_proj = _linear4bit(hidden, inter, seed + 1)
    mlp.down_proj = _linear4bit(inter, hidden, seed + 2)
    return mlp


def _spans(lengths):
    ends = torch.tensor(lengths).cumsum(0).tolist()
    return [(end - length, end) for end, length in zip(ends, lengths)]


def _positions(lengths):
    return torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).cuda()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _lora_params(module):
    return [p for n, p in module.named_parameters() if "lora_" in n]


# ------------------------------------------------------------------------------------------------
# a tiny 4-bit Qwen2 with peft through the packed model forward


def _tiny_peft_model(dropout=0.0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(0)
    config = _config(hidden=64, inter=160, heads=4, kv=2, layers=2)
    model = Qwen2ForCausalLM(config).cuda().float()
    for i, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        hidden = attn.q_proj.in_features
        attn.q_proj = _linear4bit(hidden, attn.q_proj.out_features, 10 * i, bias=True)
        attn.k_proj = _linear4bit(hidden, attn.k_proj.out_features, 10 * i + 1, bias=True)
        attn.v_proj = _linear4bit(hidden, attn.v_proj.out_features, 10 * i + 2, bias=True)
        attn.o_proj = _linear4bit(attn.o_proj.in_features, hidden, 10 * i + 3)
        _quantize_mlp(mlp, 10 * i + 4)
    model.is_loaded_in_4bit = True  # what peft reads to wrap bnb layers with its Linear4bit
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout,
                             target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, peft_config)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in _lora_params(model):
            p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    model.train()
    return model


def _tiny_step(model, ids, lengths):
    from solution.unpad import varlen_model_forward

    for p in model.parameters():
        p.grad = None
    inner = model.base_model.model
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = varlen_model_forward(inner.model, ids, _positions(lengths), _spans(lengths))
        logits = inner.lm_head(hidden).float()
    loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
    # fp16 training runs with the GradScaler (initial scale 65536): without it the half adapter
    # gradients of this tiny model sit in the fp16 subnormals in both paths
    (loss * 65536.0).backward()
    return hidden.detach(), loss.detach(), [p.grad.clone() / 65536.0 for p in _lora_params(model)]


def test_tiny_peft_model_matches_unfused(monkeypatch):
    from peft.tuners.lora.bnb import Linear4bit as LoraLinear4bit

    from solution import fused_proj

    model = _tiny_peft_model()
    attn = model.base_model.model.model.layers[0].self_attn
    assert type(attn.q_proj) is LoraLinear4bit and type(attn.v_proj) is LoraLinear4bit
    # more than 128 tokens: below that bitsandbytes runs its own small-M GEMM and the patch steps back
    lengths = [200, 151, 300, 177]
    ids = torch.randint(0, 512, (1, sum(lengths)), device="cuda", generator=torch.Generator("cuda").manual_seed(2))
    _fused(False)
    ref_hidden, ref_loss, ref_grads = _tiny_step(model, ids, lengths)
    _fused(True)
    calls = []
    fused_attention = fused_proj._fused_attention
    monkeypatch.setattr(fused_proj, "_fused_attention",
                        lambda *a: calls.append(1) or fused_attention(*a))
    hidden, loss, grads = _tiny_step(model, ids, lengths)
    assert len(calls) == 2  # both layers took the fused path
    assert _rel(hidden, ref_hidden) < 1e-2
    assert abs(loss.item() - ref_loss.item()) < 1e-2 * abs(ref_loss.item())
    assert len(grads) == len(ref_grads) == 8
    errors = [_rel(g, ref) for g, ref in zip(grads, ref_grads)]
    assert max(errors) < 1e-2, errors


# ------------------------------------------------------------------------------------------------
# the layers at the shapes of the task (Qwen2.5-1.5B)

LENGTHS = [212, 187, 310, 117]  # 826 tokens in 4 rows


@pytest.fixture(scope="module")
def layer_1p5b():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    config = _config()
    attn = _quantize_attention(mq.Qwen2Attention(config, 0), seed=3).cuda()
    mlp = _quantize_mlp(mq.Qwen2MLP(config), seed=20).cuda()
    rotary = mq.Qwen2RotaryEmbedding(config=config).cuda()
    attn.train()
    mlp.train()
    return config, attn, mlp, rotary


def _attention_inputs(rotary, seed=5, lengths=LENGTHS):
    g = torch.Generator("cuda").manual_seed(seed)
    x = torch.randn(1, sum(lengths), 1536, device="cuda", generator=g)
    cos, sin = rotary(x, _positions(lengths))
    grad = torch.randn(1, sum(lengths), 1536, device="cuda", generator=g, dtype=torch.float16)
    return x, (cos, sin), grad


def _attention_step(attn, x, position_embeddings, grad, lengths=LENGTHS):
    from solution.unpad import varlen

    x = x.detach().requires_grad_(True)
    for p in attn.parameters():
        p.grad = None
    with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(lengths)):
        out, _ = attn(x, position_embeddings, None)
    out.backward(grad)
    return out.detach(), x.grad, [p.grad.clone() for p in _lora_params(attn)]


def test_attention_1p5b_matches_unfused(layer_1p5b):
    _, attn, _, rotary = layer_1p5b
    x, pe, grad = _attention_inputs(rotary)
    _fused(False)
    ref_out, ref_dx, ref_grads = _attention_step(attn, x, pe, grad)
    _fused(True)
    out, dx, grads = _attention_step(attn, x, pe, grad)
    assert out.dtype == ref_out.dtype == torch.float16 and dx.dtype == torch.float32
    assert _rel(out, ref_out) < 1e-2
    assert _rel(dx, ref_dx) < 2e-2
    assert len(grads) == 4
    for g, ref in zip(grads, ref_grads):
        assert _rel(g, ref) < 2e-2


def test_attention_single_row(layer_1p5b):
    _, attn, _, rotary = layer_1p5b
    lengths = [300]
    x, pe, grad = _attention_inputs(rotary, seed=6, lengths=lengths)
    _fused(False)
    ref = _attention_step(attn, x, pe, grad, lengths)
    _fused(True)
    got = _attention_step(attn, x, pe, grad, lengths)
    for a, b in zip([got[0], got[1]] + got[2], [ref[0], ref[1]] + ref[2]):
        assert _rel(a, b) < 2e-2


def test_attention_1p5b_with_alignment_tail(layer_1p5b):
    """Spans that end before the tokens (unpad.varlen_model_forward's alignment tail): the fused
    path takes them, matches the unfused path, the tail rows stay finite and, with no loss on them,
    get an input gradient of exactly zero."""
    from solution import fused_proj
    from solution.unpad import varlen

    _, attn, _, rotary = layer_1p5b
    tail = 10
    x, pe, grad = _attention_inputs(rotary, seed=11, lengths=LENGTHS + [tail])  # tail at 0..9
    grad[:, -tail:] = 0

    def step(fused):
        _fused(fused)
        xg = x.detach().requires_grad_(True)
        for p in attn.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(LENGTHS)):
            plan = fused_proj._attention_plan(attn, xg, pe, None, None)
            assert (plan is not None and plan.bounds[-2:] == (826, 836)) if fused else plan is None
            out, _ = attn(xg, pe, None)
        out.backward(grad)
        return out.detach(), xg.grad, [p.grad.clone() for p in _lora_params(attn)]

    ref_out, ref_dx, ref_grads = step(False)
    out, dx, grads = step(True)
    assert torch.isfinite(out).all() and torch.isfinite(dx).all()
    assert torch.all(dx[:, -tail:] == 0) and torch.all(ref_dx[:, -tail:] == 0)
    assert _rel(out, ref_out) < 1e-2
    assert _rel(dx, ref_dx) < 2e-2
    for g, ref in zip(grads, ref_grads):
        assert _rel(g, ref) < 2e-2


def _mlp_step(mlp, x, grad):
    x = x.detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = mlp(x)
    out.backward(grad.to(out.dtype))
    return out.detach(), x.grad


@pytest.mark.parametrize("split", [False, True])
def test_mlp_1p5b_matches_unfused(layer_1p5b, monkeypatch, split):
    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1" if split else "0")
    _, _, mlp, _ = layer_1p5b
    g = torch.Generator("cuda").manual_seed(7)
    x = torch.randn(1, 826, 1536, device="cuda", generator=g)
    grad = torch.randn(1, 826, 1536, device="cuda", generator=g, dtype=torch.float16)
    _fused(False)
    ref_out, ref_dx = _mlp_step(mlp, x, grad)
    _fused(True)
    out, dx = _mlp_step(mlp, x, grad)
    assert ref_out.dtype == torch.float32 and out.dtype == torch.float16  # same values, half storage
    assert dx.dtype == torch.float32
    assert _rel(out, ref_out) < 1e-2
    assert _rel(dx, ref_dx) < 2e-2


# ------------------------------------------------------------------------------------------------
# MLP memory: x16 saved and (gate|up) recomputed against (gate|up) saved


@pytest.fixture
def mlp_recompute():
    """fused_proj.set_mlp_recompute, put back as it was after the test."""
    from solution import fused_proj

    before = fused_proj._MLP_MEMORY["recompute"]
    yield fused_proj.set_mlp_recompute
    fused_proj.set_mlp_recompute(before)


def _saved_mlp_step(mlp, x, grad):
    """_mlp_step with the shapes and dtypes of every tensor the forward saves."""
    saved = []

    def pack(t):
        saved.append((tuple(t.shape), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out, dx = _mlp_step(mlp, x, grad)
    return out, dx, saved


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tokens,dtype", [(826, torch.float32), (1427, torch.float32),
                                          (832, torch.float32), (1427, torch.float16)])
def test_mlp_recompute_matches_saved_bit_for_bit(layer_1p5b, monkeypatch, mlp_recompute, split,
                                                 tokens, dtype):
    """The recompute mode gives the same output and input gradient, bit for bit, as the mode that
    saves (gate|up) (offload / none), at the task's shapes, token counts that are and are not a
    multiple of 8, an fp32 (the RMSNorm output in training) or half input, under fp16 autocast."""
    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1" if split else "0")
    _, _, mlp, _ = layer_1p5b
    g = torch.Generator("cuda").manual_seed(tokens)
    x = torch.randn(1, tokens, 1536, device="cuda", generator=g).to(dtype)
    grad = torch.randn(1, tokens, 1536, device="cuda", generator=g, dtype=torch.float16)
    padded = -(-tokens // 8) * 8
    _fused(True)
    mlp_recompute(False)
    ref_out, ref_dx, ref_saved = _saved_mlp_step(mlp, x, grad)
    mlp_recompute(True)
    out, dx, saved = _saved_mlp_step(mlp, x, grad)
    assert ref_saved == [((2 * 8960, padded), torch.float16)]
    assert saved == [((padded, 1536), torch.float16)]  # x16 with its zero padding rows
    assert out.dtype == torch.float16 and dx.dtype == dtype
    assert torch.equal(out, ref_out)
    assert torch.equal(dx, ref_dx)


def test_mlp_recompute_twice_with_retained_graph(layer_1p5b, mlp_recompute):
    """The recompute backward writes the SwiGLU gradient over its own recomputed (gate|up), never
    over anything saved: a second backward through a retained graph gives the same gradient."""
    _, _, mlp, _ = layer_1p5b
    g = torch.Generator("cuda").manual_seed(12)
    x = torch.randn(1, 300, 1536, device="cuda", generator=g).requires_grad_(True)
    grad = torch.randn(1, 300, 1536, device="cuda", generator=g, dtype=torch.float16)
    _fused(True)
    mlp_recompute(True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = mlp(x)
    (first,) = torch.autograd.grad(out, x, grad, retain_graph=True)
    (second,) = torch.autograd.grad(out, x, grad)
    assert torch.equal(first, second)


def test_tiny_peft_model_recompute_matches_saved(mlp_recompute):
    """Through the packed model forward of a tiny 4-bit Qwen2 with peft: hidden states, loss and
    the LoRA gradients are bit for bit the same with the MLP recomputed as with (gate|up) saved.
    SDPA runs its math backend here: the mem-efficient backward accumulates dq with atomics, so its
    rounding (and the LoRA gradients below it) differs from run to run in either mode."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model = _tiny_peft_model()
    lengths = [200, 151, 301, 177]  # 829 tokens, not a multiple of 8
    ids = torch.randint(0, 512, (1, sum(lengths)), device="cuda",
                        generator=torch.Generator("cuda").manual_seed(3))
    _fused(True)
    with sdpa_kernel(SDPBackend.MATH):
        mlp_recompute(False)
        ref_hidden, ref_loss, ref_grads = _tiny_step(model, ids, lengths)
        mlp_recompute(True)
        hidden, loss, grads = _tiny_step(model, ids, lengths)
    assert torch.equal(hidden, ref_hidden) and torch.equal(loss, ref_loss)
    assert len(grads) == len(ref_grads) == 8
    assert all(torch.equal(a, b) for a, b in zip(grads, ref_grads))


# ------------------------------------------------------------------------------------------------
# LoRA dropout: two masks, drawn again in the backward


def _reference_qkv(attn, x, cos, sin, masks, p):
    """q, k, v of the fused function computed with autograd in fp32 from the dequantized weights and
    the given dropout masks (one per adapter)."""
    import bitsandbytes.functional as bf
    from transformers.models.qwen2 import modeling_qwen2 as mq

    def weight(layer):
        return bf.dequantize_4bit(layer.weight.data, layer.weight.quant_state).float()

    x16 = x.half().float()
    outs = []
    for proj, mask in ((attn.q_proj, masks[0]), (attn.k_proj, None), (attn.v_proj, masks[1])):
        base = proj.base_layer if mask is not None else proj
        y = x16 @ weight(base).t() + base.bias.half().float()
        if mask is not None:
            a, b = proj.lora_A["default"].weight, proj.lora_B["default"].weight
            dropped = x16 * mask / (1 - p)
            y = y + (dropped @ a.half().float().t()) @ b.half().float().t() * proj.scaling["default"]
        outs.append(y.view(1, x.shape[1], -1, 128).transpose(1, 2))
    q, k = mq._soup_original_apply_rotary_pos_emb(outs[0], outs[1], cos, sin)
    return q, k, outs[2]


def test_qkv_with_dropout_against_explicit_masks(layer_1p5b):
    from solution import fused_proj
    from solution.unpad import varlen

    p = 0.05
    _, attn, _, rotary = layer_1p5b
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(p).train()
    try:
        x, (cos, sin), _ = _attention_inputs(rotary, seed=8)
        x = x.requires_grad_(True)
        g = torch.Generator("cuda").manual_seed(9)
        grads_out = [torch.randn(1, h, 826, 128, device="cuda", generator=g, dtype=torch.float16)
                     for h in (12, 2, 2)]
        with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(LENGTHS)):
            plan = fused_proj._attention_plan(attn, x, (cos, sin), None, None)
            assert plan is not None and plan.p == p
            state = torch.cuda.get_rng_state()
            a_q, b_q = attn.q_proj.lora_A["default"].weight, attn.q_proj.lora_B["default"].weight
            a_v, b_v = attn.v_proj.lora_A["default"].weight, attn.v_proj.lora_B["default"].weight
            q, k, v = fused_proj._QKVRope.apply(x, a_q, b_q, a_v, b_v, cos, sin, plan)
        after_forward = torch.cuda.get_rng_state()
        params = [x, a_q, b_q, a_v, b_v]
        got = torch.autograd.grad((q, k, v), params, grads_out)
        # the backward leaves the RNG where the forward left it
        assert torch.equal(torch.cuda.get_rng_state(), after_forward)

        torch.cuda.set_rng_state(state)
        _, masks = torch.native_dropout(x.detach().half().expand(2, 826, 1536), p, True)
        assert not torch.equal(masks[0], masks[1])  # one mask per adapter
        keep = masks.float().mean().item()
        assert abs(keep - (1 - p)) < 0.005
        ref = _reference_qkv(attn, x, cos, sin, masks.float(), p)
        for a, b in zip((q, k, v), ref):
            assert _rel(a, b) < 1e-2
        ref_grads = torch.autograd.grad(ref, params, [t.float() for t in grads_out])
        for a, b in zip(got, ref_grads):
            assert a.dtype == b.dtype == torch.float32
            assert _rel(a, b) < 2e-2
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()



@pytest.mark.parametrize("fused_norm", [False, True])
def test_qkv_rebuilds_x16_from_the_input_norm(layer_1p5b, fused_norm):
    """x straight out of rmsnorm.py's input norm (plain, or fused with the carried residual add):
    _QKVRope keeps no x16 and rebuilds it in the backward from the norm's saved input and rstd; q,
    k, v and every gradient are bitwise those of keeping it, LoRA dropout redrawn included. (The
    function alone: the attention backward after it is not bit-reproducible run to run, dq sums
    key blocks in any order, per-row SDPA's as well.)"""
    from solution import fused_proj, rmsnorm
    from solution.unpad import varlen

    p = 0.05
    _, attn, _, rotary = layer_1p5b
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(p).train()
    g = torch.Generator("cuda").manual_seed(12)
    weight = 1.0 + 0.3 * torch.randn(1536, device="cuda", generator=g)
    branch = torch.randn(1, 826, 1536, device="cuda", generator=g).half()
    grads_out = [torch.randn(1, heads, 826, 128, device="cuda", generator=g, dtype=torch.float16)
                 for heads in (12, 2, 2)]
    h, (cos, sin), _ = _attention_inputs(rotary, seed=13)
    a_q, b_q = attn.q_proj.lora_A["default"].weight, attn.q_proj.lora_B["default"].weight
    a_v, b_v = attn.v_proj.lora_A["default"].weight, attn.v_proj.lora_B["default"].weight
    results = []
    try:
        for rebuild in (False, True):
            fused_proj._ATTN["rebuild_x16"] = rebuild
            hg = h.detach().requires_grad_(True)
            saved = []

            def pack(t):
                saved.append(t.numel() * t.element_size())
                return t

            torch.cuda.manual_seed(21)  # the same dropout masks in both runs
            with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(LENGTHS)):
                if fused_norm:
                    _, x = rmsnorm.add_rmsnorm(hg, branch, weight, 1e-6)
                else:
                    x = rmsnorm.rmsnorm(hg, weight, 1e-6)
                plan = fused_proj._attention_plan(attn, x, (cos, sin), None, None)
                assert plan is not None and plan.p == p
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    qkv = fused_proj._QKVRope.apply(x, a_q, b_q, a_v, b_v, cos, sin, plan)
            grads = torch.autograd.grad(qkv, [hg, a_q, b_q, a_v, b_v], grads_out)
            results.append(([t.detach() for t in qkv], grads, sum(saved)))
    finally:
        fused_proj._ATTN["rebuild_x16"] = True
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()
    (outs0, grads0, saved0), (outs1, grads1, saved1) = results
    for a, b in zip(outs1 + list(grads1), outs0 + list(grads0)):
        assert torch.equal(a, b)
    assert saved0 - saved1 == 826 * 1536 * 2  # exactly x16 less

# ------------------------------------------------------------------------------------------------
# guards


def test_guards_take_the_previous_path(layer_1p5b):
    from solution import fused_proj
    from solution.unpad import varlen

    _, attn, mlp, rotary = layer_1p5b
    x, pe, _ = _attention_inputs(rotary)
    spans = _spans(LENGTHS)

    def plan(**overrides):
        with torch.autocast("cuda", dtype=overrides.get("dtype", torch.float16)), varlen(spans):
            return fused_proj._attention_plan(attn, overrides.get("x", x), pe,
                                              overrides.get("mask"), None)

    assert plan() is not None
    assert fused_proj._attention_plan(attn, x, pe, None, None) is None  # no varlen context
    assert plan(dtype=torch.bfloat16) is None  # autocast dtype is not the compute dtype
    assert plan(mask=torch.ones(1, device="cuda")) is None
    assert plan(x=x.cpu()) is None
    assert plan(x=x[:, :100]) is None  # spans past the tokens (a tail after them is fine)
    attn.q_proj._disable_adapters = True
    assert plan() is None
    attn.q_proj._disable_adapters = False
    attn.v_proj.merged_adapters.append("default")
    assert plan() is None
    attn.v_proj.merged_adapters.clear()
    attn.q_proj.lora_dropout["default"] = torch.nn.Dropout(0.1, inplace=True)
    assert plan() is None
    attn.q_proj.lora_dropout["default"] = torch.nn.Dropout(0.1)  # p differs from v's adapter
    assert plan() is None
    attn.q_proj.lora_dropout["default"] = torch.nn.Identity()
    handle = attn.k_proj.register_forward_hook(lambda *a: None)
    assert plan() is None
    handle.remove()
    k_proj = attn.k_proj
    attn.k_proj = _lora(k_proj, seed=40)  # an adapter on k as well
    assert plan() is None
    attn.k_proj = k_proj
    assert plan() is not None

    with torch.autocast("cuda", dtype=torch.float16):
        assert fused_proj._mlp_plan(mlp, x) is not None
        assert fused_proj._mlp_plan(mlp, x[:, :64]) is None  # bitsandbytes' small-M GEMM
        assert fused_proj._mlp_plan(mlp, x.cpu()) is None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert fused_proj._mlp_plan(mlp, x) is None


def test_outside_varlen_runs_previous_forward(layer_1p5b):
    """A padded batch with a mask (eval, generation) runs the attention it ran before."""
    _, attn, _, rotary = layer_1p5b
    x = torch.randn(2, 40, 1536, device="cuda")
    cos, sin = rotary(x, torch.arange(40, device="cuda").expand(2, 40))
    mask = torch.zeros(2, 1, 40, 40, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        _fused(False)
        ref, _ = attn(x, (cos, sin), mask)
        _fused(True)
        out, _ = attn(x, (cos, sin), mask)
    assert torch.equal(out, ref)


def test_patches_are_idempotent_and_keep_previous():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj

    for cls in (mq.Qwen2Attention, mq.Qwen2MLP):
        forward, previous = cls.forward, cls._soup_fused_proj_previous
        fused_proj.patch_qwen2_attention_fused()
        fused_proj.patch_qwen2_mlp_fused()
        assert cls.forward is forward and cls._soup_fused_proj_previous is previous
        assert previous is not forward
    assert mq.Qwen2MLP._soup_fused_proj_previous.__qualname__.startswith("patch_qwen2_mlp_half")
    assert mq.Qwen2Attention._soup_fused_proj_previous.__qualname__.startswith("patch_qwen2_attention")


# ------------------------------------------------------------------------------------------------
# kernel launches and saved memory per layer (numbers printed with -s)


def _kernels(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    count = sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)
    return count, result


def _attention_forward(attn, x, pe):
    from solution.unpad import varlen

    with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(LENGTHS)):
        out, _ = attn(x, pe, None)
    return out


def _mlp_forward(mlp, x):
    with torch.autocast("cuda", dtype=torch.float16):
        return mlp(x)


def _launches_and_memory(module, forward, x, grad):
    for _ in range(2):  # warm up (Triton compiles, cuBLAS workspaces)
        out = forward(module, x.detach().requires_grad_(True))
        out.backward(grad.to(out.dtype))
    xg = x.detach().requires_grad_(True)
    n_fwd, out = _kernels(lambda: forward(module, xg))
    n_bwd, _ = _kernels(lambda: out.backward(grad.to(out.dtype)))
    del out
    # in the model the layer input is a norm output, not a leaf (autocast caches casts of leaves only)
    inp = xg * 1.0
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = forward(module, inp)
    torch.cuda.synchronize()
    saved = torch.cuda.memory_allocated() - before - out.numel() * out.element_size()
    out.backward(grad.to(out.dtype))
    return n_fwd, n_bwd, saved


def test_launches_and_saved_memory_per_layer(layer_1p5b):
    from solution import fused_proj

    _, attn, mlp, rotary = layer_1p5b
    x, pe, grad = _attention_inputs(rotary)
    rows = {}
    for name, module, forward, g in (
            ("attention", attn, lambda m, t: _attention_forward(m, t, pe), grad),
            ("mlp", mlp, _mlp_forward, grad)):
        for on in (False, True):
            _fused(on)
            rows[name, on] = _launches_and_memory(module, forward, x, g)
    _fused(True)
    print()
    for name in ("attention", "mlp"):
        (f0, b0, m0), (f1, b1, m1) = rows[name, False], rows[name, True]
        print(f"{name}: launches fwd {f0} -> {f1}, bwd {b0} -> {b1}; "
              f"saved {m0 / 2**20:.1f} MB -> {m1 / 2**20:.1f} MB")
        # gate and up split over a half-size buffer (the default) rebuild and multiply twice: the
        # MLP's forward then launches as many kernels as mlp16's two projections (9 and 9)
        slack = 1 if name == "mlp" and fused_proj._gate_up_split() else 0
        assert f1 < f0 + slack and b1 < b0
        # the MLP keeps (gate|up) with the token axis padded to a multiple of 8 (GEMM alignment):
        # at most 7 columns of 2 x 8960 halves more than mlp16's gate and up
        assert m1 <= m0 + (2 * 8960 * 7 * 2 if name == "mlp" else 0)


# ------------------------------------------------------------------------------------------------
# x16 rebuilt from the norm in the MLP's recompute mode, the adapters' masked gradient sum


@pytest.mark.parametrize("tokens", [832, 826])
def test_mlp_recompute_rebuilds_x16_from_the_post_attention_norm(layer_1p5b, mlp_recompute, tokens):
    """x straight out of the post-attention add + norm, as the decoder layer feeds the MLP: in recompute
    mode the MLP keeps nothing and rebuilds x16 from the norm's saved sum and rstd; the output and
    every gradient are bitwise those of keeping x16. With a token count that is not a multiple of 8
    (x16 gets zero padding rows; not after unpad's alignment tail) it keeps x16 as before."""
    from solution import fused_proj, rmsnorm

    _, _, mlp, _ = layer_1p5b
    g = torch.Generator("cuda").manual_seed(31)
    residual = torch.randn(1, tokens, 1536, device="cuda", generator=g)
    branch = torch.randn(1, tokens, 1536, device="cuda", generator=g).half()
    weight = 1.0 + 0.3 * torch.randn(1536, device="cuda", generator=g)
    grad = torch.randn(1, tokens, 1536, device="cuda", generator=g, dtype=torch.float16)
    _fused(True)
    mlp_recompute(True)
    results = []
    try:
        for rebuild in (False, True):
            fused_proj._ATTN["rebuild_x16"] = rebuild
            r = residual.detach().requires_grad_(True)
            saved = []

            def pack(t):
                saved.append((tuple(t.shape), t.dtype))
                return t

            with torch.autocast("cuda", dtype=torch.float16):
                _, x = rmsnorm.add_rmsnorm(r, branch, weight, 1e-6)
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                    out = mlp(x)
            out.backward(grad)
            results.append((out.detach(), r.grad, saved))
    finally:
        fused_proj._ATTN["rebuild_x16"] = True
    (out0, dr0, saved0), (out1, dr1, saved1) = results
    assert torch.equal(out1, out0) and torch.equal(dr1, dr0)
    padded = -(-tokens // 8) * 8
    assert saved0 == [((padded, 1536), torch.float16)]
    assert saved1 == ([] if padded == tokens else saved0)


def test_masked_pair_sum_matches_mul_addcmul():
    """The adapters' share of the input gradient through the two dropout masks, one launch: bitwise
    torch's mul (half by bool into fp32) then addcmul_, signed zeros included."""
    from solution.fused_proj import _masked_pair_sum

    g = torch.Generator("cuda").manual_seed(32)
    grads = torch.randn(2, 838, 1536, device="cuda", generator=g).half()
    grads[:, :5] = 0.0
    grads[:, 5:9] = -0.0
    mask = torch.rand(2, 838, 1536, device="cuda", generator=g) > 0.05
    ref = torch.mul(grads[0], mask[0], out=torch.empty(838, 1536, device="cuda"))
    ref.addcmul_(grads[1], mask[1])
    out = _masked_pair_sum(grads, mask)
    assert out.dtype == torch.float32 and out.shape == (838, 1536)
    assert torch.equal(out, ref) and torch.equal(torch.signbit(out), torch.signbit(ref))


def test_attention_kernels_are_warmed_up():
    """The warm-ups fused_proj registers with warmup.py compile the classes training launches."""
    from solution import fused_proj, warmup

    compiles = warmup.watch_compiles(log=False)
    fused_proj.masked_pair_sum_kernel.device_caches.clear()
    compiles.clear()
    fused_proj._warm_masked_pair_sum()
    assert [c["kernel"] for c in compiles] == ["masked_pair_sum_kernel"]
    compiles.clear()
    for n in (838, 1440):
        grads = torch.randn(2, n, 1536, device="cuda").to(warmup.training_half_dtype())
        fused_proj._masked_pair_sum(grads, torch.rand(2, n, 1536, device="cuda") > 0.05)
    assert [c for c in compiles if c["kernel"] == "masked_pair_sum_kernel"] == []
    names = [name for name, _ in warmup.WARMUPS]
    assert "fused_proj_masked_pair_sum" in names and "fused_proj_x16_rebuild" in names


# ------------------------------------------------------------------------------------------------
# CPU side: q/k/v dequantized in one launch, nf4's checks kept per packed weight, the adapters'
# half copies refreshed for every layer in one launch per step


@pytest.fixture
def cpu_diet():
    from solution import fused_proj

    before = fused_proj._CPU["diet"]
    yield fused_proj._CPU
    fused_proj._CPU["diet"] = before


def test_dequant3_matches_three_dequantizations():
    from solution import fused_proj
    from solution.nf4 import dequantize_nf4

    layers = [_linear4bit(1536, rows, seed) for rows, seed in ((1536, 60), (256, 61), (256, 62))]
    packed = [(layer.weight.data, layer.weight.quant_state) for layer in layers]
    out = torch.empty(2048, 1536, dtype=torch.float16, device="cuda")
    fused_proj._dequant3(packed, out)
    ref = torch.cat([dequantize_nf4(layer.weight, layer.weight.quant_state, torch.float16)
                     for layer in layers])
    assert torch.equal(out, ref)


def test_weights_are_the_same_with_the_cpu_diet(layer_1p5b, cpu_diet):
    from solution import fused_proj

    _, attn, mlp, _ = layer_1p5b
    groups = [(attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer), (attn.o_proj,),
              (mlp.gate_proj, mlp.up_proj), (mlp.down_proj,)]
    for bases in groups:
        cpu_diet["diet"] = False
        ref = fused_proj._weights(bases, torch.float16, torch.device("cuda")).clone()
        cpu_diet["diet"] = True
        got = fused_proj._weights(bases, torch.float16, torch.device("cuda"))
        assert torch.equal(got, ref), [b.out_features for b in bases]
        assert all("_soup_nf4_args" in vars(b) and vars(b)["_soup_nf4_args"][1] for b in bases)


def test_adapter_halves_refresh_every_layer_in_one_copy(monkeypatch):
    from solution import fused_proj

    cache = fused_proj._AdapterHalves()
    owners = [torch.nn.Linear(1, 1) for _ in range(3)]
    params = [tuple(torch.randn(shape, device="cuda", requires_grad=True)
                    for shape in ((16, 64), (16, 64), (96, 16), (32, 16))) for _ in owners]
    copies = []
    foreach = torch._foreach_copy_
    monkeypatch.setattr(torch, "_foreach_copy_", lambda dst, src: (copies.append(len(dst)),
                                                                   foreach(dst, src)))

    def check(owner, ps):
        a16, b16, _ = cache.halves(owner, ps, torch.float16)
        assert torch.equal(a16, torch.cat([ps[0], ps[1]]).half())
        assert torch.equal(b16, torch.cat([ps[2], ps[3]]).half())

    for owner, ps in zip(owners, params):  # first step: every layer fills its own
        check(owner, ps)
    assert copies == [4, 4, 4]
    copies.clear()
    for owner, ps in zip(owners, params):  # nothing changed: nothing copied
        check(owner, ps)
    assert copies == []
    with torch.no_grad():  # the optimizer step
        for ps in params:
            for p in ps:
                p.add_(0.25)
    for owner, ps in zip(owners, params):
        check(owner, ps)
    assert copies == [12]  # all three layers in the first layer's one copy


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_attention_is_the_same_with_the_cpu_diet(layer_1p5b, cpu_diet, dropout):
    """The fused attention with the q/k/v launch and the adapter cache against the per-weight launches
    and per-layer casts: bitwise, across an optimizer-like change of the adapters."""
    _, attn, _, rotary = layer_1p5b
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(dropout).train() if dropout else \
            torch.nn.Identity()
    x, pe, grad = _attention_inputs(rotary, seed=33)
    _fused(True)
    try:
        for change in range(2):
            results = []
            for diet in (False, True):
                cpu_diet["diet"] = diet
                torch.cuda.manual_seed(40 + change)
                results.append(_attention_step(attn, x, pe, grad))
            (out0, dx0, g0), (out1, dx1, g1) = results
            assert torch.equal(out1, out0)
            # the attention backward's dq is not bit-reproducible run to run (key-split sums)
            assert _rel(dx1, dx0) < 1e-3
            for a, b in zip(g1, g0):
                assert _rel(a, b) < 1e-3
            with torch.no_grad():
                for p in _lora_params(attn):
                    p.mul_(0.9)
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()


def test_dequant4_kernel_is_warmed_up(layer_1p5b, cpu_diet):
    from types import SimpleNamespace

    from solution import fused_proj, warmup

    _, attn, _, _ = layer_1p5b
    cpu_diet["diet"] = True
    compiles = warmup.watch_compiles(log=False)
    fused_proj.nf4_dequant4_kernel.device_caches.clear()
    compiles.clear()
    fused_proj._warm_dequant4()
    assert [c["kernel"] for c in compiles] == ["nf4_dequant4_kernel"]
    compiles.clear()
    plan = SimpleNamespace(half=warmup.training_half_dtype(), o_base=attn.o_proj,
                           bases=(attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer))
    fused_proj._attn_weights(plan, torch.device("cuda"))
    assert [c for c in compiles if c["kernel"] == "nf4_dequant4_kernel"] == []
    assert "fused_proj_nf4_dequant4" in [name for name, _ in warmup.WARMUPS]


def test_attention_weights_one_launch_and_reuse(layer_1p5b, cpu_diet, monkeypatch):
    """q, k, v and o in one launch, bitwise the four dequantizations; asked again with nothing
    written into the buffer since: no launch, the same views; after another weight went into the
    buffer: launched again. _weights skips a repeat of exactly what the buffer holds the same way."""
    from types import SimpleNamespace

    from solution import fused_proj
    from solution.nf4 import dequantize_nf4

    _, attn, mlp, _ = layer_1p5b
    cpu_diet["diet"] = True
    launches = []
    many, one = fused_proj._launch_many, fused_proj._launch_one
    monkeypatch.setattr(fused_proj, "_launch_many", lambda ops, out: (launches.append(len(ops)),
                                                                     many(ops, out)))
    monkeypatch.setattr(fused_proj, "_launch_one", lambda ops, out: (launches.append(1), one(ops, out)))
    bases = (attn.q_proj.base_layer, attn.k_proj, attn.v_proj.base_layer)
    plan = SimpleNamespace(half=torch.float16, bases=bases, o_base=attn.o_proj)
    device = torch.device("cuda")
    ref_qkv = torch.cat([dequantize_nf4(b.weight, b.weight.quant_state, torch.float16) for b in bases])
    ref_o = dequantize_nf4(attn.o_proj.weight, attn.o_proj.weight.quant_state, torch.float16)
    fused_proj._buffer(device, torch.float16, 16)  # whatever an earlier test left in the buffer
    qkv, o = fused_proj._attn_weights(plan, device)
    assert launches == [4] and torch.equal(qkv, ref_qkv) and torch.equal(o, ref_o)
    again = fused_proj._attn_weights(plan, device)
    assert launches == [4] and again[0] is qkv and again[1] is o
    down = fused_proj._weights((mlp.down_proj,), torch.float16, device)
    assert launches == [4, 1]
    assert fused_proj._weights((mlp.down_proj,), torch.float16, device) is down and launches == [4, 1]
    assert torch.equal(down, dequantize_nf4(mlp.down_proj.weight, mlp.down_proj.weight.quant_state,
                                            torch.float16))
    qkv, o = fused_proj._attn_weights(plan, device)
    assert launches == [4, 1, 4] and torch.equal(qkv, ref_qkv) and torch.equal(o, ref_o)
    fused_proj._buffer(device, torch.float16, 16)  # any other writer of the buffer
    fused_proj._attn_weights(plan, device)
    assert launches == [4, 1, 4, 4]
    cpu_diet["diet"] = False
    assert fused_proj._attn_weights(plan, device) is None


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_attention_one_function_is_bitwise(layer_1p5b, monkeypatch, dropout):
    """A recomputing layer's attention as one autograd node (SOUP_SOLUTION_ATTN_ONE_FUNCTION) against
    the three nodes it replaces: output, input gradient and LoRA gradients bit for bit (one key
    split in the attention backward, so that dq is reproducible run to run)."""
    from solution import fused_proj, unpad

    _, attn, _, rotary = layer_1p5b
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "all")
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(dropout).train() if dropout else \
            torch.nn.Identity()
    monkeypatch.setitem(fused_proj._ATTN, "two_nodes", False)
    x, pe, grad = _attention_inputs(rotary, seed=44)
    _fused(True)
    nodes = []
    try:
        results = []
        for one in (False, True):
            monkeypatch.setitem(fused_proj._ATTN, "one_function", one)
            torch.cuda.manual_seed(7)
            with torch.autocast("cuda", dtype=torch.float16), unpad.varlen(_spans(LENGTHS)):
                probe, _ = attn(x.detach().requires_grad_(True), pe, None)
            nodes.append(type(probe.grad_fn.next_functions[0][0]).__name__)  # below the view
            del probe
            torch.cuda.manual_seed(7)
            results.append(_attention_step(attn, x, pe, grad))
        (out0, dx0, g0), (out1, dx1, g1) = results
        assert torch.equal(out1, out0) and torch.equal(dx1, dx0)
        assert all(torch.equal(a, b) for a, b in zip(g1, g0))
        assert nodes == ["_Linear4bitHalfBackward", "_AttnBlockBackward"], nodes
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()


def _attention_mode(monkeypatch, mode):
    from solution import fused_proj

    monkeypatch.setitem(fused_proj._ATTN, "one_function", mode == "one")
    monkeypatch.setitem(fused_proj._ATTN, "two_nodes", mode == "two")


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_attention_two_nodes_is_bitwise(layer_1p5b, monkeypatch, dropout):
    """A recomputing layer's q/k/v and attention as one autograd node with the o_proj its own
    (SOUP_SOLUTION_ATTN_TWO_NODES) against the three nodes and the one node: output, input gradient
    and LoRA gradients bit for bit (one key split in the attention backward)."""
    from solution import fused_proj, unpad

    _, attn, _, rotary = layer_1p5b
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "all")
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(dropout).train() if dropout else \
            torch.nn.Identity()
    x, pe, grad = _attention_inputs(rotary, seed=46)
    _fused(True)
    nodes = []
    try:
        results = []
        for mode in ("three", "two", "one"):
            _attention_mode(monkeypatch, mode)
            torch.cuda.manual_seed(7)
            with torch.autocast("cuda", dtype=torch.float16), unpad.varlen(_spans(LENGTHS)):
                probe, _ = attn(x.detach().requires_grad_(True), pe, None)
            below = probe.grad_fn.next_functions[0][0]  # below the view
            nodes.append((type(below).__name__, type(below.next_functions[0][0]).__name__))
            del probe, below
            torch.cuda.manual_seed(7)
            results.append(_attention_step(attn, x, pe, grad))
        (out0, dx0, g0), *others = results
        for out1, dx1, g1 in others:
            assert torch.equal(out1, out0) and torch.equal(dx1, dx0)
            assert all(torch.equal(a, b) for a, b in zip(g1, g0))
        assert nodes[0] == ("_Linear4bitHalfBackward", "ViewBackward0"), nodes
        assert nodes[1] == ("_Linear4bitHalfBackward", "_QKVAttnBackward"), nodes
        assert nodes[2][0] == "_AttnBlockBackward", nodes
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()


def test_attention_two_nodes_backward_peak(layer_1p5b, monkeypatch):
    """The backward's allocated peak of one recomputing layer whose output gradient autograd makes
    (and frees once its consumer returns): two nodes no higher than three; one node keeps the o_proj
    output's gradient alive on top (the T4's +4.2 MiB; here +N x 2048 bytes)."""
    from solution import fused_proj, unpad

    _, attn, _, rotary = layer_1p5b
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "all")
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    x, pe, grad = _attention_inputs(rotary, seed=47)
    scale = torch.ones((), device="cuda", dtype=torch.float16, requires_grad=True)
    _fused(True)
    peaks = {}
    for mode in ("warm", "three", "two", "one", "two", "three"):
        _attention_mode(monkeypatch, mode)
        xr = x.detach().requires_grad_(True)
        for p in attn.parameters():
            p.grad = None
        with torch.autocast("cuda", dtype=torch.float16), unpad.varlen(_spans(LENGTHS)):
            out, _ = attn(xr, pe, None)
        loss = (out * scale * grad).float().sum()  # the output's gradient is made in the backward
        del out
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        assert peaks.setdefault(mode, peak) == peak or mode == "warm", (mode, peaks, peak)
        del loss, xr
    n = sum(LENGTHS)
    assert peaks["two"] <= peaks["three"], peaks
    assert peaks["one"] >= peaks["three"] + n * 1024, peaks


@pytest.mark.parametrize("mode", ["three", "two", "one"])
def test_attention_expand_kv_once_is_bitwise(layer_1p5b, monkeypatch, mode):
    """The recompute's k and v expanded once for its forward and the attention backward
    (SOUP_SOLUTION_EXPAND_KV_ONCE) against the compact ones expanded twice: output, input gradient and
    LoRA gradients bit for bit in each node layout, and the recompute's own check passes (its k and v
    equal the forward's, expanded)."""
    from solution import fused_proj, unpad

    _, attn, _, rotary = layer_1p5b
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "all")
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    _attention_mode(monkeypatch, mode)
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(0.05).train()
    x, pe, grad = _attention_inputs(rotary, seed=48)
    _fused(True)
    try:
        results = []
        for once in (False, True):
            monkeypatch.setitem(fused_proj._ATTN, "expand_once", once)
            monkeypatch.setitem(fused_proj._RECOMPUTE, "check", True)
            monkeypatch.setitem(fused_proj._RECOMPUTE, "mismatches", [])
            torch.cuda.manual_seed(9)
            results.append(_attention_step(attn, x, pe, grad))
            assert fused_proj._RECOMPUTE["mismatches"] == []
        (out0, dx0, g0), (out1, dx1, g1) = results
        assert torch.equal(out1, out0) and torch.equal(dx1, dx0)
        assert all(torch.equal(a, b) for a, b in zip(g1, g0))
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()


@pytest.mark.parametrize("mlp_recompute", [False, True])
def test_mlp_addmm_in_place_is_bitwise(layer_1p5b, monkeypatch, mlp_recompute):
    """The MLP's input gradient with its second GEMM adding into the first one's fp32 result in place
    (SOUP_SOLUTION_MLP_ADDMM_INPLACE) against the out-of-place addmm: bit for bit, both MLP modes,
    and on a token count that is not a multiple of 8 (the padded rows)."""
    from solution import fused_proj

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    monkeypatch.setitem(fused_proj._MLP_MEMORY, "recompute", mlp_recompute)
    _, _, mlp, _ = layer_1p5b
    g = torch.Generator("cuda").manual_seed(9)
    x = torch.randn(1, 821, 1536, device="cuda", generator=g)
    grad = torch.randn(1, 821, 1536, device="cuda", generator=g, dtype=torch.float16)
    _fused(True)
    results = []
    for inplace in (False, True):
        monkeypatch.setitem(fused_proj._MLP_MEMORY, "addmm_inplace", inplace)
        results.append(_mlp_step(mlp, x, grad))
    (out0, dx0), (out1, dx1) = results
    assert torch.equal(out1, out0) and torch.equal(dx1, dx0)


@pytest.mark.parametrize("mode", ["all", None])
def test_attention_fewer_views_is_bitwise(layer_1p5b, monkeypatch, mode):
    """The three-node attention with rope launched from strides and the recompute's q, k, v made
    token-major directly (SOUP_SOLUTION_FEWER_VIEWS) against the views it replaces: output, input
    gradient and LoRA gradients bit for bit, with and without the attention recompute."""
    from solution import fused_proj, unpad

    _, attn, _, rotary = layer_1p5b
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", mode)
    monkeypatch.setitem(fused_proj._ATTN, "one_function", False)
    monkeypatch.setitem(fused_proj._ATTN, "two_nodes", False)
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    for proj in (attn.q_proj, attn.v_proj):
        proj.lora_dropout["default"] = torch.nn.Dropout(0.05).train()
    x, pe, grad = _attention_inputs(rotary, seed=45)
    _fused(True)
    try:
        results = []
        for views in (False, True):
            monkeypatch.setitem(fused_proj._ATTN, "views", views)
            torch.cuda.manual_seed(8)
            results.append(_attention_step(attn, x, pe, grad))
        (out0, dx0, g0), (out1, dx1, g1) = results
        assert torch.equal(out1, out0) and torch.equal(dx1, dx0)
        assert all(torch.equal(a, b) for a, b in zip(g1, g0))
    finally:
        for proj in (attn.q_proj, attn.v_proj):
            proj.lora_dropout["default"] = torch.nn.Identity()
