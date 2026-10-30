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
    yield
    fused_proj._STATE["enabled"] = True
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
        assert f1 < f0 and b1 < b0
        # the MLP keeps (gate|up) with the token axis padded to a multiple of 8 (GEMM alignment):
        # at most 7 columns of 2 x 8960 halves more than mlp16's gate and up
        assert m1 <= m0 + (2 * 8960 * 7 * 2 if name == "mlp" else 0)
