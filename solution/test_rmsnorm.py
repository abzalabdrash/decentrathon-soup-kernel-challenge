"""The RMSNorm kernels against Qwen2RMSNorm and eager PyTorch, forward and backward, and the patched
Qwen2DecoderLayer against the original on a tiny model."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SHAPES = [(2048, 1536), (1, 826, 1536), (3, 77, 1536)]
EPS = 1e-6
# fp32 input: rounding-level differences only (the order of the sum of squares); half input: the
# normed rows are rounded to half, and a last-bit difference in rstd can flip one half ulp
TOL = {torch.float32: dict(rtol=1e-5, atol=1e-5), torch.float16: dict(rtol=2e-3, atol=2e-3)}


def _dw_tol(dtype, n_rows):
    """dweight is a sum over the rows with cancellation: the rounding error of either side's fp32
    sum grows as sqrt(rows)."""
    tol = dict(TOL[dtype])
    tol["atol"] *= n_rows ** 0.5
    return tol


def _original_norm_forward():
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    return Qwen2RMSNorm.__dict__.get("_soup_original_forward", Qwen2RMSNorm.forward)


def _norm_module(n_cols, train_weight, seed=0):
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

    torch.manual_seed(seed)
    norm = Qwen2RMSNorm(n_cols, eps=EPS).cuda()
    with torch.no_grad():
        norm.weight.copy_(1.0 + 0.5 * torch.randn(n_cols, device="cuda"))
    norm.weight.requires_grad_(train_weight)
    return norm


@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_rmsnorm_matches_qwen2(shape, dtype, train_weight):
    from solution.rmsnorm import rmsnorm

    norm = _norm_module(shape[-1], train_weight)
    torch.manual_seed(1)
    x = (3.0 * torch.randn(shape, device="cuda")).to(dtype).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn(shape, device="cuda")

    ref = _original_norm_forward()(norm, x_ref)
    ref.backward(grad)
    ref_dw = norm.weight.grad
    norm.weight.grad = None

    out = rmsnorm(x, norm.weight, EPS)
    out.backward(grad)
    assert out.dtype == ref.dtype == torch.float32
    assert out.shape == ref.shape
    assert x.grad.dtype == dtype
    tol = TOL[dtype]
    torch.testing.assert_close(out, ref, **tol)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), **tol)
    if train_weight:
        torch.testing.assert_close(norm.weight.grad, ref_dw, **_dw_tol(dtype, x.numel() // shape[-1]))
    else:
        assert norm.weight.grad is None


def test_dweight_is_deterministic():
    from solution.rmsnorm import rmsnorm

    norm = _norm_module(1536, train_weight=True)
    x = torch.randn(2048, 1536, device="cuda", requires_grad=True)
    grad = torch.randn(2048, 1536, device="cuda")
    grads = []
    for _ in range(3):
        norm.weight.grad = None
        rmsnorm(x, norm.weight, EPS).backward(grad)
        grads.append(norm.weight.grad.clone())
    assert all(torch.equal(g, grads[0]) for g in grads[1:])


def _eager_add_norm(residual, branch, weight):
    from solution.rmsnorm import reference_rmsnorm

    total = residual + branch
    return total, reference_rmsnorm(total, weight, EPS)


@pytest.mark.parametrize("train_weight", [False, True])
@pytest.mark.parametrize("branch_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", SHAPES)
def test_add_rmsnorm_matches_eager(shape, branch_dtype, train_weight):
    from solution.rmsnorm import add_rmsnorm

    norm = _norm_module(shape[-1], train_weight)
    torch.manual_seed(2)
    residual = (2.0 * torch.randn(shape, device="cuda")).requires_grad_(True)
    branch = torch.randn(shape, device="cuda").to(branch_dtype).requires_grad_(True)
    residual_ref = residual.detach().clone().requires_grad_(True)
    branch_ref = branch.detach().clone().requires_grad_(True)
    g_sum = torch.randn(shape, device="cuda")
    g_out = torch.randn(shape, device="cuda")

    ref_sum, ref_out = _eager_add_norm(residual_ref, branch_ref, norm.weight)
    torch.autograd.backward((ref_sum, ref_out), (g_sum, g_out))
    ref_dw = norm.weight.grad
    norm.weight.grad = None

    total, out = add_rmsnorm(residual, branch, norm.weight, EPS)
    torch.autograd.backward((total, out), (g_sum, g_out))
    assert total.dtype == ref_sum.dtype and out.dtype == ref_out.dtype
    assert torch.equal(total, ref_sum)  # the same fp32 add
    tol = TOL[torch.float32]
    torch.testing.assert_close(out, ref_out, **tol)
    assert residual.grad.dtype == torch.float32 and branch.grad.dtype == branch_dtype
    torch.testing.assert_close(residual.grad, residual_ref.grad, **tol)
    torch.testing.assert_close(branch.grad.float(), branch_ref.grad.float(), **TOL[branch_dtype])
    if train_weight:
        torch.testing.assert_close(norm.weight.grad, ref_dw,
                                   **_dw_tol(torch.float32, residual.numel() // shape[-1]))


@pytest.fixture
def only_norm_patches(monkeypatch):
    """Other modules' patches of Qwen2MLP and Qwen2Attention (applied by other tests in the same
    session) are undone for the test, so it compares the norm patches alone."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    for cls in (mq.Qwen2MLP, mq.Qwen2Attention):
        original = cls.__dict__.get("_soup_original_forward")
        if original is not None:
            monkeypatch.setattr(cls, "forward", original)


@contextlib.contextmanager
def _original_forwards():
    """The unpatched Qwen2RMSNorm and Qwen2DecoderLayer forwards while active."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    saved = {}
    for cls in (mq.Qwen2RMSNorm, mq.Qwen2DecoderLayer):
        original = cls.__dict__.get("_soup_original_forward")
        if original is not None:
            saved[cls] = cls.forward
            cls.forward = original
    try:
        yield
    finally:
        for cls, forward in saved.items():
            cls.forward = forward


def _tiny_model(train_norms, seed=0):
    from transformers import Qwen2Config, Qwen2Model

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         attn_implementation="eager")
    model = Qwen2Model(config).cuda().float()
    model.train()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
                p.requires_grad_(train_norms)
    return model


@pytest.mark.parametrize("train_norms", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_patched_decoder_layer_matches_original(autocast, train_norms, monkeypatch,
                                                only_norm_patches):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import rmsnorm as rn

    rn.patch_qwen2_rmsnorm()
    rn.patch_qwen2_decoder_layer()
    assert mq.Qwen2DecoderLayer.forward is not mq.Qwen2DecoderLayer._soup_original_forward
    calls = {"norm": 0, "add_norm": 0}

    def counted(name, fn):
        def wrapper(*args):
            calls[name] += 1
            return fn(*args)
        return wrapper

    monkeypatch.setattr(rn, "rmsnorm", counted("norm", rn.rmsnorm))
    monkeypatch.setattr(rn, "add_rmsnorm", counted("add_norm", rn.add_rmsnorm))

    torch.manual_seed(3)
    embeds = torch.randn(2, 19, 64, device="cuda")
    grad = torch.randn(2, 19, 64, device="cuda")
    results = []
    for patched in (False, True):
        model = _tiny_model(train_norms)
        x = embeds.clone().requires_grad_(True)
        ctx = contextlib.nullcontext() if patched else _original_forwards()
        with ctx, torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
            out = model(inputs_embeds=x).last_hidden_state
            out.backward(grad)
        grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        results.append((out.detach(), x.grad, grads))
        # the reference runs none of our kernels; the patched model runs input_layernorm of both
        # layers and the final norm alone, and the post-attention norms fused with the add
        assert calls == ({"norm": 3, "add_norm": 2} if patched else {"norm": 0, "add_norm": 0})
    (ref, ref_dx, ref_grads), (out, dx, grads) = results
    assert out.dtype == ref.dtype == torch.float32
    assert set(grads) == set(ref_grads)
    assert any("norm" in n for n in grads) == train_norms
    # fp32: rounding differences of the norms only; autocast: they can flip fp16 roundings of the
    # matmul inputs downstream
    tol = dict(rtol=2e-2, atol=2e-2) if autocast else dict(rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out, ref, **tol)
    torch.testing.assert_close(dx, ref_dx, **tol)
    for name, g_ref in ref_grads.items():
        scale = g_ref.abs().max().clamp(min=1e-6)
        err = ((grads[name] - g_ref).abs().max() / scale).item()
        assert err < (2e-2 if autocast else 1e-4), f"{name}: relative grad error {err:.2e}"


def test_patches_fall_back_to_the_original(only_norm_patches):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution.rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm

    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_qwen2_rmsnorm()  # a second call changes nothing
    patch_qwen2_decoder_layer()
    original = mq.Qwen2RMSNorm._soup_original_forward
    assert mq.Qwen2RMSNorm.forward is not original
    assert getattr(original, "__qualname__", "").startswith("Qwen2RMSNorm.")

    # CPU input
    norm = mq.Qwen2RMSNorm(16, eps=EPS)
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(norm(x), original(norm, x))
    # fp16 weight on CUDA (not what the kernels take): the original's dtype rules apply
    norm16 = mq.Qwen2RMSNorm(16, eps=EPS).cuda().half()
    x32 = torch.randn(3, 5, 16, device="cuda")
    assert norm16(x32).dtype == original(norm16, x32).dtype
    torch.testing.assert_close(norm16(x32), original(norm16, x32))
    # a fake module, as Soup's regression tests use them
    fake = SimpleNamespace(weight=torch.ones(16), variance_epsilon=EPS)
    torch.testing.assert_close(mq.Qwen2RMSNorm.forward(fake, x), original(fake, x))
    # rank 4 is not a shape the patch expects
    norm_cuda = mq.Qwen2RMSNorm(16, eps=EPS).cuda()
    x4 = torch.randn(2, 3, 5, 16, device="cuda")
    torch.testing.assert_close(norm_cuda(x4), original(norm_cuda, x4))

    # the decoder layer on CPU takes the original forward
    from transformers import Qwen2Config

    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")
    torch.manual_seed(0)
    layer = mq.Qwen2DecoderLayer(config, layer_idx=0)
    rotary = mq.Qwen2RotaryEmbedding(config)
    h = torch.randn(1, 7, 32)
    pos = torch.arange(7)[None]
    emb = rotary(h, pos)
    torch.testing.assert_close(
        layer(h, position_embeddings=emb, position_ids=pos),
        mq.Qwen2DecoderLayer._soup_original_forward(layer, h, position_embeddings=emb,
                                                    position_ids=pos))
    # a hook on the post-attention norm must run, so the layer takes the original forward
    layer = layer.cuda()
    seen = []
    handle = layer.post_attention_layernorm.register_forward_hook(lambda *a: seen.append(1))
    layer(h.cuda(), position_embeddings=tuple(t.cuda() for t in emb), position_ids=pos.cuda())
    handle.remove()
    assert seen == [1]


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float16])
def test_rebuild_matches_the_forward_bit_for_bit(fused, source_dtype):
    """rmsnorm.rebuild recomputes a norm's output from what its node saved (input or sum, rstd,
    weight) exactly as the forward rounded it, for a consumer that drops its own copy."""
    from solution import rmsnorm as rn

    norm = _norm_module(1536, train_weight=False)
    g = torch.Generator("cuda").manual_seed(4)
    x = torch.randn(1, 838, 1536, device="cuda", generator=g).to(source_dtype).requires_grad_(True)
    if fused:
        branch = torch.randn(1, 838, 1536, device="cuda", generator=g).half()
        s, y = rn.add_rmsnorm(x, branch, norm.weight, EPS)
        assert rn.norm_source(s) is None  # the sum, not the normed output
    else:
        y = rn.rmsnorm(x, norm.weight, EPS)
    node = rn.norm_source(y)
    assert node is not None and y.dtype == torch.float32
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        rebuilt = rn.rebuild(node, dtype)
        assert rebuilt.dtype == dtype and rebuilt.shape == (838, 1536)
        assert torch.equal(rebuilt, y.detach().reshape(838, 1536).to(dtype))
    assert rn.norm_source(y * 1.0) is None and rn.norm_source(x) is None
    assert rn.norm_source(y.detach()) is None


def test_rebuild_kernel_is_warmed_up():
    """warm_rebuild (registered with warmup.py by fused_proj's attention patch) compiles the class
    that rebuild launches in training: fp32 rows of 1536, out in the run's half dtype."""
    from solution import rmsnorm as rn
    from solution import warmup

    compiles = warmup.watch_compiles(log=False)
    rn.rms_norm_rows_rebuild.device_caches.clear()
    compiles.clear()
    assert rn.warm_rebuild() is None
    assert [c["kernel"] for c in compiles] == ["rms_norm_rows_rebuild"]
    compiles.clear()
    norm = _norm_module(1536, train_weight=False)
    for n in (838, 1427):
        x = torch.randn(1, n, 1536, device="cuda", requires_grad=True)
        branch = torch.randn(1, n, 1536, device="cuda").half()
        _, y = rn.add_rmsnorm(x, branch, norm.weight, EPS)
        rn.rebuild(rn.norm_source(y), warmup.training_half_dtype())
    assert [c["kernel"] for c in compiles if c["kernel"] == "rms_norm_rows_rebuild"] == []
