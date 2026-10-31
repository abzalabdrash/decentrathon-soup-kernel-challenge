"""Night task N1: the post-attention norm's split sum (rmsnorm.py, SOUP_SOLUTION_SPLIT_SUM) and the
attention recompute (fused_proj.py, SOUP_SOLUTION_RECOMPUTE_ATTN): what the backward rebuilds is the
forward's tensors bit for bit, the gradients are the kept path's, the memory kept per layer drops,
the CUDA random state is untouched, and the auto policy picks layers by the step's tokens."""

import pytest
import torch

from solution.test_fused_proj import (_config, _lora_params, _quantize_attention,
                                      _quantize_mlp, _rel, _spans, _tiny_peft_model, _tiny_step,
                                      patched)  # noqa: F401 - the module's autouse patch fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def modes():
    """fused_proj's recompute and rmsnorm's split switches, restored after the test."""
    from solution import fused_proj, rmsnorm

    saved = (dict(fused_proj._RECOMPUTE), dict(rmsnorm._SPLIT))
    fused_proj._RECOMPUTE["mismatches"] = []
    yield fused_proj._RECOMPUTE, rmsnorm._SPLIT
    fused_proj._RECOMPUTE.clear()
    fused_proj._RECOMPUTE.update(saved[0])
    rmsnorm._SPLIT.clear()
    rmsnorm._SPLIT.update(saved[1])


def _set(modes, recompute=None, split=False, check=False, tokens=640):
    rec, spl = modes
    rec.update(mode=recompute, check=check, tokens=tokens, mismatches=[])
    spl["keep"] = split


# ------------------------------------------------------------------------------------------------
# the split backward kernel


@pytest.mark.parametrize("rows", [826, 1440, 7])
def test_split_sum_backward_is_bitwise(modes, rows):
    from solution.rmsnorm import KEEP_BRANCH, add_rmsnorm

    g = torch.Generator("cuda").manual_seed(rows)
    residual = torch.randn(1, rows, 1536, device="cuda", generator=g) * 3
    branch = (torch.randn(1, rows, 1536, device="cuda", generator=g) * 0.5).half()
    weight = torch.rand(1536, device="cuda", generator=g) + 0.5
    ds = torch.randn(1, rows, 1536, device="cuda", generator=g)
    dy = torch.randn(1, rows, 1536, device="cuda", generator=g)

    def run(addends):
        r = residual.clone().requires_grad_(True)
        b = branch.clone().requires_grad_(True)
        s, y = add_rmsnorm(r, b, weight, 1e-6, addends)
        torch.autograd.backward([s, y], [ds, dy])
        return s.detach(), y.detach(), r.grad, b.grad

    ref = run(None)
    got = run(KEEP_BRANCH)
    for a, b in zip(got, ref):
        assert a.dtype == b.dtype and torch.equal(a, b)


def test_split_sum_with_a_trained_weight_matches(modes):
    from solution.rmsnorm import KEEP_BRANCH, add_rmsnorm

    g = torch.Generator("cuda").manual_seed(3)
    residual = torch.randn(1, 300, 1536, device="cuda", generator=g)
    branch = torch.randn(1, 300, 1536, device="cuda", generator=g).half()
    dy = torch.randn(1, 300, 1536, device="cuda", generator=g)

    def run(addends):
        r = residual.clone().requires_grad_(True)
        b = branch.clone().requires_grad_(True)
        w = torch.ones(1536, device="cuda", requires_grad=True)
        _, y = add_rmsnorm(r, b, w, 1e-6, addends)
        y.backward(dy)
        return r.grad, b.grad, w.grad

    for a, b in zip(run(KEEP_BRANCH), run(None)):
        assert torch.equal(a, b)


def test_post_attention_addends_rules(modes):
    from solution.rmsnorm import KEEP_BRANCH, post_attention_addends

    residual = torch.randn(1, 8, 16, device="cuda")
    branch = torch.randn(1, 8, 16, device="cuda").half()
    _set(modes, split=False)
    assert post_attention_addends(residual.requires_grad_(True), branch) is None
    _set(modes, split=True)
    assert post_attention_addends(residual, branch) is KEEP_BRANCH
    leaf = torch.randn(1, 8, 16, device="cuda")  # an input nobody keeps (layer 0's embedding output)
    assert post_attention_addends(leaf, branch) is None
    assert post_attention_addends(residual, branch.float()) is None  # not a half branch
    with torch.no_grad():
        assert post_attention_addends(residual, branch) is None


# ------------------------------------------------------------------------------------------------
# the whole packed step of a tiny 4-bit Qwen2 with peft (two layers)


LENGTHS = [200, 151, 300, 177]


def _ids():
    return torch.randint(0, 512, (1, sum(LENGTHS)), device="cuda",
                         generator=torch.Generator("cuda").manual_seed(2))


@pytest.mark.parametrize("dropout", [0.0, 0.05])
def test_tiny_model_recompute_rebuilds_the_forward_bit_for_bit(modes, dropout):
    from solution import fused_proj

    model = _tiny_peft_model(dropout)
    ids = _ids()
    _set(modes)
    torch.manual_seed(5)
    ref_hidden, ref_loss, ref_grads = _tiny_step(model, ids, LENGTHS)
    rebuilds = []
    original = fused_proj._AttnRecompute._rebuild
    try:
        fused_proj._AttnRecompute._rebuild = (
            lambda self, with_branch: rebuilds.append(with_branch) or original(self, with_branch))
        _set(modes, recompute="all", split=True, check=True)
        torch.manual_seed(5)
        hidden, loss, grads = _tiny_step(model, ids, LENGTHS)
    finally:
        fused_proj._AttnRecompute._rebuild = original
    assert rebuilds == [True, True]  # both layers, started by the post-attention norm's backward
    assert modes[0]["mismatches"] == []  # x16, q, k, v, output, lse, o_proj output: bit for bit
    assert torch.equal(hidden, ref_hidden) and torch.equal(loss, ref_loss)
    # the attention backward's dq accumulates with atomics: equal up to its own run-to-run noise
    for g, ref in zip(grads, ref_grads):
        assert _rel(g, ref) < 1e-3


def test_tiny_model_split_sum_only(modes):
    model = _tiny_peft_model()
    ids = _ids()
    _set(modes)
    ref = _tiny_step(model, ids, LENGTHS)
    _set(modes, split=True)
    got = _tiny_step(model, ids, LENGTHS)
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    for g, r in zip(got[2], ref[2]):
        assert _rel(g, r) < 1e-3


def test_recompute_leaves_the_random_state_alone(modes):
    model = _tiny_peft_model(dropout=0.05)
    ids = _ids()
    states = []
    for mode in (None, "all"):
        _set(modes, recompute=mode, split=mode is not None)
        torch.manual_seed(9)
        _tiny_step(model, ids, LENGTHS)
        states.append(torch.cuda.get_rng_state())
    assert torch.equal(states[0], states[1])


def test_auto_recomputes_only_long_steps(modes, monkeypatch):
    from solution import fused_proj

    model = _tiny_peft_model()
    ids = _ids()
    made = []
    original = fused_proj._AttnRecompute.__init__

    def count(self, plan):
        made.append(plan.n_tokens)
        original(self, plan)

    monkeypatch.setattr(fused_proj._AttnRecompute, "__init__", count)
    _set(modes, recompute="auto", split=True, tokens=10_000)
    _tiny_step(model, ids, LENGTHS)
    assert made == []  # 828 tokens <= 10 000: no layer recomputes
    _set(modes, recompute="auto", split=True, tokens=400)
    _tiny_step(model, ids, LENGTHS)
    assert len(made) in (1, 2)  # some layers, from layer 0


def test_policy_counts():
    from types import SimpleNamespace

    from solution import fused_proj

    base = SimpleNamespace(out_features=1536)
    kv = SimpleNamespace(out_features=256)
    plan = SimpleNamespace(bases=(base, kv, kv), heads=12,
                           lora_q=(SimpleNamespace(weight=torch.empty(16, 1536)), None))
    saved = dict(fused_proj._RECOMPUTE)
    try:
        fused_proj._RECOMPUTE.update(mode="auto", tokens=640)
        counts = []
        for n in (300, 640, 700, 900, 1200, 1427, 1600, 4000):
            plan.n_tokens = n
            counts.append(fused_proj.recompute_layers(plan, 1536, 28))
        assert counts[:2] == [0, 0] and counts == sorted(counts) and counts[-1] == 28
        assert 0 < counts[5] < 28
        fused_proj._RECOMPUTE.update(mode="all")
        assert fused_proj.recompute_layers(plan, 1536, 28) == 28
        fused_proj._RECOMPUTE.update(mode=None)
        assert fused_proj.recompute_layers(plan, 1536, 28) == 0
    finally:
        fused_proj._RECOMPUTE.clear()
        fused_proj._RECOMPUTE.update(saved)


# ------------------------------------------------------------------------------------------------
# a decoder layer at the shapes of the task: rebuilt tensors and the memory a layer keeps


@pytest.fixture(scope="module")
def decoder_1p5b():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    config = _config(layers=2)
    layer = mq.Qwen2DecoderLayer(config, 1)  # layer 1: its input comes from a norm with a node
    layer.self_attn = _quantize_attention(layer.self_attn, seed=3, dropout=0.05)
    layer.mlp = _quantize_mlp(layer.mlp, seed=20)
    layer = layer.cuda()
    rotary = mq.Qwen2RotaryEmbedding(config=config).cuda()
    layer.train()
    return layer, rotary


def _layer_forward(layer, rotary, lengths, seed=5):
    """The decoder layer as unpad's loop calls it (carry on): returns (branch, residual, inputs)."""
    from solution.unpad import varlen

    g = torch.Generator("cuda").manual_seed(seed)
    n = sum(lengths)
    hidden = torch.randn(1, n, 1536, device="cuda", generator=g).requires_grad_(True)
    branch_in = (torch.randn(1, n, 1536, device="cuda", generator=g) * 0.1).half().requires_grad_(True)
    positions = torch.cat([torch.arange(k) for k in lengths]).unsqueeze(0).cuda()
    pe = rotary(hidden, positions)
    with torch.autocast("cuda", dtype=torch.float16), varlen(_spans(lengths)):
        out = layer(hidden, attention_mask=None, position_ids=positions, past_key_values=None,
                    use_cache=False, position_embeddings=pe, _soup_branch=branch_in, _soup_carry=True)
    return out, (hidden, branch_in)


@pytest.mark.parametrize("lengths", [[212, 187, 310, 117], [500, 480, 447]])
def test_decoder_layer_recompute_bitwise_and_memory(modes, decoder_1p5b, lengths, cuda_bytes):
    """One layer between an input and the next layer's input norm (which consumes the layer's two
    outputs, as in training, so that what stays allocated is what the layer keeps for its
    backward): outputs and gradients against the kept path, and the bytes per token kept."""
    from solution.rmsnorm import add_rmsnorm

    layer, rotary = decoder_1p5b
    n = sum(lengths)
    next_norm = torch.ones(1536, device="cuda")

    def run(mode, split, check=False):
        # the check mode keeps clones of what the recompute rebuilds (to compare them): measured
        # without it, checked in a run of its own
        _set(modes, recompute=mode, split=split, check=check)
        for p in layer.parameters():
            p.grad = None
        torch.cuda.synchronize()
        torch.manual_seed(7)
        (mlp_out, residual), (hidden, branch_in) = _layer_forward(layer, rotary, lengths)
        s_next, y_next = add_rmsnorm(residual, mlp_out, next_norm, 1e-6)  # the next input norm
        # compared on the host: no run's outputs or gradients stay on the GPU during the next runs,
        # and the layer's output (the sum) is not held past the next norm, whose input it is
        out = (mlp_out.detach().cpu(), residual.detach().cpu())
        del mlp_out, residual
        torch.cuda.synchronize()
        held = cuda_bytes.now()
        g = torch.Generator("cuda").manual_seed(8)
        grads = [torch.randn(s_next.shape, device="cuda", generator=g),
                 torch.randn(y_next.shape, device="cuda", generator=g)]
        torch.autograd.backward([s_next, y_next], grads)
        del s_next, y_next
        return {"held": held, "out": out,
                "grads": [t.cpu() for t in [hidden.grad, branch_in.grad]
                          + [p.grad for p in _lora_params(layer)]]}

    run(None, False)  # first call: buffers, caches, the allocator's pool
    checked = run("all", True, check=True)
    assert modes[0]["mismatches"] == []
    results = {name: run(mode, split) for name, mode, split in
               (("kept", None, False), ("split", None, True), ("recompute", "all", True))}
    for a, b in zip(checked["grads"], results["recompute"]["grads"]):
        assert _rel(a, b) < 1e-4  # dQ of the attention backward is added with atomics
    for name in ("split", "recompute"):
        assert torch.equal(results[name]["out"][0], results["kept"]["out"][0])
        assert torch.equal(results[name]["out"][1], results["kept"]["out"][1])
        for a, b in zip(results[name]["grads"], results["kept"]["grads"]):
            assert _rel(a, b) < 1e-3
    saved_split = (results["kept"]["held"] - results["split"]["held"]) / n
    saved_recompute = (results["kept"]["held"] - results["recompute"]["held"]) / n
    print(f"\nkept for the backward, {n} tokens: split sum -{saved_split:.0f} B per token, "
          f"recompute -{saved_recompute:.0f} B per token "
          f"(-{saved_recompute * n / 2 ** 20:.1f} MiB per layer)")
    # the split sum keeps the half branch instead of the fp32 sum: 2 bytes x 1536 per token less
    assert saved_split > 0.8 * 2 * 1536
    # the recompute keeps neither the sum nor q, k, v, the output and lse: > 10 KB per token less
    assert saved_recompute > 10_000


@pytest.mark.parametrize("split", [False, True])
def test_mlp_backward_in_place_is_bitwise(monkeypatch, split):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1" if split else "0")
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()
    g = torch.Generator("cuda").manual_seed(7)
    x = torch.randn(1, 826, 1536, device="cuda", generator=g)
    grad = torch.randn(1, 826, 1536, device="cuda", generator=g).half()
    saved = dict(fused_proj._MLP_MEMORY)
    results = []
    try:
        for inplace in (False, True):
            fused_proj._MLP_MEMORY.update(recompute=False, inplace=inplace)
            xg = x.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = mlp(xg)
            out.backward(grad.to(out.dtype))
            results.append((out.detach(), xg.grad))
    finally:
        fused_proj._MLP_MEMORY.clear()
        fused_proj._MLP_MEMORY.update(saved)
    assert torch.equal(results[0][0], results[1][0]) and torch.equal(results[0][1], results[1][1])
