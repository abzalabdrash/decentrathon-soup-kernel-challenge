"""mem_policy.py: the attention layers a step keeps under the fixed cap (RECOMPUTE_ATTN=cap)."""

from solution import mem_policy


def test_kept_layers_fit_under_the_cap():
    per_layer = 10288  # bytes per token one layer's attention keeps (fused_proj._kept_bytes)
    mib = mem_policy.MIB
    previous = 28
    for tokens in (64, 200, 400, 600, 830, 1000, 1200, 1440, 1680, 2048):
        k = mem_policy.kept_layers(tokens, per_layer, 28)
        assert 0 <= k <= 28 and k <= previous  # fewer layers as the step grows
        predicted = mem_policy.BASE_MIB + mem_policy.PER_TOKEN_MIB * tokens + k * tokens * per_layer / mib
        assert k == 0 or predicted <= mem_policy.CAP_MIB
        previous = k
    assert mem_policy.kept_layers(2048, per_layer, 28) == 0  # the longest step a batch can make
    assert mem_policy.kept_layers(200, per_layer, 28) == 28
    assert mem_policy.kept_layers(0, per_layer, 28) == 0


def test_recompute_layers_cap_mode(monkeypatch):
    from types import SimpleNamespace

    from solution import fused_proj

    plan = SimpleNamespace(n_tokens=830, bases=(SimpleNamespace(out_features=1536),
                                                SimpleNamespace(out_features=256)),
                           lora_q=(SimpleNamespace(weight=SimpleNamespace(shape=(16, 1536))),),
                           heads=12)
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "cap")
    plain, split, rebuilt = fused_proj._kept_bytes(plan, 1536)
    from solution.rmsnorm import _SPLIT

    per_layer = (split if _SPLIT["keep"] else plain) - rebuilt
    assert fused_proj.recompute_layers(plan, 1536, 28) == 28 - mem_policy.kept_layers(830, per_layer, 28)
    plan.n_tokens = 2048
    assert fused_proj.recompute_layers(plan, 1536, 28) == 28
