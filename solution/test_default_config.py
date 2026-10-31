"""The configuration the training runs, end to end: every other test of solution/ starts with the int8
switches pinned off (conftest.py); these run whole LoRA training steps of a two-layer 4-bit Qwen2 at
the 1.5B shapes with the switches as the training has them (conftest.INT8_DEFAULTS: int8 MLP backward
and int8 (gate|up) offload on, int8 forward off) through the training's path (packed varlen forward,
fused projections, offload_saved_tensors with the model's weights, fp16 autocast, GradScaler, AdamW),
against the same steps with all three off.

The int8 paths take over after START_CALLS eligible calls (int8_bwd 112 MLP backwards, offload_int8 196
offloads: steps 4 and 7 of the 28-layer model); here they are set to switch at steps 2 and 1."""

import os

import pytest
import torch

from solution.conftest import INT8_DEFAULTS
from solution.test_fused_proj import (_config, _linear4bit, _lora_params, _positions, _quantize_mlp,
                                      _rel, _spans, patched)  # noqa: F401 - the module's autouse patch fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

LENGTHS = [212, 187, 310, 123]  # 832 tokens: a multiple of 16, as unpad.py's alignment tail leaves a step
LAYERS, STEPS = 2, 4
BWD_FROM, OFFLOAD_FROM = 2, 1  # the first step of each int8 path (START_CALLS = step x LAYERS)
LR, SCALE = 2.0e-4, 65536.0  # task.yaml's lr; the GradScaler's initial scale
# dQ of the attention backward is added with atomics: the same step twice differs by up to ~2e-4 in
# a LoRA gradient (measured)
ATOMICS = 1e-3
# against fp16 on the same weights, in a LoRA gradient: the int8 (gate|up) on the host up to 9e-3
# (measured; the input gradient of three MLPs is within 2e-2, test_offload_int8.py), with the int8 MLP
# backward too up to 4e-2 (the input gradient of one MLP is within 5e-2, test_int8_bwd.py)
OFFLOAD, INT8 = 0.03, 0.1


def _model():
    """Qwen2 with two decoder layers at the 1.5B shapes, NF4 base weights, peft LoRA r=16 / alpha 32 /
    dropout 0.05 (Soup's default) on q_proj and v_proj, adapters not at their zero init."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(_config(layers=LAYERS))
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.q_proj = _linear4bit(1536, 1536, 10 * i, bias=True)
        attn.k_proj = _linear4bit(1536, 256, 10 * i + 1, bias=True)
        attn.v_proj = _linear4bit(1536, 256, 10 * i + 2, bias=True)
        attn.o_proj = _linear4bit(1536, 1536, 10 * i + 3)
        _quantize_mlp(layer.mlp, 10 * i + 4)
    model = model.cuda()
    model.is_loaded_in_4bit = True  # what peft reads to wrap bnb layers with its Linear4bit
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"]))
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in _lora_params(model):
            p.copy_(torch.randn(p.shape, generator=g) * 0.02)
    model.train()
    return model


@pytest.fixture(scope="module")
def model():
    return _model()


def _train(model, monkeypatch, weights=None):
    """STEPS training steps from the model's LoRA weights as they are (put back afterwards), or each
    step from `weights[step]` (another run's): the same weights in two runs, so that their gradients
    differ by what the two runs compute differently. AdamW turns the attention backward's atomics
    noise (1e-4 in a gradient) into different weights at once: two free runs of the same
    configuration differ by 3e-2 in the gradients of step 2. Per step: the weights it started from,
    loss, the last hidden state, the unscaled LoRA gradients, how many MLP backwards ran int8 and how
    many (gate|up) went to the host as int8; and the GradScaler's scale at the end."""
    from solution import int8_bwd, offload, offload_int8
    from solution.unpad import varlen_model_forward

    monkeypatch.setattr(int8_bwd, "START_CALLS", BWD_FROM * LAYERS)
    monkeypatch.setitem(int8_bwd._STATE, "calls", 0)
    monkeypatch.setattr(offload_int8, "START_CALLS", OFFLOAD_FROM * LAYERS)
    monkeypatch.setitem(offload_int8._STATE, "calls", 0)
    backwards = []
    original = int8_bwd.mlp_backward

    def counted(*args, **kwargs):
        backwards.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(int8_bwd, "mlp_backward", counted)
    params = _lora_params(model)
    start = [p.detach().clone() for p in params]
    inner = model.base_model.model
    optimizer = torch.optim.AdamW(params, lr=LR)
    scaler = torch.amp.GradScaler("cuda", init_scale=SCALE)
    steps = []
    try:
        with offload.offload_saved_tensors(weights=[*model.parameters(), *model.buffers()]) as off:
            for step in range(STEPS):
                if weights is not None:
                    with torch.no_grad():
                        for p, w in zip(params, weights[step]):
                            p.copy_(w)
                ids = torch.randint(0, 512, (1, sum(LENGTHS)), device="cuda",
                                    generator=torch.Generator("cuda").manual_seed(100 + step))
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(200 + step)  # the LoRA dropout masks
                record = {"weights": [p.detach().clone() for p in params]}
                int8_before, backwards_before = off.stats.get("int8", 0), len(backwards)
                with torch.autocast("cuda", dtype=torch.float16):
                    hidden = varlen_model_forward(inner.model, ids, _positions(LENGTHS), _spans(LENGTHS))
                    logits = inner.lm_head(hidden).float()
                loss = torch.nn.functional.cross_entropy(logits[0, :-1], ids[0, 1:])
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                record["grads"] = [p.grad.clone() for p in params]
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                record.update(loss=loss.detach(), hidden=hidden.detach(),
                              int8_offloads=off.stats.get("int8", 0) - int8_before,
                              int8_backwards=len(backwards) - backwards_before)
                steps.append(record)
        scale = scaler.get_scale()
    finally:
        with torch.no_grad():
            for p, s in zip(params, start):
                p.copy_(s)
                p.grad = None
    return steps, scale


def test_int8_defaults_are_the_trained_configuration(_fp16_mlp_backward):
    """INT8_DEFAULTS, which test_default_config runs, are the switches a process starts with when no
    SOUP_SOLUTION_* variable sets them (the harness sets none): what the training runs."""
    names = ("SOUP_SOLUTION_INT8_BWD", "SOUP_SOLUTION_OFFLOAD_INT8", "SOUP_SOLUTION_INT8_FWD")
    if _fp16_mlp_backward is None:
        pytest.skip("needs triton")
    if any(name in os.environ for name in names):
        pytest.skip("an int8 switch is set in the environment")
    assert _fp16_mlp_backward == INT8_DEFAULTS


def test_default_configuration_trains_like_fp16(model, default_config, monkeypatch):
    """Four LoRA steps in the default configuration against the all-fp16 one. On the same weights
    (each step from the fp16 run's): the forward is the same code (int8 forward off), so loss and
    output are bit for bit the fp16 run's; the LoRA gradients (A and B of q and v in both layers)
    match up to the atomics before the int8 paths take over and stay within their quantization noise
    after. Running free, the loss follows the fp16 run's, and nothing overflows: no inf or NaN, the
    GradScaler skips no step."""
    default_config(False)
    ref, ref_scale = _train(model, monkeypatch)
    default_config(True)
    got, scale = _train(model, monkeypatch, weights=[s["weights"] for s in ref])
    free, free_scale = _train(model, monkeypatch)

    assert [s["int8_backwards"] for s in ref] == [0] * STEPS
    assert [s["int8_offloads"] for s in ref] == [0] * STEPS
    assert scale == free_scale == ref_scale == SCALE  # no inf/NaN gradient: no step skipped
    for step, (s, f, r) in enumerate(zip(got, free, ref)):
        assert torch.equal(s["loss"], r["loss"]) and torch.equal(s["hidden"], r["hidden"])
        assert all(torch.isfinite(g).all() for g in s["grads"] + f["grads"])
        errors = [_rel(g, gr) for g, gr in zip(s["grads"], r["grads"])]
        drift = abs(f["loss"].item() - r["loss"].item()) / r["loss"].item()
        print(f"\nstep {step}: int8 offloads {s['int8_offloads']}, int8 backwards {s['int8_backwards']}; "
              f"same weights: LoRA gradients max relative error {max(errors):.2e}; "
              f"free run: loss {f['loss'].item():.5f} vs {r['loss'].item():.5f}")
        assert drift < 1e-3
        assert max(errors) < (ATOMICS if step < OFFLOAD_FROM else OFFLOAD if step < BWD_FROM else INT8)


def test_default_configuration_switch_points_are_fixed(model, default_config, monkeypatch):
    """The steps at which the int8 paths take over are the START_CALLS counts, not the compile
    threads' timing: two runs switch at the same steps (offload from step 1, backward from step 2,
    every layer), and on the same weights the int8 paths give the same gradients up to the
    attention backward's atomics."""
    first, _ = _train(model, monkeypatch)
    second, _ = _train(model, monkeypatch, weights=[s["weights"] for s in first])
    for run in (first, second):
        assert [s["int8_offloads"] for s in run] == [0] * OFFLOAD_FROM + [LAYERS] * (STEPS - OFFLOAD_FROM)
        assert [s["int8_backwards"] for s in run] == [0] * BWD_FROM + [LAYERS] * (STEPS - BWD_FROM)
    for a, b in zip(first, second):
        assert torch.equal(a["loss"], b["loss"])
        assert max(_rel(g, h) for g, h in zip(a["grads"], b["grads"])) < ATOMICS
