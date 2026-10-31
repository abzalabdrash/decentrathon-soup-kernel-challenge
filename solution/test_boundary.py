"""boundary.py: the step boundary's lists give what the four walks over the model give (the FLOP
count, the clipped gradients and their norm, the cleared gradients, the training flags), and the
patches step aside for models they have no lists for."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _peft_model(seed=0):
    """The 4-bit tiny Qwen2 with peft LoRA of test_fused_proj (a plain-linear peft model makes peft
    probe torchao, whose old version on Kaggle raises)."""
    from solution.test_fused_proj import _tiny_peft_model

    torch.manual_seed(seed)
    return _tiny_peft_model()


@pytest.fixture
def boundary_on():
    from solution import boundary

    def forget():
        # the ids of earlier tests' models stay in `watched` after their lists are dropped: a module
        # made later at a freed address (the ModuleList of a layer slice) would count as a listed
        # model's and bump the generation (a rebuild, not an error, but test_layer_slices counts builds)
        boundary._CACHE.clear()
        boundary._STATE["watched"].clear()

    saved = boundary._STATE["on"]
    boundary._STATE["on"] = True
    forget()
    boundary.patch_step_boundary()
    yield boundary
    boundary.unpatch_step_boundary()
    boundary._STATE["on"] = saved
    forget()


def _grads(model):
    ids = torch.randint(0, 512, (2, 160), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, labels=ids)
    (out.loss * 1024.0).backward()


def test_floating_point_ops_and_clip_match(boundary_on):
    from transformers import Trainer

    model = _peft_model()
    inputs = {"input_ids": torch.zeros(4, 77, dtype=torch.long, device="cuda")}

    class Fake:
        pass

    fake = Fake()
    fake.model = model
    original = Trainer._soup_step_boundary_previous[1](fake, inputs)
    boundary_on._entry(model)
    assert Trainer.floating_point_ops(fake, inputs) == original

    _grads(model)
    ref = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    norm_ref = torch.nn.utils.clip_grad_norm_([p for p in model.parameters()], 0.01)
    clipped_ref = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    for p, g in zip([p for p in model.parameters() if p.grad is not None], ref):
        p.grad.copy_(g)
    norm = torch.nn.utils.clip_grad_norm_(boundary_on._entry(model)["trainable"], 0.01)
    clipped = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    assert torch.equal(norm, norm_ref)
    assert all(torch.equal(a, b) for a, b in zip(clipped, clipped_ref))


def test_zero_grad_and_train_fast_paths(boundary_on, monkeypatch):
    model = _peft_model()
    entry = boundary_on._entry(model)
    # Qwen2ForCausalLM and Qwen2Model use PreTrainedModel.train (transformers 5.16): still listed
    assert entry["modules"] is not None and len(entry["kernel_checked"]) == 2
    _grads(model)
    assert any(p.grad is not None for p in model.parameters())
    model.zero_grad()
    assert all(p.grad is None for p in model.parameters())
    model.eval()
    assert not any(m.training for m in model.modules())
    model.train()
    assert all(m.training for m in model.modules())
    model.train(False)
    assert not any(m.training for m in model.modules())
    base = model.base_model.model
    # PreTrainedModel.train would re-cast kernels: the original runs
    monkeypatch.setattr(type(base), "use_kernels", property(lambda self: True))
    calls = []
    base.set_use_kernels = lambda flag: calls.append(flag)
    model.train()
    assert calls == [True] and all(m.training for m in model.modules())


def test_models_without_lists_take_the_original(boundary_on):
    model = _peft_model()
    model.train()
    _grads(model)
    model.zero_grad()  # no list for this model: nn.Module.zero_grad
    assert all(p.grad is None for p in model.parameters())
    model.eval()
    assert not any(m.training for m in model.modules())
    assert model not in boundary_on._CACHE


def test_registration_drops_the_lists(boundary_on):
    model = _peft_model()
    entry = boundary_on._entry(model)
    model.base_model.model.extra = torch.nn.Linear(2, 2).cuda()  # a module registered anywhere
    assert boundary_on._entry(model) is not entry
    assert any(m is model.base_model.model.extra for m in boundary_on._entry(model)["modules"])


def test_layer_slices_keep_the_lists(boundary_on):
    """Every forward slices `layers[:num_hidden_layers]` (unpad.varlen_model_forward, HF's
    Qwen2Model), and a sliced ModuleList registers the layers into a new ModuleList: that must not
    drop the lists (it rebuilt them every step, 8.8 ms of the FLOP counter's call on the T4)."""
    model = _peft_model()
    entry = boundary_on._entry(model)
    decoder = model.base_model.model.model
    builds = boundary_on._STATE["builds"]
    sliced = decoder.layers[: decoder.config.num_hidden_layers]
    assert len(sliced) == decoder.config.num_hidden_layers
    assert boundary_on._entry(model) is entry and boundary_on._STATE["builds"] == builds
