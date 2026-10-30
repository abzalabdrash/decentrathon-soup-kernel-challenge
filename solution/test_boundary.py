"""boundary.py: the step boundary's lists give what the four walks over the model give (the FLOP
count, the clipped gradients and their norm, the cleared gradients, the training flags), and the
patches step aside for models they have no lists for."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _peft_model(seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config).cuda()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.05)
    return model


@pytest.fixture
def boundary_on():
    from solution import boundary

    saved = boundary._STATE["on"]
    boundary._STATE["on"] = True
    boundary.patch_step_boundary()
    yield boundary
    boundary.unpatch_step_boundary()
    boundary._STATE["on"] = saved
    boundary._CACHE.clear()


def _grads(model):
    ids = torch.randint(0, 256, (2, 40), device="cuda", generator=torch.Generator("cuda").manual_seed(1))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()


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


def test_zero_grad_and_train_fast_paths(boundary_on):
    model = _peft_model()
    boundary_on._entry(model)
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
