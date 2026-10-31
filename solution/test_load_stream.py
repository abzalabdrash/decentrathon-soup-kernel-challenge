"""The load-time moves (load_stream.py) on a tiny 4-bit Qwen2 in Soup's setup order: the weights in
pinned host memory as soon as they are loaded, adopted by the streamers without a second copy, the
same training step bit for bit as a load without the moves, the fallbacks, and loads outside Soup's
setup left alone."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

VOCAB, HIDDEN, LAYERS = 1000, 128, 4


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=384, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(4.0)
    path = tmp_path_factory.mktemp("tiny_qwen2_load")
    model.to(torch.bfloat16).save_pretrained(path)  # bf16 on disk, loaded in fp16, as the task's checkpoint
    return path


@pytest.fixture
def patched(monkeypatch):
    """The patches of apply() that take part in the setup, with a fake Soup setup to run inside."""
    import peft
    from soup_cli.trainer import sft as soup_sft

    from solution import fused_ce, head_stream, load_stream, weight_stream
    from solution.unpad import patch_qwen2_attention

    cls = soup_sft.SFTTrainerWrapper
    monkeypatch.setenv("SOUP_SOLUTION_HEAD_ROWS", "96")
    monkeypatch.setattr(cls, "_setup_transformers", lambda self: self._fake_setup())
    monkeypatch.setattr(cls, "_soup_head_stream", False, raising=False)
    monkeypatch.setattr(cls, "_soup_weight_stream", False, raising=False)
    prepare = peft.prepare_model_for_kbit_training
    while hasattr(prepare, "_soup_original"):
        prepare = prepare._soup_original
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", prepare)
    monkeypatch.setitem(load_stream._STATE, "layers", False)
    monkeypatch.setitem(load_stream._STATE, "embedding", False)
    from transformers import core_model_loading, modeling_utils

    for module, name in ((modeling_utils, "caching_allocator_warmup"),
                         (modeling_utils, "convert_and_load_state_dict_in_model"),
                         (core_model_loading, "get_device"), (core_model_loading, "set_param_for_module")):
        monkeypatch.setattr(module, name, getattr(module, name))
    fused_ce.patch_qwen2_causal_lm()
    patch_qwen2_attention()
    head_stream.patch_head_stream()
    weight_stream.patch_weight_stream(slots=2)
    return cls


def _load(path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, dtype=torch.float16,
                                                device_map={"": 0})


def _peft(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(2)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.copy_(torch.randn(param.shape, generator=gen) * 0.02)
    return model


def _setup(cls, path, checkpointing=False, after_load=None):
    """Soup's order inside _setup_transformers: load, prepare_model_for_kbit_training, get_peft_model."""
    import peft

    seen = {}

    def fake_setup(self):
        model = _load(path)
        seen["after_load"] = torch.cuda.memory_allocated()
        if after_load is not None:
            after_load(model)
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing)
        self.model = _peft(model)

    wrapper = cls.__new__(cls)
    wrapper._fake_setup = fake_setup.__get__(wrapper)
    cls._setup_transformers(wrapper)
    return wrapper.model, seen


def _step(model):
    gen = torch.Generator().manual_seed(3)
    lengths = (40, 31, 17)
    ids = torch.randint(0, VOCAB, (3, max(lengths)), generator=gen).cuda()
    mask = (torch.arange(ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]).long().cuda()
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :5] = -100
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    (loss * 1024.0).backward()
    return [loss.detach()] + [p.grad.clone() for _, p in model.named_parameters() if p.requires_grad]


def _layer_weights(model):
    import bitsandbytes as bnb

    return [m.weight for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]


def test_load_moves_weights_and_the_step_is_the_same(patched, tiny_dir, monkeypatch):
    from solution import load_stream
    from solution.head_stream import streamer_for
    from solution.weight_stream import _LOADED, streamer_of

    checks = {}

    def after_load(model):
        # right after from_pretrained: every 4-bit weight in pinned host memory, the embedding a
        # placeholder on the GPU tied to the lm_head, nothing of the weights left on the GPU
        weights = _layer_weights(model)
        checks["layers_on_host"] = all(not w.is_cuda and w.data.is_pinned() for w in weights)
        checks["absmax_on_host"] = all(not w.quant_state.absmax.is_cuda for w in weights)
        emb = model.get_input_embeddings().weight
        checks["embedding"] = (emb.is_cuda and emb.untyped_storage().nbytes() == 2
                               and model.lm_head.weight is emb and streamer_for(emb) is not None)
        checks["loaded"] = len(_LOADED)
        # the buffers pinned ahead by the thread were used, the predicted layer size exact
        checks["pinned_ahead"] = (all(h._pinned is not None and h._pinned.nbytes == h.nbytes for h in _LOADED.values())
                                  and streamer_for(emb)._pinned is not None)

    model, seen = _setup(patched, tiny_dir, after_load=after_load)
    assert checks == {"layers_on_host": True, "absmax_on_host": True, "embedding": True, "loaded": LAYERS,
                      "pinned_ahead": True}
    load = load_stream.LAST["load"]
    assert load.layers_moved == LAYERS and load.embedding_bytes == VOCAB * HIDDEN * 2
    streamer = streamer_of(model)
    assert streamer is not None and streamer.adopted == LAYERS and len(_LOADED) == 0
    param = model.get_base_model().get_input_embeddings().weight
    assert param.dtype == torch.float32 and streamer_for(param) is not None  # peft's dtype, no cast
    ours = _step(model)
    streamer.restore()
    streamer_for(param).restore()

    monkeypatch.setitem(load_stream._STATE, "layers", False)
    monkeypatch.setitem(load_stream._STATE, "embedding", False)
    reference, seen_ref = _setup(patched, tiny_dir)
    assert seen_ref["after_load"] > seen["after_load"]
    theirs = _step(reference)
    assert len(ours) == len(theirs) and all(torch.equal(a, b) for a, b in zip(ours, theirs))
    streamer_of(reference).restore()


def test_gradient_checkpointing_takes_the_layers_back(patched, tiny_dir):
    from solution.weight_stream import _LOADED, streamer_of

    model, _ = _setup(patched, tiny_dir, checkpointing=True)
    assert streamer_of(model) is None and len(_LOADED) == 0
    assert all(w.is_cuda and w.quant_state.absmax.is_cuda for w in _layer_weights(model))


def test_declined_head_takes_the_embedding_back(patched, tiny_dir):
    from solution.head_stream import streamer_for

    def untie(model):
        model.lm_head.weight = torch.nn.Parameter(torch.zeros(VOCAB, HIDDEN, dtype=torch.float16, device="cuda"))

    model, _ = _setup(patched, tiny_dir, after_load=untie)
    emb = model.get_base_model().get_input_embeddings().weight
    assert streamer_for(emb) is None and emb.untyped_storage().nbytes() == VOCAB * HIDDEN * 4
    fresh = _load(tiny_dir)  # outside Soup's setup: the plain loader
    assert torch.equal(emb, fresh.get_input_embeddings().weight.float())


def test_loads_outside_soup_setup_are_untouched(patched, tiny_dir):
    from solution import load_stream
    from solution.weight_stream import _LOADED

    load_stream.LAST["load"] = None
    model = _load(tiny_dir)
    assert load_stream.LAST["load"] is None and len(_LOADED) == 0
    assert all(w.is_cuda for w in _layer_weights(model))
    assert model.get_input_embeddings().weight.untyped_storage().nbytes() == VOCAB * HIDDEN * 2


def test_meta_init_skips_meta_tensors_only():
    """SOUP_SOLUTION_META_INIT: torch's init functions return meta tensors untouched; a real tensor gets
    the same values and leaves the generator in the same state; undo puts torch's functions back."""
    from solution import load_stream

    original = torch.nn.init.kaiming_uniform_
    torch.manual_seed(7)
    expected = torch.nn.Linear(16, 8)
    expected_rng = torch.get_rng_state()
    before = load_stream.LAST_META_INIT["skipped"]
    undo = load_stream._install_meta_init()
    try:
        assert torch.nn.init.kaiming_uniform_ is not original
        with torch.device("meta"):
            torch.nn.Linear(16, 8)
        assert load_stream.LAST_META_INIT["skipped"] >= before + 2  # kaiming_uniform_ and the bias' uniform_
        torch.manual_seed(7)
        got = torch.nn.Linear(16, 8)
        assert torch.get_rng_state().equal(expected_rng)
        assert torch.equal(got.weight, expected.weight) and torch.equal(got.bias, expected.bias)
    finally:
        undo()
    assert torch.nn.init.kaiming_uniform_ is original


def test_meta_init_same_setup(patched, tiny_dir, monkeypatch):
    """The whole setup with SOUP_SOLUTION_META_INIT: the generators after the load and the training step
    are the same as without it."""
    from solution import load_stream
    from solution.head_stream import streamer_for
    from solution.weight_stream import streamer_of

    def run(flag):
        monkeypatch.setenv("SOUP_SOLUTION_META_INIT", flag)
        seen = {}

        def after_load(model):
            seen["rng"] = (torch.get_rng_state(), torch.cuda.get_rng_state())

        torch.manual_seed(11)
        before = load_stream.LAST_META_INIT["skipped"]
        model, _ = _setup(patched, tiny_dir, after_load=after_load)
        seen["skipped"] = load_stream.LAST_META_INIT["skipped"] - before
        step = _step(model)
        streamer_of(model).restore()
        streamer_for(model.get_base_model().get_input_embeddings().weight).restore()
        return seen, step

    ours, ours_step = run("1")
    theirs, theirs_step = run("0")
    assert ours["skipped"] > 0 and theirs["skipped"] == 0
    assert all(a.equal(b) for a, b in zip(ours["rng"], theirs["rng"]))
    assert len(ours_step) == len(theirs_step) and all(torch.equal(a, b) for a, b in zip(ours_step, theirs_step))
