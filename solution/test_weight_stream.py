"""Streaming of the 4-bit layer weights (weight_stream.py) against the same layers resident on the GPU:
outputs and gradients bit for bit, memory, the ordering of the copies against the compute, and a tiny
real Qwen2 in 4 bits through peft, the HF forward, unpad.py's layer loop and Soup's setup hooks."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# the task's layer: hidden 1536, intermediate 8960, k/v 256 wide (2 kv heads of 128)
HIDDEN, INTER, KV = 1536, 8960, 256


def _linear4bit(k, n, bias, seed):
    """A bnb Linear4bit (NF4, double quant, fp16 compute) holding a random weight, quantized by the
    move to CUDA, as transformers loads the task's model."""
    import bitsandbytes as bnb

    gen = torch.Generator().manual_seed(seed)
    layer = bnb.nn.Linear4bit(k, n, bias=bias, compute_dtype=torch.float16, quant_type="nf4",
                              compress_statistics=True)
    weight = (torch.randn(n, k, generator=gen) * 0.02).half()
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4",
                                     compress_statistics=True, module=layer)
    if bias:
        layer.bias = torch.nn.Parameter(torch.randn(n, generator=gen) * 0.01, requires_grad=False)
    return layer.cuda()


class _Sleep(torch.autograd.Function):
    """Keeps the GPU busy before what follows, in the forward and in the backward, so that the CPU
    queues far ahead of the compute: a copy that does not wait for the compute lands too early."""

    @staticmethod
    def forward(ctx, x, cycles):
        ctx.cycles = cycles
        torch.cuda._sleep(cycles)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        torch.cuda._sleep(ctx.cycles)
        return grad, None


class Block(torch.nn.Module):
    """A decoder-layer-shaped block: the seven 4-bit projections of a Qwen2 layer and a LoRA on q and
    v, fp32 residual stream, fp16 matmuls; every projection's input gradient reaches the block input."""

    def __init__(self, seed, hidden=HIDDEN, inter=INTER, kv=KV, rank=16, sleep=0):
        super().__init__()
        self.q_proj = _linear4bit(hidden, hidden, True, seed * 10 + 1)
        self.k_proj = _linear4bit(hidden, kv, True, seed * 10 + 2)
        self.v_proj = _linear4bit(hidden, kv, True, seed * 10 + 3)
        self.o_proj = _linear4bit(hidden, hidden, False, seed * 10 + 4)
        self.gate_proj = _linear4bit(hidden, inter, False, seed * 10 + 5)
        self.up_proj = _linear4bit(hidden, inter, False, seed * 10 + 6)
        self.down_proj = _linear4bit(inter, hidden, False, seed * 10 + 7)
        gen = torch.Generator().manual_seed(seed * 10 + 8)
        self.lora = torch.nn.ParameterList([
            torch.nn.Parameter((torch.randn(rank, hidden, generator=gen) * 0.02).cuda()),
            torch.nn.Parameter((torch.randn(hidden, rank, generator=gen) * 0.02).cuda()),
            torch.nn.Parameter((torch.randn(rank, hidden, generator=gen) * 0.02).cuda()),
            torch.nn.Parameter((torch.randn(kv, rank, generator=gen) * 0.02).cuda()),
        ])
        self.sleep = sleep

    @staticmethod
    def _norm(h):
        return (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.float16)

    def forward(self, h):
        if self.sleep:
            h = _Sleep.apply(h, self.sleep)
        x = self._norm(h)
        qa, qb, va, vb = self.lora
        q = self.q_proj(x).float() + (x.float() @ qa.t()) @ qb.t()
        k = self.k_proj(x).float()
        v = self.v_proj(x).float() + (x.float() @ va.t()) @ vb.t()
        mix = k.mean(-1, keepdim=True) * v.mean(-1, keepdim=True)
        h = h + self.o_proj((torch.tanh(q) * (1 + mix)).half()).float()
        x = self._norm(h)
        h = h + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)).float()
        if self.sleep:
            h = _Sleep.apply(h, self.sleep)
        return h


def make_stack(n_layers=6, sleep=0, **shape):
    return torch.nn.ModuleList([Block(i, sleep=sleep, **shape) for i in range(n_layers)])


def run_stack(blocks, hidden):
    """The decoder loop of unpad.varlen_model_forward."""
    for block in blocks:
        hidden = block(hidden)
    return hidden


def _step(blocks, x, weight):
    """One forward and backward: output, input gradient and LoRA gradients (cloned)."""
    for p in blocks.parameters():
        p.grad = None
    x = x.detach().requires_grad_(True)
    out = run_stack(blocks, x)
    (out * weight).sum().backward()
    grads = [x.grad.clone()] + [p.grad.clone() for p in blocks.parameters() if p.requires_grad]
    return out.detach().clone(), grads


def _equal(a, b):
    out_a, grads_a = a
    out_b, grads_b = b
    return torch.equal(out_a, out_b) and len(grads_a) == len(grads_b) and all(
        torch.equal(ga, gb) for ga, gb in zip(grads_a, grads_b))


def _inputs(tokens=256, hidden=HIDDEN, seed=5):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(1, tokens, hidden, generator=gen).cuda()
    weight = torch.randn(1, tokens, hidden, generator=gen).cuda()
    return x, weight


@pytest.fixture(params=["nf4", "bnb"])
def matmul_path(request, monkeypatch):
    """The patched MatMul4Bit of nf4.py, or bitsandbytes' own: both save a view of the packed weight
    in the forward and read it in the backward."""
    from bitsandbytes.autograd import _functions as bnb_autograd

    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    if request.param == "bnb":
        cls = bnb_autograd.MatMul4Bit

        class Original(torch.autograd.Function):
            forward = staticmethod(cls._soup_original_forward)
            backward = staticmethod(cls._soup_original_backward)

        monkeypatch.setattr(bnb_autograd, "MatMul4Bit", Original)
    return request.param


@pytest.mark.parametrize("slots", [2, 3])
def test_stream_matches_resident(slots, matmul_path):
    from solution.weight_stream import WeightStreamer

    blocks = make_stack()
    x, weight = _inputs()
    reference = _step(blocks, x, weight)
    streamer = WeightStreamer(blocks, slots=slots)
    try:
        first = _step(blocks, x, weight)
        second = _step(blocks, x, weight)  # starts from the slots the backward left
        assert _equal(reference, first)
        assert _equal(reference, second)
        n = len(blocks)
        assert streamer.late_loads == 0  # every copy was started ahead of its layer
        assert streamer.loads == slots + 2 * 2 * (n - slots)
        # a forward without a backward (eval) leaves the last layers in the ring; the next step
        # loads layer 0 on the spot and is still exact
        with torch.no_grad():
            assert torch.equal(run_stack(blocks, x), reference[0])
        assert _equal(reference, _step(blocks, x, weight))
        assert streamer.late_loads > 0
    finally:
        streamer.restore()
    assert _equal(reference, _step(blocks, x, weight))


def test_stream_memory():
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.weight_stream import WeightStreamer

    patch_bnb_matmul4bit()
    blocks = make_stack()
    x, weight = _inputs()
    _step(blocks, x, weight)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    _step(blocks, x, weight)
    base_peak = torch.cuda.max_memory_allocated()

    streamer = WeightStreamer(blocks, slots=2)
    try:
        assert streamer._host.is_pinned() and streamer.host_bytes == sum(streamer.layer_bytes)
        freed = base_allocated - torch.cuda.memory_allocated()
        streamed = sum(streamer.layer_bytes) - streamer.slots * streamer.slot_bytes
        assert streamer.layer_bytes[0] > 24_000_000  # 46.8 M NF4 weights and their scales
        assert freed >= 0.99 * streamed
        torch.cuda.reset_peak_memory_stats()
        _step(blocks, x, weight)
        peak = torch.cuda.max_memory_allocated()
        assert base_peak - peak >= 0.99 * streamed
        print(f"\nlayer {streamer.layer_bytes[0] / 2**20:.2f} MiB, freed {freed / 2**20:.1f} MiB, "
              f"peak {base_peak / 2**20:.1f} -> {peak / 2**20:.1f} MiB")
    finally:
        streamer.restore()


# about 10 ms of GPU time per call on a ~1.5 GHz card; the whole stack runs 24 of them per step
SLEEP_CYCLES = 15_000_000


@pytest.mark.parametrize("slots", [2, 3])
def test_slot_not_overwritten_while_read(slots):
    """With the GPU far behind the CPU, a copy that did not wait for the compute queued before it
    would overwrite a slot that a layer (forward or backward) has not read yet."""
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.weight_stream import WeightStreamer

    patch_bnb_matmul4bit()
    blocks = make_stack(sleep=SLEEP_CYCLES)
    x, weight = _inputs()
    reference = _step(blocks, x, weight)
    streamer = WeightStreamer(blocks, slots=slots)
    try:
        assert _equal(reference, _step(blocks, x, weight))
        assert _equal(reference, _step(blocks, x, weight))
    finally:
        streamer.restore()


def test_race_check_detects_a_missing_wait():
    """The check above catches the bug it is there for: copies that skip the wait for the compute
    (events that are never recorded, so the copy stream waits for nothing) corrupt the result."""
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.weight_stream import WeightStreamer

    class NeverRecorded(torch.cuda.Event):
        def record(self, stream=None):
            pass

    patch_bnb_matmul4bit()
    blocks = make_stack(sleep=SLEEP_CYCLES)
    x, weight = _inputs()
    reference = _step(blocks, x, weight)
    streamer = WeightStreamer(blocks, slots=2)
    try:
        streamer._free = [NeverRecorded() for _ in streamer._free]
        assert not _equal(reference, _step(blocks, x, weight))
    finally:
        streamer.restore()


def test_exit_without_restore():
    """After training the weights stay streamed; the process exits cleanly (the host memory is
    unpinned at exit while CUDA is still up)."""
    import pathlib
    import subprocess
    import sys

    code = (
        "import torch\n"
        "from solution.test_weight_stream import make_stack, _inputs, _step\n"
        "from solution.weight_stream import WeightStreamer\n"
        "blocks = make_stack(4, hidden=256, inter=512, kv=64)\n"
        "x, w = _inputs(tokens=32, hidden=256)\n"
        "streamer = WeightStreamer(blocks, slots=2)\n"
        "_step(blocks, x, w)\n"
        "torch.cuda.synchronize()\n"
        "print('done', streamer.loads)\n"
    )
    root = pathlib.Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                            timeout=300)
    assert result.returncode == 0, result.stderr
    assert "done 6" in result.stdout
    assert "Error" not in result.stderr and "Traceback" not in result.stderr


# a tiny Qwen2 in 4 bits, loaded the way Soup loads the task's model


@pytest.fixture(scope="module")
def tiny_qwen_dir(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=384, num_hidden_layers=5,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=True)
    path = tmp_path_factory.mktemp("tiny_qwen2")
    Qwen2ForCausalLM(config).half().save_pretrained(path)
    return path


def _load_4bit(path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, dtype=torch.float16,
                                                device_map={"": 0})


def _peft(model):
    """Soup's adapter (r=16 on q_proj and v_proj; dropout 0 here so that runs compare exactly), with
    random B so that A gets a gradient too."""
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    gen = torch.Generator().manual_seed(2)
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.copy_(torch.randn(param.shape, generator=gen) * 0.02)
    return model


def _soup_like(path):
    from peft import prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(_load_4bit(path), use_gradient_checkpointing=False)
    return _peft(model)


def _batch(vocab=512):
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, vocab, (3, 40), generator=gen).cuda()
    mask = torch.ones_like(ids)
    mask[1, 31:] = 0
    mask[2, 17:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    return ids, mask, labels


def _hf_step(model):
    ids, mask, labels = _batch()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    loss.backward()
    return [loss.detach().clone()] + [p.grad.clone() for _, p in model.named_parameters()
                                      if p.requires_grad]


def _varlen_step(model):
    """The decoder through unpad.varlen_model_forward with the solution's layer patches."""
    from solution.unpad import pack_batch, varlen_model_forward

    ids, mask, _ = _batch()
    packed, pos, _, spans = pack_batch(ids, mask)
    decoder = model.get_base_model().model
    gen = torch.Generator().manual_seed(4)
    weight = torch.randn(1, packed.shape[1], decoder.config.hidden_size, generator=gen).cuda()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = varlen_model_forward(decoder, packed, pos, spans)
    (hidden.float() * weight).sum().backward()
    return [hidden.detach().clone()] + [p.grad.clone() for _, p in model.named_parameters()
                                        if p.requires_grad]


def _same(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _apply_layer_patches():
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope
    from solution.unpad import patch_qwen2_attention

    patch_qwen2_attention()
    patch_qwen2_mlp_half()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    patch_bnb_matmul4bit()
    patch_qwen2_rope()


@pytest.mark.parametrize("path", ["hf", "varlen"])
def test_tiny_qwen2_4bit(tiny_qwen_dir, path):
    from solution.weight_stream import find_decoder_layers, stream_model_weights, streamer_of

    if path == "varlen":
        _apply_layer_patches()
    step = _hf_step if path == "hf" else _varlen_step
    model = _soup_like(tiny_qwen_dir)
    assert len(find_decoder_layers(model)) == 5
    reference = step(model)
    allocated = torch.cuda.memory_allocated()
    streamer = stream_model_weights(model, slots=2)
    assert streamer is not None and streamer_of(model) is streamer
    assert stream_model_weights(model, slots=2) is streamer  # once per model
    try:
        # 7 linear layers per decoder layer, 3 tensors each
        assert all(len(plan) == 21 for plan in streamer._plans)
        assert torch.cuda.memory_allocated() < allocated
        assert _same(reference, step(model))
        assert _same(reference, step(model))
        assert streamer.late_loads == 0
    finally:
        streamer.restore()
    assert streamer_of(model) is None
    assert _same(reference, step(model))


def test_early_hook_and_adapter_save(tiny_qwen_dir, tmp_path, monkeypatch):
    """Soup's order with the early hook: load, (stream,) prepare_model_for_kbit_training,
    get_peft_model. The step and the saved adapter equal those of a run without streaming."""
    import peft
    from safetensors.torch import load_file
    from soup_cli.trainer import sft as soup_sft

    from solution.weight_stream import patch_weight_stream, streamer_of

    cls = soup_sft.SFTTrainerWrapper
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", peft.prepare_model_for_kbit_training)
    monkeypatch.setattr(cls, "_setup_transformers", cls._setup_transformers)
    monkeypatch.setattr(cls, "_soup_weight_stream", False, raising=False)

    reference_model = _soup_like(tiny_qwen_dir)
    reference = _hf_step(reference_model)
    reference_model.save_pretrained(tmp_path / "reference")
    del reference_model

    patch_weight_stream(slots=2, early=True)
    patch_weight_stream(slots=2, early=True)  # idempotent
    # with gradient checkpointing (peft's default) nothing is streamed
    plain = peft.prepare_model_for_kbit_training(_load_4bit(tiny_qwen_dir))
    assert streamer_of(plain) is None
    del plain

    model = _load_4bit(tiny_qwen_dir)
    allocated = torch.cuda.memory_allocated()
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    streamer = streamer_of(model)
    assert streamer is not None
    assert torch.cuda.memory_allocated() < allocated
    model = _peft(model)
    assert streamer_of(model) is streamer
    try:
        model.save_pretrained(tmp_path / "streamed")
        assert _same(reference, _hf_step(model))
    finally:
        streamer.restore()
    saved = load_file(tmp_path / "streamed" / "adapter_model.safetensors")
    expected = load_file(tmp_path / "reference" / "adapter_model.safetensors")
    assert sorted(saved) == sorted(expected)
    assert all(torch.equal(saved[k], expected[k]) for k in saved)


def test_setup_hook_leaves_other_models_alone(monkeypatch, tiny_qwen_dir):
    from soup_cli.trainer import sft as soup_sft

    from solution.weight_stream import patch_weight_stream, stream_model_weights, streamer_of

    cls = soup_sft.SFTTrainerWrapper
    models = iter([object(), None, torch.nn.Linear(4, 4)])

    def fake_setup(self, *args, **kwargs):
        self.model = next(models)
        return "setup result"

    monkeypatch.setattr(cls, "_setup_transformers", fake_setup)
    monkeypatch.setattr(cls, "_soup_weight_stream", False, raising=False)
    patch_weight_stream(slots=2, early=False)
    for _ in range(3):
        wrapper = cls.__new__(cls)
        assert cls._setup_transformers(wrapper) == "setup result"

    assert stream_model_weights(object()) is None
    assert stream_model_weights(torch.nn.Linear(4, 4).cuda()) is None
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=2, num_key_value_heads=1)
    messages = []
    assert stream_model_weights(Qwen2ForCausalLM(config).cuda(), log=messages.append) is None
    assert "without 4-bit" in messages[0]
    checkpointed = _load_4bit(tiny_qwen_dir)
    checkpointed.gradient_checkpointing_enable()
    assert stream_model_weights(checkpointed, log=messages.append) is None
    assert streamer_of(checkpointed) is None
    assert "checkpointing" in messages[-1]
