"""The streamed head (head_stream.py): the online pass over vocabulary chunks against the base's head
on the task's shapes, the embedding lookup from host memory, and Soup's setup order on a tiny 4-bit
Qwen2 (the weight off the GPU in a training forward, the step, eval logits, the saved adapter, the
guards and the fallback)."""

import copy
import gc

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

V, D = 151936, 1536
SCALE = 65536.0  # the AMP loss scale of a T4 run: without it the base's fp16 softmax gradient underflows


@pytest.fixture(scope="module")
def streamed():
    """A (V, D) fp16 weight on the GPU for the references, and a Parameter with the same values
    streamed from pinned host memory."""
    from solution.head_stream import HeadStreamer

    torch.manual_seed(0)
    weight = (torch.randn(V, D, device="cuda") * 0.02).half()
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    streamer = HeadStreamer(param, rows=8192, slots=2)
    yield weight, param, streamer
    streamer.restore()


def _targets(n, pattern, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    targets = torch.randint(0, V, (n,), device="cuda", generator=gen)
    cut = int(0.3 * n)
    if pattern == "unlabelled_first":  # a prompt
        targets[:cut] = -100
    elif pattern == "unlabelled_last":  # e.g. a padding tail of dummy tokens
        targets[n - cut:] = -100
    elif pattern == "mixed":
        targets[torch.rand(n, device="cuda", generator=gen) < 0.5] = -100
    elif pattern == "none_labelled":
        targets[:] = -100
    else:
        assert pattern == "all_labelled"
    return targets


def _reference(hidden, weight, targets, num_items):
    """The base's head (fused_ce.reference_lm_head_loss) in blocks of rows, with the AMP scale on the
    backward as in a real fp16 step."""
    from solution.fused_ce import reference_lm_head_loss

    labelled = targets != -100
    denom = float(num_items) if num_items is not None else max(int(labelled.sum()), 1)
    grad = torch.zeros_like(hidden)
    total, token_loss, entropy, correct = 0.0, [], [], []
    for start in range(0, hidden.shape[0], 256):
        rows = hidden[start:start + 256].detach().clone().requires_grad_(True)
        loss, tl, en, co = reference_lm_head_loss(rows, weight, targets[start:start + 256], denom)
        (loss * SCALE).backward()
        grad[start:start + 256] = rows.grad / SCALE
        total += loss.item()
        token_loss.append(tl)
        entropy.append(en)
        correct.append(co)
    return total, torch.cat(token_loss), torch.cat(entropy), torch.cat(correct), grad


def _chunked_statistics(hidden, weight, targets, rows):
    """Per-token loss, entropy and correctness in fp32 from the logits the streamed head computes:
    the same rows (labelled first), the same GEMM per vocabulary chunk, hence the same fp16 bits."""
    labelled = targets != -100
    order = torch.argsort(labelled.to(torch.int8), descending=True, stable=True)
    h16, t = hidden.detach()[order].half(), targets[order]
    logits = torch.empty(h16.shape[0], V, dtype=torch.float16, device="cuda")
    for c0 in range(0, V, rows):
        logits[:, c0:c0 + rows] = torch.mm(h16, weight[c0:c0 + rows].t())
    token_loss, entropy, correct = [], [], []
    for start in range(0, h16.shape[0], 256):
        block, tb = logits[start:start + 256].float(), t[start:start + 256]
        token_loss.append(F.cross_entropy(block, tb, ignore_index=-100, reduction="none"))
        logp = F.log_softmax(block, dim=-1)
        entropy.append(-(logp.exp() * logp).sum(-1))
        correct.append(((block.argmax(-1) == tb) & (tb != -100)).to(torch.int32))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return tuple(torch.cat(x)[inverse] for x in (token_loss, entropy, correct))


def _exact_grad(hidden, weight, targets, rows, denom):
    """dL/dh in fp64 from the fp16 logits, for the given rows."""
    h16 = hidden[rows].detach().half()
    logits = (h16 @ weight.t()).double()
    p = torch.softmax(logits, dim=-1)
    p[torch.arange(len(rows)), targets[rows]] -= 1.0
    return (p @ weight.double()) / denom


@pytest.mark.parametrize("n,pattern,rows,scale,num_items", [
    (354, "unlabelled_first", 8192, 2.0, None),
    (826, "unlabelled_last", 5000, 8.0, 900.0),
    (1427, "all_labelled", 16384, 2.0, None),
    (826, "mixed", 3000, 30.0, None),  # peaked rows: the running max moves between chunks
    (97, "none_labelled", 8192, 2.0, None),
])
def test_streamed_head_matches_base(streamed, n, pattern, rows, scale, num_items):
    from solution.fused_ce import fused_lm_head_loss

    weight, param, streamer = streamed
    streamer.configure(rows=rows, slots=2)
    torch.manual_seed(n)
    hidden = (torch.randn(n, D, device="cuda") * scale).requires_grad_(True)
    targets = _targets(n, pattern, seed=n)
    labelled = targets != -100

    loss, token_loss, entropy, correct = fused_lm_head_loss(hidden, param, targets, num_items)
    (loss * SCALE).backward()
    grad = hidden.grad / SCALE
    ref_loss, ref_token_loss, ref_entropy, ref_correct, ref_grad = _reference(
        hidden, weight, targets, num_items)

    # statistics from the same fp16 logits: fp32 both ways, only the summation order differs
    chunk_loss, chunk_entropy, chunk_correct = _chunked_statistics(hidden, weight, targets, rows)
    torch.testing.assert_close(token_loss, chunk_loss, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, chunk_entropy, rtol=1e-5, atol=5e-5)
    assert torch.equal(correct, chunk_correct)
    assert torch.all(token_loss[~labelled] == 0)
    # against the base's head over all of V at once: cuBLAS may pick another GEMM kernel for a
    # chunk of the vocabulary (the T4 does), so a few fp16 logits differ in their last bit (0.06 at
    # |x| = 87, 0.125 at 150): the loss agrees to 1e-4, per-token values to that rounding
    assert abs(loss.item() - ref_loss) <= 1e-4 * max(abs(ref_loss), 1.0)
    torch.testing.assert_close(token_loss, ref_token_loss, rtol=1e-3, atol=5e-2)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-3, atol=5e-2)
    assert int((correct != ref_correct).sum()) <= 1  # a near-tie of fp16 logits may flip
    if not labelled.any():
        assert torch.all(hidden.grad == 0)
        return
    assert torch.all(grad[~labelled] == 0)
    # dL/dh: the base rounds the gradient of the logits and the GEMM's output to fp16; ours rounds
    # P to fp16 and accumulates in fp32. Both within fp16 rounding of the exact value, ours closer.
    top = ref_grad.abs().max()
    assert (grad - ref_grad).abs().max() / top < 4e-3
    rows_checked = labelled.nonzero().squeeze(1)[:48]
    denom = float(num_items) if num_items is not None else int(labelled.sum())
    exact = _exact_grad(hidden, weight, targets, rows_checked, denom)
    ours_err = (grad[rows_checked].double() - exact).abs().max()
    base_err = (ref_grad[rows_checked].double() - exact).abs().max()
    assert ours_err <= max(base_err * 1.5, 1e-3 * top)


def test_streamed_head_counts_and_ring(streamed):
    """Every chunk is copied once per head, ahead of its use once the first ones are prefetched, and
    the ring is gone after the head."""
    from solution.fused_ce import fused_lm_head_loss

    _, param, streamer = streamed
    streamer.configure(rows=8192, slots=2)
    hidden = torch.randn(64, D, device="cuda")
    targets = _targets(64, "mixed", seed=3)
    streamer.prefetch()  # what a training embedding lookup does
    loads, late = streamer.loads, streamer.late_loads
    fused_lm_head_loss(hidden, param, targets)
    assert streamer.loads - loads == streamer.n_chunks - 2
    assert streamer.late_loads == late
    assert streamer._ring is None
    stats = streamer.stats()
    assert stats["copies_timed"] == stats["loads"]


def test_embedding_lookup_is_exact(streamed):
    weight, param, streamer = streamed
    gen = torch.Generator(device="cuda").manual_seed(5)
    for shape in [(1, 826), (4, 512), (7,)]:
        ids = torch.randint(0, V, shape, device="cuda", generator=gen)
        ids.view(-1)[0], ids.view(-1)[-1] = 0, V - 1
        out = streamer.lookup(ids)
        assert out.dtype == torch.float32 and out.shape == (*shape, D)
        assert torch.equal(out, F.embedding(ids, weight.float()))
        assert torch.equal(out, F.embedding(ids, weight).float())


def test_placeholder_and_deepcopy(streamed):
    weight, param, streamer = streamed
    assert param.is_cuda and param.dtype == torch.float32 and param.shape == (V, D)
    assert param.untyped_storage().nbytes() == 4
    copied = copy.deepcopy(param)
    assert torch.equal(copied, weight.float()) and copied.requires_grad == param.requires_grad
    assert torch.equal(streamer.linear(torch.eye(D, device="cuda")[:3]), weight.float()[:, :3].t())


def test_kernel_argmax_ties_across_chunks():
    """torch.argmax keeps the first index of the maximum; so must the chunked pass, inside a chunk
    and across chunks (the second chunk's equal maximum must not win)."""
    from solution.head_stream import (BLOCK_D, IGNORE_INDEX, head_chunk_kernel, head_finish_kernel,
                                      _num_warps)

    rows, vocab, chunk = 4, 10000, 4096
    logits = torch.zeros(rows, vocab, device="cuda", dtype=torch.float16)
    logits[0, 10] = logits[0, 5000] = 3.0  # tie across chunks
    logits[1, 4100] = logits[1, 4200] = 2.0  # tie inside the second chunk
    logits[2, vocab - 1] = 1.0  # last column, partial chunk
    logits[3, 4095], logits[3, 4096] = 5.0, 6.0  # max at the start of a chunk
    targets = torch.tensor([10, 4200, vocab - 1, -100], device="cuda")
    state = torch.empty(5, 4, device="cuda")
    idx = torch.empty(rows, dtype=torch.int32, device="cuda")
    dummy32 = torch.zeros(1, 16, device="cuda")
    dummy16 = torch.zeros(1, 16, device="cuda", dtype=torch.float16)
    block = 4096
    for c, c0 in enumerate(range(0, vocab, chunk)):
        vc = min(chunk, vocab - c0)
        part = logits[:, c0:c0 + vc].contiguous()
        head_chunk_kernel[(rows,)](part, part.stride(0), targets, *state, idx, dummy32, 16, dummy16, 16,
                                   dummy16, 16, c0, vc, 0, 16, FIRST=c == 0, IGNORE=IGNORE_INDEX,
                                   BLOCK=block, BLOCK_D=BLOCK_D, num_warps=_num_warps(block))
    loss = torch.empty(rows, device="cuda")
    entropy = torch.empty(rows, device="cuda")
    correct = torch.empty(rows, dtype=torch.int32, device="cuda")
    head_finish_kernel[(rows,)](targets, state[0], state[1], state[2], state[4], idx, loss, entropy, correct,
                                dummy32, 16, dummy16, 16, 1.0, 0, 16, IGNORE=IGNORE_INDEX, BLOCK_D=BLOCK_D)
    assert idx.tolist() == logits.float().argmax(-1).tolist() == [10, 4100, vocab - 1, 4096]
    assert correct.tolist() == [1, 0, 1, 0]
    ref = F.cross_entropy(logits.float(), targets, reduction="none", ignore_index=-100)
    torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5)
    logp = F.log_softmax(logits.float(), -1)
    torch.testing.assert_close(entropy, -(logp.exp() * logp).sum(-1), rtol=1e-5, atol=1e-5)


# ---- Soup's setup order on a tiny 4-bit Qwen2 ------------------------------------------------------

TINY_VOCAB, TINY_ROWS = 1000, 96  # 11 chunks, the last one partial


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=TINY_VOCAB, hidden_size=128, intermediate_size=384,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(4.0)  # livelier logits than the init's 0.02
    path = tmp_path_factory.mktemp("tiny_qwen2_head")
    model.half().save_pretrained(path)
    return path


def _load_4bit(path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, dtype=torch.float16,
                                                device_map={"": 0})


def _peft_original_prepare():
    import peft

    prepare = peft.prepare_model_for_kbit_training
    while hasattr(prepare, "_soup_original"):
        prepare = prepare._soup_original
    return prepare


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


def _soup_like(path, stream, before_finish=None):
    """load, (stream,) prepare_model_for_kbit_training, get_peft_model, (finish_setup)."""
    from solution.fused_ce import patch_qwen2_causal_lm
    from solution.head_stream import finish_setup, stream_before_prepare
    from solution.unpad import patch_qwen2_attention

    patch_qwen2_causal_lm()
    patch_qwen2_attention()
    model = _load_4bit(path)
    streamer = stream_before_prepare(model, rows=TINY_ROWS, slots=2) if stream else None
    if stream:
        assert streamer is not None
    model = _peft(_peft_original_prepare()(model, use_gradient_checkpointing=False))
    if before_finish is not None:
        before_finish(model)
    if stream:
        assert (finish_setup(model) is streamer) == (before_finish is None)
    return model, streamer


def _batch(lengths=(40, 31, 17)):
    gen = torch.Generator().manual_seed(3)
    ids = torch.randint(0, TINY_VOCAB, (len(lengths), max(lengths)), generator=gen).cuda()
    mask = (torch.arange(ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]).long().cuda()
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :5] = -100  # a prompt
    return ids, mask, labels


def _train_step(model):
    from solution import fused_ce

    ids, mask, labels = _batch()
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
    assert out.logits is None
    (out.loss * SCALE).backward()
    step = fused_ce.LAST_STEP["step"]
    grads = {n: p.grad / SCALE for n, p in model.named_parameters() if p.requires_grad}
    return out.loss.detach(), step, grads


def _weights_on_gpu(vocab, hidden, known):
    """CUDA tensors alive with the shape of the head's weight and storage to hold it (other than the
    `known` data pointers)."""
    found = []
    for obj in gc.get_objects():
        try:
            if (isinstance(obj, torch.Tensor) and obj.is_cuda and tuple(obj.shape) == (vocab, hidden)
                    and obj.untyped_storage().nbytes() >= vocab * hidden * 2
                    and obj.data_ptr() not in known):
                found.append(obj)
        except Exception:  # noqa: BLE001 - objects that fail isinstance or attribute access
            pass
    return found


def test_tiny_qwen2_streamed_step(tiny_dir, tmp_path):
    from safetensors.torch import load_file

    reference, _ = _soup_like(tiny_dir, stream=False)
    reference.save_pretrained(tmp_path / "reference")
    ref_loss, ref_step, ref_grads = _train_step(reference)

    model, streamer = _soup_like(tiny_dir, stream=True)
    base = model.get_base_model()
    param = base.get_input_embeddings().weight
    assert base.lm_head.weight is param and param.dtype == torch.float32 and param.is_cuda
    assert param.untyped_storage().nbytes() == 4 and model.dtype == reference.dtype
    model.save_pretrained(tmp_path / "streamed")

    seen, ring = [], []
    known = {p.data_ptr() for p in reference.parameters()}
    hooks = [base.model.norm.register_forward_hook(
                 lambda *_: seen.append(len(_weights_on_gpu(TINY_VOCAB, 128, known)))),
             base.model.layers[-1].register_forward_hook(lambda *_: ring.append(streamer._ring)),
             base.model.norm.register_forward_hook(lambda *_: ring.append(streamer._ring))]
    try:
        loss, step, grads = _train_step(model)
    finally:
        for hook in hooks:
            hook.remove()
    assert seen == [0]  # nothing of the weight's size on the GPU at the end of the forward
    # the first chunks start at the final norm: no ring through the layers (the last MLP's peak)
    assert ring[0] is None and ring[1] is not None
    assert streamer.late_loads == 0 and streamer.heads == 1 and streamer.loads == streamer.n_chunks

    torch.testing.assert_close(loss, ref_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(step.token_loss, ref_step.token_loss, rtol=5e-4, atol=2e-3)
    torch.testing.assert_close(step.token_entropy, ref_step.token_entropy, rtol=5e-4, atol=2e-3)
    assert int((step.correct != ref_step.correct).sum()) <= 1
    assert torch.equal(step.labelled, ref_step.labelled)
    assert sorted(grads) == sorted(ref_grads)
    bad = [n for n, g in ref_grads.items() if not torch.isfinite(g).all()]
    bad_ours = [n for n, g in grads.items() if not torch.isfinite(g).all()]
    assert not bad and not bad_ours, (bad[:3], bad_ours[:3])
    for name, g in grads.items():
        err = (g - ref_grads[name]).abs().max() / ref_grads[name].abs().max().clamp(min=1e-12)
        assert err < 2e-2, f"{name}: relative gradient difference {err:.4f}"

    saved = load_file(tmp_path / "streamed" / "adapter_model.safetensors")
    expected = load_file(tmp_path / "reference" / "adapter_model.safetensors")
    assert sorted(saved) == sorted(expected)
    assert all(torch.equal(saved[k], expected[k]) for k in saved)

    # eval / generation: the original forward through the lm_head override, the base's dtypes
    ids, mask, _ = _batch()
    model.eval()
    reference.eval()
    with torch.no_grad():
        for autocast in (True, False):
            with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
                ours = model(input_ids=ids, attention_mask=mask).logits
                theirs = reference(input_ids=ids, attention_mask=mask).logits
            assert ours.dtype == theirs.dtype
            torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-3)
    streamer.restore()
    assert param.untyped_storage().nbytes() == TINY_VOCAB * 128 * 4
    assert torch.equal(param, reference.get_base_model().get_input_embeddings().weight)


@pytest.mark.parametrize("where", ["embedding", "head"])
def test_prefetch_points(tiny_dir, monkeypatch, where):
    """SOUP_SOLUTION_HEAD_PREFETCH=embedding: the ring exists through the layers; =head: the head
    copies its first chunks itself. The step is the same."""
    monkeypatch.setenv("SOUP_SOLUTION_HEAD_PREFETCH", where)
    model, streamer = _soup_like(tiny_dir, stream=True)
    assert streamer.prefetch_at == where and streamer._hook is None
    ring = []
    hook = model.get_base_model().model.layers[-1].register_forward_hook(
        lambda *_: ring.append(streamer._ring is not None))
    try:
        _train_step(model)
    finally:
        hook.remove()
    assert ring == [where == "embedding"]
    assert streamer.late_loads == 0 and streamer.loads == streamer.n_chunks
    streamer.restore()


def test_trainable_or_wrapped_head_is_restored(tiny_dir):
    from solution.head_stream import streamer_for

    def unfreeze(model):
        model.get_base_model().get_input_embeddings().weight.requires_grad_(True)

    model, streamer = _soup_like(tiny_dir, stream=True, before_finish=unfreeze)
    base = model.get_base_model()
    param = base.get_input_embeddings().weight
    assert streamer_for(param) is None and streamer.host is None
    assert param.dtype == torch.float32 and param.untyped_storage().nbytes() == TINY_VOCAB * 128 * 4
    assert "forward" not in base.get_input_embeddings().__dict__
    assert "forward" not in base.lm_head.__dict__
    assert "__deepcopy__" not in param.__dict__
    fresh = _load_4bit(tiny_dir).get_input_embeddings().weight.float()
    assert torch.equal(param, fresh)

    # peft's modules_to_save deep-copies the lm_head while the weight is in host memory: the copy
    # holds the real values, and the check after the setup takes the weight back
    def wrap(model):
        from peft.utils import ModulesToSaveWrapper

        inner = model.get_base_model()
        inner.lm_head = ModulesToSaveWrapper(inner.lm_head, "default")
        copied = inner.lm_head.modules_to_save["default"].weight
        assert copied.untyped_storage().nbytes() == TINY_VOCAB * 128 * 4
        assert torch.equal(copied, fresh)

    model, streamer = _soup_like(tiny_dir, stream=True, before_finish=wrap)
    assert streamer.host is None
    assert torch.equal(model.get_base_model().get_input_embeddings().weight, fresh)


def test_guards(tiny_dir, monkeypatch):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from solution.fused_ce import patch_qwen2_causal_lm
    from solution.head_stream import _SETUP, patch_head_stream, stream_before_prepare, streamer_for

    patch_qwen2_causal_lm()
    messages = []
    assert stream_before_prepare(object(), log=messages.append) is None
    assert stream_before_prepare(torch.nn.Linear(4, 4).cuda(), log=messages.append) is None
    def config(tied=True):
        return Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                           num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=tied)

    cpu = Qwen2ForCausalLM(config()).half()
    assert stream_before_prepare(cpu, log=messages.append) is None
    untied = Qwen2ForCausalLM(config(tied=False)).half().cuda()
    assert stream_before_prepare(untied, log=messages.append) is None
    fp32 = Qwen2ForCausalLM(config()).cuda()
    assert stream_before_prepare(fp32, log=messages.append) is None
    gptq = Qwen2ForCausalLM(config()).half().cuda()
    gptq.quantization_method = "gptq"
    assert stream_before_prepare(gptq, log=messages.append) is None
    assert len(messages) == 6 and all("skipped" in m for m in messages)
    for model in (cpu, untied, fp32, gptq):
        assert streamer_for(model.get_input_embeddings().weight) is None

    # the prepare hook acts only inside Soup's setup
    import peft

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", _peft_original_prepare())
    patch_head_stream(rows=TINY_ROWS)
    model = _load_4bit(tiny_dir)
    model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    assert _SETUP["depth"] == 0
    assert streamer_for(model.get_input_embeddings().weight) is None
    assert model.get_input_embeddings().weight.untyped_storage().nbytes() == TINY_VOCAB * 128 * 4


def test_setup_hook_with_fake_models(monkeypatch):
    """Soup's regression tests run _setup_transformers with fake modules: they pass through."""
    from soup_cli.trainer import sft as soup_sft

    from solution.head_stream import _SETUP, patch_head_stream

    cls = soup_sft.SFTTrainerWrapper
    models = iter([object(), None, torch.nn.Linear(4, 4)])
    depth = []

    def fake_setup(self, *args, **kwargs):
        depth.append(_SETUP["depth"])
        self.model = next(models)
        return "setup result"

    monkeypatch.setattr(cls, "_setup_transformers", fake_setup)
    monkeypatch.setattr(cls, "_soup_head_stream", False, raising=False)
    patch_head_stream()
    for _ in range(3):
        wrapper = cls.__new__(cls)
        assert cls._setup_transformers(wrapper) == "setup result"
    assert depth == [1, 1, 1] and _SETUP["depth"] == 0
