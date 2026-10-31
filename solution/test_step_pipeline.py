"""step_pipeline.py: the host plan is unpad.pack_batch's, the GPU nan/inf filter adds what
transformers' host filter adds, the prefetching iterator yields the DataLoader's batches in the
DataLoader's order, and a tiny SFT run with every B1 switch on logs, records and trains exactly
what it does with them off."""

import types

import pytest
import torch

from solution import step_pipeline, unpad


def _batch(lengths, width, seed, left=False, prompt=3):
    g = torch.Generator().manual_seed(seed)
    rows = len(lengths)
    ids = torch.randint(1, 500, (rows, width), generator=g)
    cols = torch.arange(width)[None, :]
    lens = torch.tensor(lengths)[:, None]
    mask = ((cols >= width - lens) if left else (cols < lens)).long()
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :prompt] = -100
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


pytest_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

CASES = [((40, 31, 17, 5), 40, False), ((512, 3, 200, 1), 512, False), ((7, 0, 7, 2), 9, False),
         ((16, 16), 16, False), ((30, 12, 1), 30, True), ((33,), 64, False)]


@pytest_cuda  # the plan's buffer is pinned
@pytest.mark.parametrize("lengths,width,left", CASES)
@pytest.mark.parametrize("pad", [16, 0])
def test_cpu_plan_matches_pack_batch(lengths, width, left, pad, monkeypatch):
    monkeypatch.setitem(unpad._STATE, "pad", pad)
    batch = _batch(lengths, width, seed=len(lengths) + width, left=left)
    cpu = step_pipeline._cpu_plan(batch)
    assert cpu is not None
    ids, pos, idx, spans = unpad.pack_batch(batch["input_ids"], batch["attention_mask"])
    n = cpu["n"]
    host = cpu["host"]
    assert host.is_pinned()
    assert torch.equal(host[:n], ids[0]) and torch.equal(host[n:2 * n], pos[0])
    assert torch.equal(host[2 * n:], idx)
    assert cpu["spans"] == spans
    bounds = unpad.sequences(spans, n + unpad._pad_count(n))
    assert cpu["bounds"] == bounds
    assert torch.equal(cpu["cu"], torch.tensor(bounds, dtype=torch.int32))
    from solution.fused_ce import shift_labels

    targets = shift_labels(batch["labels"]).flatten()[idx]
    assert cpu["n_labelled"] == int((targets != -100).sum())
    assert cpu["items_shifted"] == int(batch["labels"][..., 1:].ne(-100).sum())
    assert cpu["items_all"] == int(batch["labels"].ne(-100).sum())


@pytest_cuda
def test_cpu_plan_declines_what_pack_batch_would_not_see():
    batch = _batch((5, 3), 6, seed=0)
    assert step_pipeline._cpu_plan({**batch, "shift_labels": batch["labels"]}) is None
    assert step_pipeline._cpu_plan({k: v for k, v in batch.items() if k != "labels"}) is None
    assert step_pipeline._cpu_plan({**batch, "input_ids": batch["input_ids"].int()}) is None
    assert step_pipeline._cpu_plan({**batch, "labels": batch["labels"][:, :-1]}) is None
    assert step_pipeline._cpu_plan({**batch, "attention_mask": torch.zeros(2, 6, dtype=torch.long)}) is None
    assert step_pipeline._cpu_plan([batch]) is None


def _trainer_like(global_step, last_logged, tr_loss):
    return types.SimpleNamespace(state=types.SimpleNamespace(global_step=global_step),
                                 _globalstep_last_logged=last_logged, _tr_loss=tr_loss)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gpu_filter_adds_what_the_host_filter_adds(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    losses = [2.5, float("nan"), 1.25, float("inf"), float("-inf"), 0.75, float("nan"), 3.0]
    host = torch.tensor(0.0, device=device)
    gpu = torch.tensor(0.0, device=device)
    last_logged = 0
    for step, value in enumerate(losses):
        loss = torch.tensor(value, device=device)
        # transformers' loop with logging_nan_inf_filter
        if torch.isnan(loss) or torch.isinf(loss):
            host += host / (1 + step - last_logged)
        else:
            host += loss
        gpu += step_pipeline.gpu_filtered_loss(_trainer_like(step, last_logged, gpu), loss)
        assert torch.equal(host, gpu), step
        if step % 3 == 2:  # a log resets the running loss
            host -= host
            gpu -= gpu
            last_logged = step + 1


class _Recording(torch.utils.data.Sampler):
    def __init__(self, n, seen):
        self.n, self.seen = n, seen

    def __iter__(self):
        for i in torch.randperm(self.n, generator=torch.Generator().manual_seed(5)).tolist():
            self.seen.append(i)
            yield i

    def __len__(self):
        return self.n


def _loader(seen, n=10, batch=3):
    data = [{"input_ids": torch.full((4,), i), "attention_mask": torch.ones(4, dtype=torch.long),
             "labels": torch.full((4,), i)} for i in range(n)]

    def collate(rows):
        return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}

    return torch.utils.data.DataLoader(data, batch_size=batch, sampler=_Recording(n, seen),
                                       collate_fn=collate)


@pytest.mark.parametrize("pattern", ["never", "always", "every_other", "twice"])
def test_prefetching_iterator_keeps_batches_and_order(pattern):
    seen_ref, seen = [], []
    expected = [b["input_ids"][:, 0].tolist() for b in _loader(seen_ref)]
    it = step_pipeline._Prefetching(iter(_loader(seen)))
    got = []
    for i in range(10):
        if pattern == "always" or (pattern == "every_other" and i % 2) or pattern == "twice":
            it.prefetch()
            if pattern == "twice":
                it.prefetch()  # a second prefetch fetches nothing more
        try:
            got.append(next(it)["input_ids"][:, 0].tolist())
        except StopIteration:
            break
    assert got == expected
    assert seen == seen_ref  # the sampler's indices in the same order
    with pytest.raises(StopIteration):
        next(it)
    it.prefetch()
    with pytest.raises(StopIteration):
        next(it)


def test_prefetching_iterator_raises_where_the_loader_would():
    class Boom(Exception):
        pass

    def inner():
        yield {"x": 1}
        raise Boom

    it = step_pipeline._Prefetching(inner())
    assert next(it) == {"x": 1}
    it.prefetch()  # the error is kept for the next request
    with pytest.raises(Boom):
        next(it)


def test_known_values_follow_the_tensor_object():
    t = torch.zeros(3)
    step_pipeline.remember(t, 7)
    assert step_pipeline.known(t) == 7
    assert step_pipeline.known(torch.zeros(3)) is None
    assert step_pipeline.known(None) is None and step_pipeline.known(7) is None
    key = id(t)
    del t
    assert key not in step_pipeline._KNOWN


# ---- a tiny SFT run, every switch off against every switch on ------------------------------


class _Sink:
    def __init__(self):
        self.records = []

    def record_batch(self, **kwargs):
        self.records.append(kwargs)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(TINY_VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="t0", eos_token="t1")


TINY_VOCAB = 1000


def _dataset(rows=21, seed=4):
    from datasets import Dataset

    g = torch.Generator().manual_seed(seed)
    data = {"input_ids": [], "completion_mask": []}
    for _ in range(rows):
        length = int(torch.randint(6, 60, (1,), generator=g))
        prompt = int(torch.randint(1, length - 2, (1,), generator=g))
        data["input_ids"].append(torch.randint(2, TINY_VOCAB, (length,), generator=g).tolist())
        data["completion_mask"].append([0] * prompt + [1] * (length - prompt))
    return Dataset.from_dict(data)


def _run(tiny_dir, tmp_path, switches, monkeypatch):
    import trl
    from soup_cli.trainer import rewind_hf

    from solution import boundary, trainer_patch
    from solution.test_head_stream import _load_4bit, _peft, _peft_original_prepare

    monkeypatch.setitem(step_pipeline._STATE, "nan_gpu", switches)
    monkeypatch.setitem(step_pipeline._STATE, "prefetch", switches)
    monkeypatch.setitem(step_pipeline._STATE, "plan", switches)
    monkeypatch.setitem(trainer_patch.DEFER, "on", switches)
    monkeypatch.setitem(boundary._STATE, "on", switches)
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)  # a reproducible attention backward
    for key in step_pipeline._STATS:
        monkeypatch.setitem(step_pipeline._STATS, key, type(step_pipeline._STATS[key])())

    model = _peft(_peft_original_prepare()(_load_4bit(tiny_dir), use_gradient_checkpointing=False))
    config = trl.SFTConfig(output_dir=str(tmp_path), per_device_train_batch_size=3, max_length=64,
                           fp16=True, logging_steps=1, max_grad_norm=1.0, learning_rate=2e-3,
                           num_train_epochs=1, report_to=[], save_strategy="no", seed=1,
                           completion_only_loss=True, dataloader_pin_memory=True,
                           disable_tqdm=True, gradient_checkpointing=False)
    cls = rewind_hf.make_rewind_trainer_class(trl.SFTTrainer)
    trainer = cls(model=model, args=config, train_dataset=_dataset(), processing_class=_tokenizer())
    for param in model.parameters():  # TRL casts QLoRA adapters to bf16; Soup's run trains them in fp32
        if param.requires_grad:
            param.data = param.data.float()
    sink = _Sink()
    rewind_hf.attach_rewind_state(trainer, sink)
    trainer.train()
    adapter = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    history = [dict(entry) for entry in trainer.state.log_history]
    stats = dict(step_pipeline._STATS)
    del trainer, model
    torch.cuda.empty_cache()
    return history, sink.records, adapter, stats


@pytest_cuda
def test_tiny_sft_run_identical_with_the_switches(tmp_path, monkeypatch):
    pytest.importorskip("trl")
    pytest.importorskip("soup_cli.trainer.rewind_hf")
    from solution import boundary, fused_ce, trainer_patch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=TINY_VOCAB, hidden_size=128, intermediate_size=384,
                         num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True)
    tiny = Qwen2ForCausalLM(config)
    with torch.no_grad():
        tiny.get_input_embeddings().weight.mul_(4.0)
    tiny_dir = tmp_path / "tiny"
    tiny.half().save_pretrained(tiny_dir)

    fused_ce.patch_qwen2_causal_lm()
    unpad.patch_qwen2_attention()
    monkeypatch.setitem(boundary._STATE, "on", True)
    monkeypatch.setitem(step_pipeline._STATE, "nan_gpu", True)
    trainer_patch.patch_sft_trainer()
    step_pipeline.unpatch_step_pipeline()  # in apply()'s order, whatever earlier tests left
    boundary.unpatch_step_boundary()
    boundary.patch_step_boundary()
    step_pipeline.patch_step_pipeline()
    trainer_patch._patch_rewind()
    try:
        off = _run(tiny_dir, tmp_path / "off", False, monkeypatch)
        on = _run(tiny_dir, tmp_path / "on", True, monkeypatch)
    finally:
        step_pipeline.unpatch_step_pipeline()
        boundary.unpatch_step_boundary()
    history_off, records_off, adapter_off, _ = off
    history_on, records_on, adapter_on, stats = on
    steps = 7  # 21 rows, batch 3
    assert stats["prefetched"] == steps - 2  # DataLoaderShard reads two itself, the end is found
    assert stats["planned"] == steps and stats["plans_used"] == steps
    assert stats["known_used"] >= 2 * steps  # the head's count and num_items_in_batch
    timing = ("train_runtime", "train_samples_per_second", "train_steps_per_second")
    for a, b in zip(history_off, history_on):
        assert {k: v for k, v in a.items() if k not in timing} == \
               {k: v for k, v in b.items() if k not in timing}
    assert len(history_off) == len(history_on) == steps + 1
    assert [list(h) for h in history_off] == [list(h) for h in history_on]  # same keys, same order
    assert records_off == records_on and len(records_on) == steps
    assert adapter_off.keys() == adapter_on.keys()
    assert all(torch.equal(adapter_off[n], adapter_on[n]) for n in adapter_off)


# ---- the backward on the calling thread -------------------------------------------------------


@pytest_cuda
@pytest.mark.parametrize("ring", ["type", None])
def test_single_thread_backward_same_step(layer_shaped_qwen_dir, night, ring):
    """SOUP_SOLUTION_BWD_SINGLE_THREAD: a packed step through the layer patches with the night
    defaults (attention recompute, the MLP offload's events, LoRA dropout) and the weight ring's
    events gives the same outputs and gradients bit for bit with the backward on the calling thread;
    the backward's hooks run there; one cuBLAS workspace (handle) fewer is made."""
    import threading

    from solution.test_weight_ring import _ring_step
    from solution.test_weight_stream import _same, _soup_like
    from solution.weight_stream import stream_model_weights

    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    streamer = stream_model_weights(model, slots=2, ring=ring) if ring else None
    lora = next(p for n, p in model.named_parameters() if p.requires_grad)
    threads = []
    handle = lora.register_hook(lambda g: threads.append(threading.get_ident()))

    def step(single):
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        before = torch.cuda.memory_allocated()
        if single:
            with torch.autograd.set_multithreading_enabled(False):
                out = _ring_step(model)
        else:
            out = _ring_step(model)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return out, torch.cuda.memory_allocated() - before

    try:
        ref, workspaces = step(False)
        assert threads[-1] != threading.get_ident()  # the engine's device thread
        got, workspaces_single = step(True)
        assert threads[-1] == threading.get_ident()
        again, _ = step(False)
    finally:
        handle.remove()
        if streamer is not None:
            streamer.restore()
    assert _same(ref, again)  # the step is reproducible here
    assert _same(ref, got)
    assert workspaces - workspaces_single >= 4 * 2**20  # the autograd thread's cuBLAS workspace


from solution.test_weight_ring import night  # noqa: E402,F401 - fixtures
from solution.test_weight_stream import layer_shaped_qwen_dir  # noqa: E402,F401
