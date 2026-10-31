"""trainer_patch.py: the fused step's TRL metrics and Soup's rewind rows kept on the GPU until TRL's
log() give the values TRL's and Soup's immediate reads give; neither the deferred metrics nor a
forward planned on the host (step_pipeline.py) reads the GPU."""

import collections
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _trainer():
    return types.SimpleNamespace(
        _metrics={"train": collections.defaultdict(list), "eval": collections.defaultdict(list)},
        _total_train_tokens=0,
        accelerator=types.SimpleNamespace(gather_for_metrics=lambda t: t))


def _steps(seed=0):
    """(inputs, step) pairs: varying masks, one step without labelled tokens."""
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for i, (rows, cols) in enumerate(((4, 37), (4, 512), (2, 9), (4, 128))):
        mask = (torch.rand(rows, cols, generator=g) > 0.3).long()
        labelled = (torch.rand(rows, cols, generator=g) > (1.0 if i == 2 else 0.5)) & mask.bool()
        correct = (torch.rand(rows, cols, generator=g) > 0.6).int() * labelled.int()
        step = types.SimpleNamespace(token_entropy=(torch.rand(rows, cols, generator=g) * 5).cuda(),
                                     labelled=labelled.cuda(), correct=correct.cuda())
        pairs.append(({"attention_mask": mask.cuda()}, step))
    return pairs


def _immediate(trainer, mode, inputs, step):
    """trainer_patch._log_trl_metrics as it was in main: .item() right after the forward."""
    metrics, gather = trainer._metrics, trainer.accelerator.gather_for_metrics
    attention_mask = inputs.get("attention_mask")
    entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
    tokens = gather(attention_mask.sum()).sum().item()
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train":
        trainer._total_train_tokens += tokens
    metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


@pytest.mark.parametrize("defer", [True, False])
def test_metrics_equal_the_immediate_ones(defer, monkeypatch):
    from solution import trainer_patch

    monkeypatch.setitem(trainer_patch.DEFER, "on", defer)
    ref, got = _trainer(), _trainer()
    for inputs, step in _steps():
        _immediate(ref, "train", inputs, step)
        trainer_patch._log_trl_metrics(got, "train", inputs, step)
    if defer:
        assert got._metrics["train"] == {} and got._total_train_tokens == 0  # nothing read yet
    trainer_patch.flush_deferred(got)
    assert list(got._metrics["train"]) == list(ref._metrics["train"])  # the same key order
    assert got._metrics["train"] == ref._metrics["train"]
    assert got._total_train_tokens == ref._total_train_tokens
    assert type(got._total_train_tokens) is int
    assert got._metrics["train"]["mean_token_accuracy"][2] == 0.0
    assert all(type(v) is float for k in ("entropy", "mean_token_accuracy")
               for v in got._metrics["train"][k])
    trainer_patch.flush_deferred(got)  # nothing pending: no change
    assert got._metrics["train"] == ref._metrics["train"]


def test_deferred_metrics_do_not_read_the_gpu(monkeypatch):
    from solution import trainer_patch

    monkeypatch.setitem(trainer_patch.DEFER, "on", True)
    trainer = _trainer()
    pairs = _steps(seed=1)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for inputs, step in pairs:
            trainer_patch._log_trl_metrics(trainer, "train", inputs, step)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    trainer_patch.flush_deferred(trainer)
    assert len(trainer._metrics["train"]["entropy"]) == len(pairs)


class _Sink:
    def __init__(self):
        self.records = []

    def record_batch(self, **kwargs):
        self.records.append(kwargs)


def test_rewind_rows_deferred_to_the_log(monkeypatch):
    rewind_hf = pytest.importorskip("soup_cli.trainer.rewind_hf")
    from solution import trainer_patch

    trainer_patch._patch_rewind()
    original = rewind_hf._soup_original_row_losses
    g = torch.Generator().manual_seed(7)
    batches = []
    for rows, cols in ((4, 33), (3, 64), (4, 9)):
        labels = torch.randint(0, 50, (rows, cols), generator=g)
        labels[torch.rand(rows, cols, generator=g) > 0.6] = -100
        labels[0] = -100  # a row without labels
        token_loss = torch.rand(rows, cols, generator=g) * 3
        batches.append((token_loss.cuda(), labels.cuda()))

    ref_state, got_state = rewind_hf.RewindState(_Sink()), rewind_hf.RewindState(_Sink())
    for state in (ref_state, got_state):
        for i in range(11):
            state.on_index(i)
    for step, (token_loss, labels) in enumerate(batches, start=1):
        row_loss, row_tokens = original(token_loss, labels)
        ref_state.on_micro_batch(step=step, row_loss=row_loss, row_tokens=row_tokens)
    # (a record taken right away flushes the queued ones first, to keep their order)
    for step, (token_loss, labels) in enumerate(batches, start=1):
        monkeypatch.setitem(trainer_patch.DEFER, "armed", token_loss)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            row_loss, row_tokens = rewind_hf.row_losses_from_token_losses(token_loss, labels)
            got_state.on_micro_batch(step=step, row_loss=row_loss, row_tokens=row_tokens)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert got_state._sink.records == []  # queued
    trainer_patch.flush_deferred()
    assert got_state._sink.records == ref_state._sink.records
    assert len(got_state._sink.records) == 3 and got_state.dropped == ref_state.dropped == 0
    # not armed: Soup's own function and record, right away
    token_loss, labels = batches[0]
    assert rewind_hf.row_losses_from_token_losses(token_loss, labels) == original(token_loss, labels)


def test_planned_forward_does_not_read_the_gpu(tmp_path, monkeypatch):
    from solution import fused_ce, step_pipeline, unpad
    from solution.test_head_stream import _load_4bit, _peft, _peft_original_prepare
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=1000, hidden_size=128, intermediate_size=384,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=True)
    Qwen2ForCausalLM(config).half().save_pretrained(tmp_path)
    fused_ce.patch_qwen2_causal_lm()
    unpad.patch_qwen2_attention()
    model = _peft(_peft_original_prepare()(_load_4bit(tmp_path), use_gradient_checkpointing=False))
    model.train()
    g = torch.Generator().manual_seed(3)
    lengths = (40, 31, 17)
    ids = torch.randint(0, 1000, (3, 40), generator=g)
    mask = (torch.arange(40)[None, :] < torch.tensor(lengths)[:, None]).long()
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :5] = -100
    cpu = {"input_ids": ids.pin_memory(), "attention_mask": mask.pin_memory(), "labels": labels.pin_memory()}
    monkeypatch.setitem(step_pipeline._STATE, "plan", True)

    def run(planned):
        gpu = {k: v.cuda() for k, v in cpu.items()}
        if planned:
            step_pipeline._attach(step_pipeline._cpu_plan(cpu), gpu)
        items = gpu["labels"][..., 1:].ne(-100).sum()
        if planned:
            step_pipeline.remember(items, int(cpu["labels"][..., 1:].ne(-100).sum()))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**gpu, num_items_in_batch=items)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        step = fused_ce.LAST_STEP["step"]
        out.loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        return out.loss.detach(), step, grads

    with pytest.raises(RuntimeError):
        run(planned=False)  # unpad.pack_batch reads the mask's row lengths from the GPU
    model.zero_grad(set_to_none=True)
    loss, step, grads = run(planned=True)
    torch.cuda.set_sync_debug_mode("default")
    monkeypatch.setitem(step_pipeline._STATE, "plan", False)
    ref_loss, ref_step, ref_grads = _run_unplanned(model, cpu)
    assert torch.equal(loss, ref_loss)
    for name in ("token_loss", "token_entropy", "correct", "labelled"):
        assert torch.equal(getattr(step, name), getattr(ref_step, name))
    assert grads.keys() == ref_grads.keys()
    assert all(torch.equal(grads[n], ref_grads[n]) for n in grads)


def _run_unplanned(model, cpu):
    from solution import fused_ce

    gpu = {k: v.cuda() for k, v in cpu.items()}
    items = gpu["labels"][..., 1:].ne(-100).sum()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(**gpu, num_items_in_batch=items)
    step = fused_ce.LAST_STEP["step"]
    out.loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return out.loss.detach(), step, grads
