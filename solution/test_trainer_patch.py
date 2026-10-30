"""trainer_patch.py: the fused step's TRL metrics kept on the GPU until TRL's log() and then the same
values as TRL's own immediate .item() path; no host sync between the forward and the backward."""

import collections
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

_MISSING = object()


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
        entropy = torch.rand(rows, cols, generator=g) * 5
        labelled = torch.tensor(0 if i == 2 else int(mask.sum()) - rows)
        correct = torch.tensor(int(labelled) // 3)
        step = types.SimpleNamespace(token_entropy=entropy.cuda(), labelled=labelled.cuda(),
                                     correct=correct.cuda())
        pairs.append(({"attention_mask": mask.cuda()}, step))
    return pairs


def _immediate(trainer, mode, inputs, step):
    """trainer_patch._log_trl_metrics as it was: .item() right after the forward."""
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


def test_deferred_metrics_equal_the_immediate_ones():
    from solution.trainer_patch import _log_trl_metrics, flush_pending_metrics

    ref, got = _trainer(), _trainer()
    for inputs, step in _steps():
        _immediate(ref, "train", inputs, step)
        _log_trl_metrics(got, "train", inputs, step)
    assert got._metrics["train"] == {} and got._total_train_tokens == 0  # nothing read yet
    flush_pending_metrics(got)
    assert list(got._metrics["train"]) == list(ref._metrics["train"])  # the same key order
    assert got._metrics["train"] == ref._metrics["train"]
    assert got._total_train_tokens == ref._total_train_tokens
    assert type(got._total_train_tokens) is int
    assert got._metrics["train"]["mean_token_accuracy"][2] == 0.0
    assert all(type(v) is float for k in ("entropy", "mean_token_accuracy")
               for v in got._metrics["train"][k])
    flush_pending_metrics(got)  # nothing pending: no change
    assert got._metrics["train"] == ref._metrics["train"]


def test_no_host_sync_between_forward_and_backward():
    from solution.trainer_patch import _log_trl_metrics, flush_pending_metrics

    trainer = _trainer()
    pairs = _steps(seed=1)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for inputs, step in pairs:
            _log_trl_metrics(trainer, "train", inputs, step)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    flush_pending_metrics(trainer)
    assert len(trainer._metrics["train"]["entropy"]) == len(pairs)


@pytest.fixture
def restore_sft_trainer():
    import trl

    names = ("compute_loss", "log", "_soup_fused_metrics")
    saved = {name: vars(trl.SFTTrainer).get(name, _MISSING) for name in names}
    yield trl.SFTTrainer
    for name, value in saved.items():
        if value is _MISSING:
            if name in vars(trl.SFTTrainer):
                delattr(trl.SFTTrainer, name)
        else:
            setattr(trl.SFTTrainer, name, value)


def test_trl_log_gets_python_numbers(restore_sft_trainer, monkeypatch):
    """TRL's own SFTTrainer.log, reached through the patch, averages plain floats and ints: the
    pending device scalars are read right before it, every step."""
    from transformers import Trainer

    from solution import trainer_patch

    cls = restore_sft_trainer
    if "_soup_fused_metrics" in vars(cls):
        delattr(cls, "_soup_fused_metrics")
    trainer_patch.patch_sft_trainer()
    logged = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, start_time=None: logged.append(logs))
    trainer = cls.__new__(cls)
    trainer.__dict__.update(vars(_trainer()))
    trainer.model = torch.nn.Linear(1, 1).train()
    ref = _trainer()
    for inputs, step in _steps(seed=2):
        _immediate(ref, "train", inputs, step)
        trainer_patch._log_trl_metrics(trainer, "train", inputs, step)
        cls.log(trainer, {"loss": 1.0})
        expected = {k: sum(v) / len(v) for k, v in ref._metrics["train"].items()}
        ref._metrics["train"].clear()
        assert logged[-1] == {"loss": 1.0, **expected}
        assert all(type(v) in (int, float) for v in logged[-1].values())
