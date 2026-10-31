"""TRL's SFTTrainer.compute_loss for a model that returns no logits.

TRL computes `entropy` and `mean_token_accuracy` from `outputs.logits` and raises without them. With
the fused head (fused_ce.py) the model leaves the per-token entropy, loss and correctness in
`fused_ce.LAST_STEP`; this compute_loss logs the same three columns with TRL's own formulas from those
values and hands Soup's rewind recorder the per-token losses through the attribute it already reads
(`_soup_chunked_step`, see soup_cli.trainer.rewind_hf: `row_losses_from_token_losses`).

SOUP_SOLUTION_DEFER_METRICS=1: the values stay on the GPU until the step is logged. TRL reads them
with `.item()` and Soup's rewind recorder with `float()` between the forward and the backward, and
the first read waits for the whole forward and head to finish, so the CPU starts queueing the
backward with the GPU empty. Here `_log_trl_metrics` keeps detached device scalars, and the rewind
recorder gets the per-row means and token counts as device tensors (the same operations as Soup's
`row_losses_from_token_losses`), its record queued; `flush_deferred` turns them into the same Python
numbers, in the same order, into TRL's `_metrics` and Soup's `RewindState.on_micro_batch` right
before TRL's `log()` averages the metrics (every step here, logging_steps 1), when the harness has
already synchronized. The rewind log gets the same lines in the same order.

Soup's trainer is a subclass of trl.SFTTrainer built at setup time; patching the attribute on the
base class reaches it because Python looks the method up on every call. A step that is not training,
has no labels, or where the model did not take the fused path goes through TRL's original code.
"""

import os

from . import fused_ce

STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR
PENDING_ATTR = "_soup_pending_metrics"  # device scalars of the steps not logged yet
DEFER = {"on": os.environ.get("SOUP_SOLUTION_DEFER_METRICS", "0").strip().lower()
         not in ("0", "false", "no", ""),
         "armed": None,  # the token_loss tensor the rewind recorder may take deferred
         "rows": []}  # queued rewind records: (state, step, means, tokens)


def _log_trl_metrics(trainer, mode, inputs, step):
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
        tokens = gather(attention_mask.sum()).sum()
    else:
        entropy = step.token_entropy.mean()
        tokens = None
    entropy = gather(entropy).mean()
    labelled = gather(step.labelled).sum()
    accuracy = gather(step.correct).sum() / labelled  # used only when labelled > 0
    entry = (mode, entropy.detach(), tokens, labelled, accuracy.detach())
    if DEFER["on"]:
        trainer.__dict__.setdefault(PENDING_ATTR, []).append(entry)
    else:
        _record_metrics(trainer, entry)


def _record_metrics(trainer, entry, read=None):
    """One step's values into TRL's `_metrics` and `_total_train_tokens`, as TRL's compute_loss puts
    them there: entropy, num_tokens, mean_token_accuracy. `read` gives the values already read."""
    mode, entropy, tokens, labelled, accuracy = entry
    if read is None:
        read = [t.item() if t is not None else None for t in (entropy, tokens, labelled, accuracy)]
    entropy, tokens, labelled, accuracy = read
    metrics = trainer._metrics
    metrics[mode]["entropy"].append(entropy)
    if tokens is not None:
        if mode == "train" and hasattr(trainer, "_total_train_tokens"):
            trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    metrics[mode]["mean_token_accuracy"].append(accuracy if labelled > 0 else 0.0)


def read_all(tensors):
    """`[t.tolist() for t in tensors]` with one device-to-host copy: every value goes through float64,
    which holds fp32/fp16/bf16 values and integers below 2**53 exactly, and comes back as the Python
    float or int `.item()` / `.tolist()` gives."""
    import torch

    if not tensors:
        return []
    device = tensors[0].device
    if any(t.device != device or t.is_complex() or t.dtype == torch.float64 for t in tensors) or len(tensors) == 1:
        return [t.tolist() for t in tensors]
    flat = torch.cat([t.detach().reshape(-1).to(torch.float64) for t in tensors]).tolist()
    out, start = [], 0
    for t in tensors:
        n = t.numel()
        chunk = flat[start:start + n]
        start += n
        if not t.is_floating_point():
            chunk = [bool(v) for v in chunk] if t.dtype == torch.bool else [int(v) for v in chunk]
        out.append(chunk if t.dim() else chunk[0])
    return out


def flush_deferred(trainer=None):
    """The deferred rewind records (all trainers) and this trainer's deferred metrics, in order,
    read from the GPU in one copy."""
    rows, DEFER["rows"] = DEFER["rows"], []
    pending = trainer.__dict__.get(PENDING_ATTR) if trainer is not None else None
    if pending:
        trainer.__dict__[PENDING_ATTR] = []
    tensors = [t for _, _, means, tokens in rows for t in (means, tokens)]
    for entry in pending or ():
        tensors += [t for t in entry[1:] if t is not None]
    values = iter(read_all(tensors))
    if rows:
        _record_rewind(rows, values)
    for entry in pending or ():
        _record_metrics(trainer, entry, [next(values) if t is not None else None for t in entry[1:]])


# ---- Soup's rewind recorder ------------------------------------------------------------------


class _DeviceRows:
    """Per-row values still on the GPU (a sequence for whoever reads it before the flush)."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = values

    def __len__(self):
        return self.values.shape[0]

    def __iter__(self):
        return iter(self.values.tolist())


def _flush_rewind():
    rows, DEFER["rows"] = DEFER["rows"], []
    if rows:
        _record_rewind(rows, iter(read_all([t for _, _, m, k in rows for t in (m, k)])))


def _record_rewind(rows, values):
    from soup_cli.trainer import rewind_hf

    record = rewind_hf.RewindState._soup_original_on_micro_batch
    for state, step, _, _ in rows:
        means, tokens = next(values), next(values)
        try:
            # Soup: [float(mean) for mean in means], [int(count) for count in tokens]
            record(state, step=step, row_loss=means, row_tokens=tokens)
        except Exception as exc:  # noqa: BLE001 - Soup's compute_loss never raises into training
            state.fail(exc)


def _patch_rewind():
    try:
        from soup_cli.trainer import rewind_hf
    except Exception:  # noqa: BLE001 - no Soup rewind recorder
        return
    state_cls = rewind_hf.RewindState
    if getattr(state_cls, "_soup_original_on_micro_batch", None) is not None:
        return
    row_losses = rewind_hf.row_losses_from_token_losses
    on_micro_batch = state_cls.on_micro_batch

    def row_losses_from_token_losses(token_loss, labels):
        armed, DEFER["armed"] = DEFER["armed"], None
        if armed is None or token_loss is not armed:
            return row_losses(token_loss, labels)
        import torch

        with torch.no_grad():  # Soup's row_losses_from_token_losses, without its host reads
            tokens = (labels[:, 1:] != -100).sum(dim=1)
            totals = token_loss.float().sum(dim=1).to(tokens.device)
            means = totals / tokens.clamp(min=1).to(totals.dtype)
        return _DeviceRows(means), _DeviceRows(tokens)

    def on_micro_batch_deferred(self, *, step, row_loss, row_tokens):
        if isinstance(row_loss, _DeviceRows) and isinstance(row_tokens, _DeviceRows):
            DEFER["rows"].append((self, step, row_loss.values, row_tokens.values))
            return None
        _flush_rewind()  # keep the records in order
        return on_micro_batch(self, step=step, row_loss=row_loss, row_tokens=row_tokens)

    rewind_hf.row_losses_from_token_losses = row_losses_from_token_losses
    state_cls.on_micro_batch = on_micro_batch_deferred
    state_cls._soup_original_on_micro_batch = on_micro_batch
    rewind_hf._soup_original_row_losses = row_losses


def patch_sft_trainer():
    import trl
    from transformers import Trainer

    cls = trl.SFTTrainer
    if getattr(cls, "_soup_fused_metrics", False):
        return
    original = cls.compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        training = getattr(getattr(self, "model", None), "training", False)
        if (not training or not isinstance(inputs, dict) or "labels" not in inputs
                or "shift_labels" in inputs or getattr(self.args, "use_liger_kernel", False)):
            return original(self, model, inputs, return_outputs=return_outputs,
                            num_items_in_batch=num_items_in_batch)
        mode = "train"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False
        fused_ce.LAST_STEP["step"] = None
        loss, outputs = Trainer.compute_loss(self, model, inputs, return_outputs=True,
                                             num_items_in_batch=num_items_in_batch)
        step = fused_ce.LAST_STEP["step"]
        fused_ce.LAST_STEP["step"] = None
        DEFER["armed"] = None
        if step is None:
            # the model kept its logits (not a Qwen2 causal LM, or the fused path declined): TRL's
            # metrics from the logits, as TRL computes them
            setattr(self, STEP_ATTR, None)
            flush_deferred(self)  # keep the steps' order in _metrics
            _metrics_from_logits(self, mode, inputs, outputs)
        else:
            setattr(self, STEP_ATTR, step)
            if DEFER["on"]:
                DEFER["armed"] = step.token_loss  # Soup's recorder reads it right after this returns
            _log_trl_metrics(self, mode, inputs, step)
        if getattr(self, "aux_loss_enabled", False) and getattr(outputs, "aux_loss", None) is not None:
            aux = self.accelerator.gather_for_metrics(outputs.aux_loss).mean().item()
            self._metrics[mode]["aux_loss"].append(aux)
        return (loss, outputs) if return_outputs else loss

    original_log = cls.log

    def log(self, logs, *args, **kwargs):
        flush_deferred(self)
        return original_log(self, logs, *args, **kwargs)

    cls.compute_loss = compute_loss
    cls.log = log
    cls._soup_fused_metrics = True
    if DEFER["on"]:
        _patch_rewind()
    from .boundary import patch_step_boundary
    from .step_pipeline import patch_step_pipeline

    patch_step_boundary()  # SOUP_SOLUTION_STEP_BOUNDARY=1: the step boundary without model walks
    patch_step_pipeline()  # SOUP_SOLUTION_NAN_FILTER=gpu, _PREFETCH_BATCH=1, _HOST_PLAN=1


def _metrics_from_logits(trainer, mode, inputs, outputs):
    import torch
    from trl.trainer.utils import entropy_from_logits

    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("the model returned neither logits nor a fused step")
    labels = inputs["labels"]
    with torch.no_grad():
        per_token_entropy = entropy_from_logits(logits)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
            num_tokens = trainer.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        else:
            entropy = torch.mean(per_token_entropy)
            position_ids = inputs["position_ids"]
            num_tokens = trainer.accelerator.gather_for_metrics(
                torch.tensor(position_ids.size(1), device=position_ids.device)).sum().item()
        entropy = trainer.accelerator.gather_for_metrics(entropy).mean().item()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        predictions = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct = trainer.accelerator.gather_for_metrics(((predictions == shift_labels) & mask).sum())
        total = trainer.accelerator.gather_for_metrics(mask.sum())
        total_sum = total.sum()
        accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
    trainer._metrics[mode]["entropy"].append(entropy)
    if mode == "train":
        trainer._total_train_tokens += num_tokens
    trainer._metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    trainer._metrics[mode]["mean_token_accuracy"].append(accuracy)
