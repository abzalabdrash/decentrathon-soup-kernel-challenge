"""TRL's SFTTrainer.compute_loss for a model that returns no logits.

TRL computes `entropy` and `mean_token_accuracy` from `outputs.logits` and raises without them. With
the fused head (fused_ce.py) the model leaves the per-token entropy, loss and correctness in
`fused_ce.LAST_STEP`; this compute_loss logs the same three columns with TRL's own formulas from those
values and hands Soup's rewind recorder the per-token losses through the attribute it already reads
(`_soup_chunked_step`, see soup_cli.trainer.rewind_hf: `row_losses_from_token_losses`).

Soup's trainer is a subclass of trl.SFTTrainer built at setup time; patching the attribute on the
base class reaches it because Python looks the method up on every call. A step that is not training,
has no labels, or where the model did not take the fused path goes through TRL's original code.
"""

from . import fused_ce

STEP_ATTR = "_soup_chunked_step"  # soup_cli.utils.stream_loss_chunk.STEP_ATTR


def _log_trl_metrics(trainer, mode, inputs, step):
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum()
        tokens = gather(attention_mask.sum()).sum().item()
    else:
        entropy = step.token_entropy.mean()
        tokens = None
    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train" and tokens is not None and hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]
    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


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
        if step is None:
            # the model kept its logits (not a Qwen2 causal LM, or the fused path declined): TRL's
            # metrics from the logits, as TRL computes them
            setattr(self, STEP_ATTR, None)
            _metrics_from_logits(self, mode, inputs, outputs)
        else:
            setattr(self, STEP_ATTR, step)
            _log_trl_metrics(self, mode, inputs, step)
        if getattr(self, "aux_loss_enabled", False) and getattr(outputs, "aux_loss", None) is not None:
            aux = self.accelerator.gather_for_metrics(outputs.aux_loss).mean().item()
            self._metrics[mode]["aux_loss"].append(aux)
        return (loss, outputs) if return_outputs else loss

    cls.compute_loss = compute_loss
    cls._soup_fused_metrics = True


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
