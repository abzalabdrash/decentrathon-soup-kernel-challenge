"""The step boundary of the training loop without its four walks over the whole model.

Between the last backward kernel of a step and the first forward kernel of the next, the GPU waits
for the CPU (about 26 ms per step on the T4 trace of main). A large part of that CPU time is four
walks over all ~930 modules and ~450 parameters of the 4-bit PEFT model, where every attribute read
on the 196 bitsandbytes Params4bit goes through its Python `__torch_function__` (night research,
docs/RESEARCH.md "Граница шага"):

- transformers' `Trainer.floating_point_ops` calls `model.num_parameters(exclude_embeddings=True)`
  every step for the `total_flos` counter: a walk over the modules and one over the parameters;
- `Trainer._clip_grad_norm` passes `model.parameters()` to accelerate / torch, which turn it into a
  list and read `.grad` of every parameter twice;
- `model.zero_grad()` reads `.grad` of every parameter;
- `Trainer.training_step` calls `model.train()`, a recursive `Module.__setattr__` on every module.

Here the same results come from lists made once per training run, for the real CUDA trainer only:
- `num_parameters(exclude_embeddings=True)` is kept per model: it depends only on which parameters
  are registered and their numel / element size, which nothing changes during training
  (weight_stream.py and head_stream.py swap `param.data` for tensors of the same shape);
- the clip and zero_grad get the trainable parameters in `model.parameters()` order: frozen
  parameters never hold a `.grad`, so torch builds the same gradient list, the same norm, the same
  clipped values, and frees the same gradients in the same order;
- `train(mode)` writes `training` into each module's `__dict__` (what `Module.__setattr__` does for a
  bool) for a model whose every module uses `nn.Module.train` and `nn.Module.__setattr__`.
Every list is dropped when a parameter or module is registered anywhere (torch's global
registration hooks) or when the model's trainable set no longer matches the optimizer's; anything
else (Soup's tests with fake trainers, other models, DoRA's own `train`) runs the original code.
SOUP_SOLUTION_STEP_BOUNDARY=1 turns it on (off by default until measured).
"""

import os
import weakref

import torch

_STATE = {"on": os.environ.get("SOUP_SOLUTION_STEP_BOUNDARY", "0").strip().lower()
          not in ("0", "false", "no", ""), "generation": 0, "hooks": None}
_CACHE = weakref.WeakKeyDictionary()  # model -> {"generation", "trainable", "modules", "flops_params"}


def _bump(*_args, **_kwargs):
    _STATE["generation"] += 1


def _install_hooks():
    if _STATE["hooks"] is None:
        from torch.nn.modules import module as nn_module

        _STATE["hooks"] = (nn_module.register_module_parameter_registration_hook(_bump),
                           nn_module.register_module_module_registration_hook(_bump))


def _entry(model):
    """The lists of `model` for the current registration generation (built on first use)."""
    entry = _CACHE.get(model)
    if entry is None or entry["generation"] != _STATE["generation"]:
        params = list(model.parameters())
        trainable = [p for p in params if p.requires_grad]
        frozen_with_grad = any(p.grad is not None for p in params if not p.requires_grad)
        modules = list(model.modules())
        plain = all(type(m).train is torch.nn.Module.train
                    and type(m).__setattr__ is torch.nn.Module.__setattr__ for m in modules)
        entry = {"generation": _STATE["generation"], "trainable": trainable,
                 "ids": frozenset(id(p) for p in trainable), "modules": modules if plain else None,
                 "usable": not frozen_with_grad, "flops_params": None}
        _CACHE[model] = entry
    return entry


def _active(model):
    return _STATE["on"] and model in _CACHE


def _matches_optimizer(trainer, entry):
    optimizer = getattr(trainer, "optimizer", None)
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return False
    ids = {id(p) for group in groups for p in group.get("params", ())}
    return ids == entry["ids"]


def patch_step_boundary():
    """Trainer.floating_point_ops / _clip_grad_norm and PeftModel.zero_grad / train (see the module
    doc); no-op unless SOUP_SOLUTION_STEP_BOUNDARY=1."""
    if not _STATE["on"]:
        return
    from transformers import Trainer

    if getattr(Trainer, "_soup_step_boundary", False):
        return
    try:
        from peft import PeftModel
    except Exception:  # noqa: BLE001 - no peft: nothing to patch
        return
    _install_hooks()
    flops_original = Trainer.floating_point_ops
    clip_original = Trainer._clip_grad_norm
    train_step_original = Trainer.training_step

    def training_step(self, model, inputs, *args, **kwargs):
        # the lists are made for the model this trainer trains on CUDA, once per generation
        if _STATE["on"] and isinstance(model, PeftModel):
            param = next(model.parameters(), None)
            if param is not None and param.is_cuda:
                _entry(model)
        return train_step_original(self, model, inputs, *args, **kwargs)

    def floating_point_ops(self, inputs):
        model = self.model
        main = getattr(model, "main_input_name", "input_ids")
        if not _active(model) or main not in inputs or not hasattr(model, "num_parameters"):
            return flops_original(self, inputs)
        entry = _entry(model)
        if entry["flops_params"] is None:
            entry["flops_params"] = model.num_parameters(exclude_embeddings=True)
        return 6 * inputs[main].numel() * entry["flops_params"]

    def _clip_grad_norm(self, model):
        if _active(model):
            from transformers.trainer import is_sagemaker_mp_enabled

            entry = _entry(model)
            if (entry["usable"] and not (is_sagemaker_mp_enabled() and self.args.fp16)
                    and _matches_optimizer(self, entry)):
                return self.accelerator.clip_grad_norm_(entry["trainable"], self.args.max_grad_norm)
        return clip_original(self, model)

    def zero_grad(self, set_to_none=True):
        if set_to_none and _active(self) and not getattr(self, "_is_replica", False):
            entry = _entry(self)
            if entry["usable"]:
                for p in entry["trainable"]:
                    if p.grad is not None:
                        p.grad = None
                return
        torch.nn.Module.zero_grad(self, set_to_none)

    def train(self, mode=True):
        if _active(self) and isinstance(mode, bool):
            modules = _entry(self)["modules"]
            if modules is not None:
                for m in modules:
                    m.__dict__["training"] = mode
                return self
        return torch.nn.Module.train(self, mode)

    Trainer.training_step = training_step
    Trainer.floating_point_ops = floating_point_ops
    Trainer._clip_grad_norm = _clip_grad_norm
    if "zero_grad" not in vars(PeftModel) and "train" not in vars(PeftModel):
        PeftModel.zero_grad = zero_grad
        PeftModel.train = train
        PeftModel._soup_step_boundary = True
    Trainer._soup_step_boundary = True
    Trainer._soup_step_boundary_previous = (train_step_original, flops_original, clip_original)


def unpatch_step_boundary():
    """Puts the originals back (tests)."""
    from transformers import Trainer

    if getattr(Trainer, "_soup_step_boundary", False):
        (Trainer.training_step, Trainer.floating_point_ops,
         Trainer._clip_grad_norm) = Trainer._soup_step_boundary_previous
        del Trainer._soup_step_boundary_previous
        Trainer._soup_step_boundary = False
    try:
        from peft import PeftModel
    except Exception:  # noqa: BLE001
        return
    if getattr(PeftModel, "_soup_step_boundary", False):
        del PeftModel.zero_grad, PeftModel.train, PeftModel._soup_step_boundary
