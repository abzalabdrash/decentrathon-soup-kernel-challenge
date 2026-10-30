"""The AdamW moments of the training kept in pinned host memory between optimizer steps.

Soup's optimizer is torch's AdamW (`adamw_torch`, the foreach implementation) over the LoRA
parameters: 2.18 M fp32 values, so `exp_avg` and `exp_avg_sq` hold 17.4 MB on the GPU for the whole
run, through the forward and backward peaks where nothing reads them. Here a pre-step hook brings
them to the GPU right before each `optimizer.step()` and a post-step hook copies them back to pinned
host memory and drops the GPU tensors: they are on the GPU only during the step, after the backward,
when the activations are gone. The copies run on the current stream, ordered with the step's kernels
(17.4 MB each way, ~1.5 ms each at 12 GB/s); the step reads and writes the same values, so the
parameters, the moments and every later step are bit for bit what they were.

SOUP_SOLUTION_ADAM_HOST=1 turns it on (off by default until measured); only torch.optim.AdamW whose
parameters are all CUDA tensors and whose state holds plain CUDA tensors is touched, any other
optimizer runs as it is. Installed around Soup's training by offload.py's train wrapper.
"""

import contextlib
import os
import weakref

import torch

ON = os.environ.get("SOUP_SOLUTION_ADAM_HOST", "0").strip().lower() not in ("0", "false", "no", "")
MOMENTS = ("exp_avg", "exp_avg_sq")


class _HostMoments:
    def __init__(self):
        self.host = {}  # (id(param), name) -> pinned host tensor
        self.moved = 0  # bytes moved per step each way
        self.steps = 0
        self.optimizers = weakref.WeakSet()

    @staticmethod
    def _usable(optimizer):
        if type(optimizer) is not torch.optim.AdamW:
            return False
        return all(p.is_cuda for group in optimizer.param_groups for p in group["params"])

    def before(self, optimizer, args, kwargs):
        if not self._usable(optimizer) or not self.host:
            return
        self.restore(optimizer)

    def restore(self, optimizer):
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                for name in MOMENTS:
                    host = self.host.get((id(p), name))
                    if host is not None and state.get(name) is host:
                        state[name] = host.to(p.device, non_blocking=True)

    def after(self, optimizer, args, kwargs):
        if not self._usable(optimizer):
            return
        moved = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if not state:
                    continue
                for name in MOMENTS:
                    value = state.get(name)
                    if (not torch.is_tensor(value) or not value.is_cuda
                            or type(value) is not torch.Tensor):
                        continue
                    key = (id(p), name)
                    host = self.host.get(key)
                    if host is None or host.shape != value.shape or host.dtype != value.dtype:
                        host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                        self.host[key] = host
                    host.copy_(value, non_blocking=True)
                    state[name] = host  # the GPU tensor goes back to the allocator, stream-ordered
                    moved += value.numel() * value.element_size()
        self.moved, self.steps = moved, self.steps + 1
        self.optimizers.add(optimizer)


@contextlib.contextmanager
def moments_on_host():
    """While active (and SOUP_SOLUTION_ADAM_HOST=1), AdamW's moments live on the host between
    steps; after it, they are put back on the GPU (the state as torch would have it)."""
    if not ON:
        yield None
        return
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)

    moments = _HostMoments()
    handles = (register_optimizer_step_pre_hook(moments.before),
               register_optimizer_step_post_hook(moments.after))
    try:
        yield moments
    finally:
        for handle in handles:
            handle.remove()
        for optimizer in list(moments.optimizers):
            moments.restore(optimizer)
        torch.cuda.synchronize()
