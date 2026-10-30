"""The AdamW moments of the training kept in pinned host memory between optimizer steps.

Soup's optimizer is torch's AdamW (`adamw_torch`, the foreach implementation) over the LoRA
parameters: 2.18 M fp32 values, so `exp_avg` and `exp_avg_sq` hold 17.4 MB on the GPU for the whole
run, through the forward and backward peaks where nothing reads them. Here a pre-step hook brings
them to the GPU right before each `optimizer.step()` and a post-step hook copies them back to pinned
host memory: they are on the GPU only during the step, after the backward, when the activations are
gone. All moments live in one flat fp32 buffer on each side (the state tensors are views into it),
so each way is one copy on the current stream (~1.5 ms at 12 GB/s) and the GPU side is one large
block the allocator can hand to the activations of the next step (a first version kept 112 separate
tensors: 112 small copies each way, +20 ms per step, and their small-pool blocks stayed reserved, so
nvidia-smi saw nothing). The step reads and writes the same values: parameters, moments and every
later step are bit for bit what they were.

SOUP_SOLUTION_ADAM_HOST=1 turns it on (off by default until measured); only torch.optim.AdamW whose
parameters are all on one CUDA device and whose moments are plain contiguous fp32 tensors is
touched; any other optimizer, or a state someone else replaced, runs as it is. Installed around
Soup's training by offload.py's train wrapper; when the training returns the moments are put back on
the GPU.
"""

import contextlib
import os
import weakref

import torch

ON = os.environ.get("SOUP_SOLUTION_ADAM_HOST", "0").strip().lower() not in ("0", "false", "no", "")
MOMENTS = ("exp_avg", "exp_avg_sq")


class _HostMoments:
    def __init__(self):
        self.optimizer = None  # weak reference to the optimizer whose moments are handled
        self.entries = None  # [(param, name, offset, numel, shape)]
        self.host = None  # pinned flat fp32 buffer
        self.gpu = None  # flat fp32 buffer on the GPU during a step
        self.moved = 0  # bytes per step each way
        self.steps = 0

    @staticmethod
    def _device(optimizer):
        if type(optimizer) is not torch.optim.AdamW:
            return None
        devices = {p.device for group in optimizer.param_groups for p in group["params"]}
        if len(devices) != 1:
            return None
        device = devices.pop()
        return device if device.type == "cuda" else None

    def _mine(self, optimizer):
        return self.optimizer is not None and self.optimizer() is optimizer

    def _views_are(self, optimizer, flat):
        for p, name, offset, numel, _ in self.entries:
            value = optimizer.state.get(p, {}).get(name)
            if value is None or value.data_ptr() != flat[offset:offset + numel].data_ptr():
                return False
        return True

    def _point(self, optimizer, flat):
        for p, name, offset, numel, shape in self.entries:
            optimizer.state[p][name] = flat[offset:offset + numel].view(shape)

    def before(self, optimizer, args, kwargs):
        device = self._device(optimizer)
        if device is None or not self._mine(optimizer) or self.gpu is not None:
            return
        if not self._views_are(optimizer, self.host):  # someone replaced a moment: leave it all
            self.optimizer = None
            return
        self.gpu = self.host.to(device, non_blocking=True)
        self._point(optimizer, self.gpu)

    def after(self, optimizer, args, kwargs):
        device = self._device(optimizer)
        if device is None:
            return
        if self.optimizer is None:  # the first step: AdamW made its moments as separate tensors
            entries, offset = [], 0
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p)
                    if not state:
                        continue
                    for name in MOMENTS:
                        value = state.get(name)
                        if (type(value) is not torch.Tensor or value.dtype != torch.float32
                                or not value.is_contiguous() or value.device != device):
                            return
                        entries.append((p, name, offset, value.numel(), value.shape))
                        offset += value.numel()
            if not entries:
                return
            self.entries = entries
            self.host = torch.empty(offset, dtype=torch.float32, pin_memory=True)
            for p, name, start, numel, _ in entries:
                self.host[start:start + numel].copy_(optimizer.state[p][name].view(-1),
                                                     non_blocking=True)
            self.optimizer = weakref.ref(optimizer)
        elif self._mine(optimizer) and self.gpu is not None:
            if not self._views_are(optimizer, self.gpu):
                self.gpu, self.optimizer = None, None
                return
            self.host.copy_(self.gpu, non_blocking=True)
            self.gpu = None  # back to the allocator, ordered after the copy on this stream
        else:
            return
        self._point(optimizer, self.host)
        self.moved, self.steps = self.host.numel() * 4, self.steps + 1

    def restore(self):
        optimizer = self.optimizer() if self.optimizer is not None else None
        if optimizer is None or self.gpu is not None or not self._views_are(optimizer, self.host):
            return
        device = self._device(optimizer)
        if device is not None:
            self._point(optimizer, self.host.to(device))


@contextlib.contextmanager
def moments_on_host():
    """While active (and SOUP_SOLUTION_ADAM_HOST=1), AdamW's moments live on the host between
    steps; after it, they are put back on the GPU."""
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
        moments.restore()
        torch.cuda.synchronize()
