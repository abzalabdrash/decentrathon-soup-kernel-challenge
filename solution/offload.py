"""The MLP's saved gate/up out of the GPU's peak: asynchronous offload to pinned host memory, or
recomputation in the backward. The rules allow both ("recomputation or offloading in your own code
(its time counts)"); Soup's own switches (activation_offloading, gradient_checkpointing) stay as the
task sets them.

With mlp16.py every layer keeps two fp16 tensors for the backward, gate and up (N x 8960, 14.8 MB
each at the 826 tokens of an average step after unpad.py), about 0.83 GB over 28 layers. They are
all alive at the end of the forward, where the fused head adds its chunk of logits: the step's peak.

SOUP_SOLUTION_MLP_MEMORY picks one (patch_mlp_memory):

- offload: saved-tensor hooks around the whole SFTTrainerWrapper.train. A saved half-precision CUDA
  activation (it has a grad_fn and owns its storage) of at least MIN_NUMEL elements, i.e. gate and
  up, is copied to pinned host memory on a side stream while the forward goes on. Our reference to
  the GPU tensor is dropped once its copy is done; beyond MAX_INFLIGHT copies in flight the forward
  waits for the oldest, so the CPU cannot run many layers ahead of the GPU with every copied tensor
  still counted as allocated. The backward gets each tensor back with an H2D copy, and asking for
  one starts the copies of the PREFETCH tensors saved just before it on the side stream (the
  backward walks the layers in reverse), so the next layer's gate/up arrive during this layer's
  backward. The pinned memory is a stack of slabs: a step pushes in forward order, the backward
  releases in reverse, so the stack grows to the largest step once and is reused after that
  (pinning costs about 1 s per GB here, so it must not be paid per step or per size class).
- recompute: Qwen2MLP.forward saves only its fp16 input (N x 1536) and recomputes gate and up (two
  4-bit GEMMs) in the backward. Same ops in the same order as mlp16.py, so the same numbers.
- none: nothing.

Stream rules of the offload (current = the stream the forward or backward runs on, side = ours):
- D2H: side waits for an event recorded on current after the tensor was produced; the GPU tensor
  stays referenced (inflight) until the copy's event has completed, or gets record_stream(side)
  when the backward takes it before that;
- a pinned slot is rewritten only after every earlier H2D read of it: reads on side are ordered by
  the stream, reads on current leave an event that side waits for before its next D2H;
- prefetch: the GPU tensor is allocated on current, side waits for current before writing it (the
  block may still be in use by earlier work of current), current waits for the copy's event before
  using it, and record_stream(side) covers a prefetched tensor that is dropped unused.

Knobs for experiments (environment): SOUP_SOLUTION_OFFLOAD_MIN_NUMEL, SOUP_SOLUTION_OFFLOAD_INFLIGHT,
SOUP_SOLUTION_OFFLOAD_PREFETCH, SOUP_SOLUTION_OFFLOAD_PREPIN_MB (see prepin). bench_offload.py
measures time and peak of the three modes and the pinning speed of the machine.
"""

import collections
import contextlib
import os
import threading
import weakref

import torch

from .swiglu import swiglu

MODES = ("offload", "recompute", "none")
DEFAULT_MODE = "offload"

MIN_NUMEL = 4_000_000  # gate/up from 447 tokens on; nothing else saved per layer is this large
MAX_INFLIGHT = 2  # copies the forward may run ahead of: one layer's gate and up
PREFETCH = 2  # tensors fetched ahead of the backward: the next layer's gate and up
SLAB_BYTES = 256 << 20
ALIGN = 4096
HOST_LIMIT_BYTES = 6 << 30  # pinned memory cap; past it tensors stay on the GPU

HALF_DTYPES = (torch.float16, torch.bfloat16)

# Slabs pinned ahead of time by prepin(), taken by the stack before it pins one itself.
_SPARE = []
_SPARE_LOCK = threading.Lock()


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def prepin(megabytes):
    """Pin about `megabytes` of host memory in SLAB_BYTES slabs on a daemon thread, so that the
    offload does not pin during the first steps. Pinning holds up other CUDA calls of the process
    while it runs (measured on WSL2), so it only helps while the process does no CUDA work yet:
    Python imports before the model load."""
    count = -(-int(megabytes) // (SLAB_BYTES >> 20))

    def work():
        for _ in range(count):
            slab = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
            with _SPARE_LOCK:
                _SPARE.append(slab)

    thread = threading.Thread(target=work, name="soup-offload-prepin", daemon=True)
    thread.start()
    return thread


# ----------------------------------------------------------------------------------------------
# offload


class _Slot:
    __slots__ = ("slab", "start", "end", "released")

    def __init__(self, slab, start, end):
        self.slab, self.start, self.end, self.released = slab, start, end, False


class _HostStack:
    """Pinned host memory for the offloaded tensors: slots stacked over fixed slabs. A released
    slot leaves the stack once every slot above it is released too."""

    def __init__(self, limit_bytes=HOST_LIMIT_BYTES):
        self.slabs = []
        self.slots = []
        self.limit = limit_bytes
        self.nbytes = 0
        self.prepinned = 0  # slabs taken from prepin()

    def push(self, nbytes):
        """(slot, pinned uint8 view of nbytes), or (None, None) when the cap is reached."""
        slots = self.slots
        while slots and slots[-1].released:
            slots.pop()
        if slots:
            slab, start = slots[-1].slab, -(-slots[-1].end // ALIGN) * ALIGN
        else:
            slab, start = 0, 0
        while slab < len(self.slabs) and start + nbytes > self.slabs[slab].numel():
            slab, start = slab + 1, 0
        if slab == len(self.slabs):
            size = max(SLAB_BYTES, 1 << (nbytes - 1).bit_length())
            if self.nbytes + size > self.limit:
                return None, None
            spare = None
            if size == SLAB_BYTES:
                with _SPARE_LOCK:
                    spare = _SPARE.pop() if _SPARE else None
            if spare is None:
                spare = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            else:
                self.prepinned += 1
            self.slabs.append(spare)
            self.nbytes += size
        slot = _Slot(slab, start, start + nbytes)
        slots.append(slot)
        return slot, self.slabs[slab][start:start + nbytes]

    def live(self):
        return sum(not slot.released for slot in self.slots)


class _Saved:
    """What the pack hook returns for an offloaded tensor. `gpu` holds a GPU copy while the D2H is
    in flight (the original) or after a prefetch (with `ready`, the H2D's event)."""

    __slots__ = ("shape", "dtype", "device", "index", "slot", "host", "copied", "gpu", "ready",
                 "__weakref__")

    def __del__(self):
        slot = getattr(self, "slot", None)
        if slot is not None:
            slot.released = True


class Offloader:
    """The pack/unpack pair of torch.autograd.graph.saved_tensors_hooks (see the module doc)."""

    def __init__(self, device=None, min_numel=None, max_inflight=None, prefetch=None,
                 host_limit=HOST_LIMIT_BYTES):
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        self.stream = torch.cuda.Stream(self.device)
        self.min_numel = _env_int("SOUP_SOLUTION_OFFLOAD_MIN_NUMEL", MIN_NUMEL) \
            if min_numel is None else min_numel
        self.max_inflight = _env_int("SOUP_SOLUTION_OFFLOAD_INFLIGHT", MAX_INFLIGHT) \
            if max_inflight is None else max_inflight
        self.prefetch = _env_int("SOUP_SOLUTION_OFFLOAD_PREFETCH", PREFETCH) \
            if prefetch is None else prefetch
        self.host = _HostStack(host_limit)
        self.inflight = collections.deque()
        self.reads = []  # events after H2D reads on a current stream, not yet waited for by side
        self.by_index = weakref.WeakValueDictionary()
        self.count = 0
        self.lock = threading.Lock()
        self.stats = collections.Counter()

    def pack(self, tensor):
        if (tensor.dtype not in HALF_DTYPES or tensor.numel() < self.min_numel
                or not tensor.is_cuda or tensor.grad_fn is None or not tensor.is_contiguous()
                or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()):
            return tensor
        with self.lock:
            return self._offload(tensor)

    def _offload(self, tensor):
        if tensor.device != self.device:
            return tensor
        inflight = self.inflight
        while inflight and inflight[0].copied.query():
            inflight.popleft().gpu = None
        nbytes = tensor.numel() * tensor.element_size()
        slot, host = self.host.push(nbytes)
        if slot is None:
            self.stats["kept"] += 1
            return tensor
        side = self.stream
        side.wait_event(torch.cuda.current_stream(tensor.device).record_event())
        for event in self.reads:
            side.wait_event(event)
        self.reads.clear()
        host = host.view(tensor.dtype).view(tensor.shape)
        with torch.cuda.stream(side):
            host.copy_(tensor, non_blocking=True)
        saved = _Saved()
        saved.shape, saved.dtype, saved.device = tensor.shape, tensor.dtype, tensor.device
        saved.slot, saved.host, saved.copied = slot, host, side.record_event()
        saved.gpu, saved.ready = tensor, None
        saved.index = self.count
        self.count += 1
        self.by_index[saved.index] = saved
        inflight.append(saved)
        while len(inflight) > self.max_inflight:
            oldest = inflight.popleft()
            oldest.copied.synchronize()
            oldest.gpu = None
            self.stats["waited"] += 1
        self.stats["offloaded"] += 1
        self.stats["bytes"] += nbytes
        return saved

    def unpack(self, saved):
        if type(saved) is not _Saved:
            return saved
        with self.lock:
            current = torch.cuda.current_stream(saved.device)
            tensor = saved.gpu
            if tensor is None:
                current.wait_event(saved.copied)
                tensor = torch.empty(saved.shape, dtype=saved.dtype, device=saved.device)
                tensor.copy_(saved.host, non_blocking=True)
                self.reads.append(current.record_event())
                self.stats["fetched"] += 1
            elif saved.ready is not None:
                current.wait_event(saved.ready)
                tensor.record_stream(current)  # allocated on the stream of an earlier unpack
                saved.gpu = saved.ready = None
                self.stats["prefetch_hits"] += 1
            else:
                tensor.record_stream(self.stream)  # the D2H may still be reading it
                tensor.record_stream(current)  # allocated on the forward's stream
                saved.gpu = None
                try:
                    self.inflight.remove(saved)
                except ValueError:
                    pass
                self.stats["still_on_gpu"] += 1
            self._prefetch_before(saved.index, current)
        return tensor

    def _prefetch_before(self, index, current):
        side = self.stream
        waited = False
        for j in range(index - 1, max(index - 1 - self.prefetch, -1), -1):
            saved = self.by_index.get(j)
            if saved is None or saved.gpu is not None:
                continue
            tensor = torch.empty(saved.shape, dtype=saved.dtype, device=saved.device)
            if not waited:
                side.wait_event(current.record_event())
                waited = True
            with torch.cuda.stream(side):
                tensor.copy_(saved.host, non_blocking=True)
            tensor.record_stream(side)
            saved.gpu, saved.ready = tensor, side.record_event()
            self.stats["prefetched"] += 1

    def close(self):
        with self.lock:
            while self.inflight:
                oldest = self.inflight.popleft()
                oldest.copied.synchronize()
                oldest.gpu = None
        self.stream.synchronize()

    def summary(self):
        s = self.stats
        return (f"{s['offloaded']} tensors, {s['bytes'] / 2**30:.2f} GiB to host; back: "
                f"{s['prefetch_hits']} prefetched, {s['fetched']} on demand, "
                f"{s['still_on_gpu']} still on GPU; forward waited {s['waited']} times; "
                f"pinned stack {self.host.nbytes >> 20} MiB in {len(self.host.slabs)} slabs "
                f"({self.host.prepinned} prepinned); kept on GPU over the cap {s['kept']}")


@contextlib.contextmanager
def offload_saved_tensors(**knobs):
    """While active, large saved activations live in pinned host memory between the forward and
    the backward. Yields the Offloader (stats, summary())."""
    offloader = Offloader(**knobs)
    try:
        with torch.autograd.graph.saved_tensors_hooks(offloader.pack, offloader.unpack):
            yield offloader
    finally:
        offloader.close()


def _real_cuda_trainer(trainer):
    """A real transformers Trainer with its model on a CUDA device (Soup's tests pass fakes)."""
    if not torch.cuda.is_available():
        return False
    try:
        from transformers import Trainer
    except Exception:  # noqa: BLE001 - no transformers, nothing to offload
        return False
    model = getattr(trainer, "model", None)
    if not isinstance(trainer, Trainer) or not isinstance(model, torch.nn.Module):
        return False
    param = next(model.parameters(), None)
    return param is not None and param.is_cuda


def patch_soup_train_offload():
    """SFTTrainerWrapper.train runs inside offload_saved_tensors when its trainer is real and on
    CUDA; anything else (Soup's regression tests with fake trainers, CPU) runs the original."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_mlp_offload", False):
        return
    original = cls.train

    def train(self, *args, **kwargs):
        if not _real_cuda_trainer(getattr(self, "trainer", None)):
            return original(self, *args, **kwargs)
        with offload_saved_tensors() as offloader:
            result = original(self, *args, **kwargs)
        print(f"[solution] mlp offload: {offloader.summary()}", flush=True)
        return result

    cls.train = train
    cls._soup_mlp_offload = True
    cls._soup_offload_previous_train = original


# ----------------------------------------------------------------------------------------------
# recompute


def half_dtype():
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return torch.float16


class GateUpRecompute(torch.autograd.Function):
    """silu(gate_proj(x)) * up_proj(x) for a half x, saving x only: the backward runs gate_proj,
    up_proj and the SwiGLU again under grad and backpropagates through that local graph (which
    also accumulates the gradients of any trainable parameter of the two projections)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x_half, mlp):
        half = x_half.dtype
        hidden = swiglu(mlp.gate_proj(x_half).to(half), mlp.up_proj(x_half).to(half))
        ctx.mlp = mlp
        ctx.save_for_backward(x_half)
        return hidden

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_hidden):
        (x_half,) = ctx.saved_tensors
        mlp = ctx.mlp
        with torch.enable_grad():
            x = x_half.detach().requires_grad_(True)
            half = x.dtype
            hidden = swiglu(mlp.gate_proj(x).to(half), mlp.up_proj(x).to(half))
        torch.autograd.backward(hidden, grad_hidden)
        return x.grad, None


def mlp_forward_recompute(mlp, x):
    """mlp16.py's forward (half input cast once, SwiGLU in half, the down projection's output back
    to the input dtype) with gate/up recomputed in the backward instead of saved."""
    half = x.dtype if x.dtype != torch.float32 else half_dtype()
    x_half = x if x.dtype == half else x.to(half)
    if torch.is_grad_enabled() and x_half.requires_grad:
        hidden = GateUpRecompute.apply(x_half, mlp)
    else:
        hidden = swiglu(mlp.gate_proj(x_half).to(half), mlp.up_proj(x_half).to(half))
    out = mlp.down_proj(hidden)
    return out if out.dtype == x.dtype else out.to(x.dtype)


def _recompute_applies(mlp, x):
    act = getattr(getattr(mlp, "config", None), "hidden_act", "silu")
    return (act == "silu" and torch.is_tensor(x) and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and all(isinstance(getattr(mlp, name, None), torch.nn.Module)
                    for name in ("gate_proj", "up_proj", "down_proj")))


def patch_qwen2_mlp_recompute():
    """Qwen2MLP.forward through mlp_forward_recompute; replaces mlp16's patch when applied after it
    (the previous forward stays the fallback for fake modules and non-CUDA inputs)."""
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_soup_mlp_recompute", False):
        return
    original = cls.forward

    def forward(self, x):
        if not _recompute_applies(self, x):
            return original(self, x)
        return mlp_forward_recompute(self, x)

    if getattr(cls, "_soup_original_forward", None) is None:
        cls._soup_original_forward = original
    cls.forward = forward
    cls._soup_mlp_recompute = True
    cls._soup_recompute_previous = original


# ----------------------------------------------------------------------------------------------


def patch_mlp_memory(mode=None):
    """Apply SOUP_SOLUTION_MLP_MEMORY=offload|recompute|none (default DEFAULT_MODE), or `mode`
    when given. Returns the mode applied. recompute replaces Qwen2MLP.forward: call it after
    mlp16's patch, not before. With offload, SOUP_SOLUTION_OFFLOAD_PREPIN_MB > 0 starts prepin()
    right away (off by default)."""
    if mode is None:
        mode = os.environ.get("SOUP_SOLUTION_MLP_MEMORY", "") or DEFAULT_MODE
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"SOUP_SOLUTION_MLP_MEMORY must be one of {MODES}, got {mode!r}")
    if mode == "offload":
        patch_soup_train_offload()
        megabytes = _env_int("SOUP_SOLUTION_OFFLOAD_PREPIN_MB", 0)
        if megabytes > 0 and torch.cuda.is_available():
            prepin(megabytes)
    elif mode == "recompute":
        patch_qwen2_mlp_recompute()
    return mode
