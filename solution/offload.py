"""The MLP's saved gate/up out of the GPU's peak: asynchronous offload to pinned host memory, or
recomputation in the backward. The rules allow both ("recomputation or offloading in your own code
(its time counts)"); Soup's own switches (activation_offloading, gradient_checkpointing) stay as the
task sets them.

With mlp16.py every layer keeps two fp16 tensors for the backward, gate and up (N x 8960, 14.8 MB
each at the 826 tokens of an average step after unpad.py), about 0.83 GB over 28 layers; with
fused_proj.py it is one, (gate|up) as (17920, N padded to 8). They are all alive at the end of the
forward, where the fused head adds its chunk of logits: the step's peak.

SOUP_SOLUTION_MLP_MEMORY picks one (patch_mlp_memory):

- recompute (default): with fused_proj.py's MLP, its autograd function saves only its half input
  x16 (N x 1536, 2.25 MB at 767 tokens instead of 26 MB of (gate|up)) and its backward runs the
  (gate|up) GEMM again on the same operands, with the weights it rebuilds for the input gradient
  anyway (one GEMM more per layer, no dequantization more, bit-identical (gate|up)). Nothing crosses
  PCIe and no tensor gets record_stream, so the caching allocator never waits for side-stream
  events (the offload's events stalled the forward 119 ms per step in torch.empty on the T4, and
  its copies took 176 ms of copy-engine time per step from the weight streaming). Without the fused
  MLP: Qwen2MLP.forward saves only its fp16 input and recomputes gate and up (two 4-bit GEMMs) in
  the backward, same ops in the same order as mlp16.py, so the same numbers.
- offload: saved-tensor hooks around the whole SFTTrainerWrapper.train. A saved half-precision CUDA
  tensor of MIN_NUMEL to MAX_NUMEL elements that owns its storage and is not a weight (a parameter,
  a trainable leaf, a view of one, or anything sharing the storage of a parameter or buffer of the
  trained model), i.e. gate and up, is copied to pinned host memory on a side
  stream while the forward goes on. It may have no grad_fn: fused_proj's MLP saves (gate|up) made
  inside its autograd.Function, where grad mode is off. Our reference to the GPU tensor is dropped
  once its copy is done; beyond `max_inflight` copies in flight the oldest is dropped behind a wait
  of its stream for its copy (on the GPU, the CPU goes on), so the copied tensors do not pile up as
  allocated while the CPU runs layers ahead of the GPU. The backward gets each tensor back with an H2D copy, and asking for one starts the
  copies of the `prefetch` tensors saved just before it on the side stream (the backward walks the
  layers in reverse), so the next layer's gate/up arrive during this layer's backward. Both
  windows default to one layer: the tensors one MLP saves (saved_per_layer). The pinned memory is a stack of slabs: a step pushes in forward order, the backward
  releases in reverse, so the stack grows to the largest step once and is reused after that
  (pinning costs about 1 s per GB here, so it must not be paid per step or per size class).
  With the fused MLP it saves (gate|up) for the hooks to take (set_mlp_recompute(False)).
- none: nothing (the fused MLP saves (gate|up) and it stays on the GPU).

Stream rules of the offload (current = the stream the forward or backward runs on, side = ours,
home = the stream a GPU tensor was allocated on: the caching allocator reuses its block only for
later allocations on home). No record_stream and no CPU wait on the training path (one stream for
the forward and the backward): every hazard is closed by making home wait for an event, so the CPU
keeps running ahead of the GPU and the allocator never has stream uses to track:
- D2H: side waits for an event recorded on current after the tensor was produced; the GPU tensor
  stays referenced (inflight) until its copy's event has completed, or, past `max_inflight` copies
  in flight, home waits for that event and the reference is dropped (later work on home, including
  any reuse of the block, comes after the copy); the backward taking the tensor before that makes
  home wait for the copy the same way;
- a pinned slot is rewritten only after every earlier H2D read of it: reads on side are ordered by
  the stream, reads on current leave an event that side waits for before its next D2H;
- prefetch: the GPU tensor is allocated on current (home), side waits for current before writing it
  (the block may still be in use by earlier work of current), current waits for the copy's event
  before using it; a prefetched tensor dropped unused makes home wait for its copy first.
A tensor used on another stream than its home (two backward streams: not this training) falls back
to record_stream.

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
DEFAULT_MODE = "recompute"

MIN_NUMEL = 4_000_000  # gate/up from 447 tokens on, (gate|up) from 224; nothing else saved is this big
MAX_NUMEL = 64_000_000  # (gate|up) up to 3571 tokens; the 151936 x 1536 embedding is far above
SLAB_BYTES = 256 << 20  # holds a MAX_NUMEL half tensor
ALIGN = 4096
HOST_LIMIT_BYTES = 6 << 30  # pinned memory cap; past it tensors stay on the GPU

HALF_DTYPES = (torch.float16, torch.bfloat16)

# Slabs pinned ahead of time by prepin(), taken by the stack before it pins one itself.
_SPARE = []
_SPARE_LOCK = threading.Lock()


def _env_int(name, default):
    value = os.environ.get(name, "").strip()
    return int(value) if value else default


def saved_per_layer():
    """Tensors one Qwen2MLP saves for the backward: 1 with fused_proj.py's MLP ((gate|up) in one
    buffer), else 2 (gate and up, mlp16.py)."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception:  # noqa: BLE001 - no transformers: the plain SwiGLU's two
        return 2
    return 1 if getattr(modeling_qwen2.Qwen2MLP, "_soup_fused_proj", False) else 2


def _is_weight(tensor):
    """A parameter, a trainable leaf or a view of one: saved by reference, offloading frees nothing."""
    base = tensor if tensor._base is None else tensor._base
    return isinstance(base, torch.nn.Parameter) or (base.requires_grad and base.grad_fn is None)


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
        """(slot, pinned uint8 view of nbytes <= SLAB_BYTES), or (None, None) at the cap."""
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
            if self.nbytes + SLAB_BYTES > self.limit:
                return None, None
            with _SPARE_LOCK:
                spare = _SPARE.pop() if _SPARE else None
            if spare is None:
                spare = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
            else:
                self.prepinned += 1
            self.slabs.append(spare)
            self.nbytes += SLAB_BYTES
        slot = _Slot(slab, start, start + nbytes)
        slots.append(slot)
        return slot, self.slabs[slab][start:start + nbytes]

    def live(self):
        return sum(not slot.released for slot in self.slots)


class _Saved:
    """What the pack hook returns for an offloaded tensor. `gpu` holds a GPU copy while the D2H is
    in flight (the original) or after a prefetch (with `ready`, the H2D's event)."""

    __slots__ = ("shape", "dtype", "device", "index", "slot", "host", "copied", "gpu", "ready",
                 "home", "__weakref__")

    def __del__(self):
        slot = getattr(self, "slot", None)
        if slot is not None:
            slot.released = True
        if getattr(self, "gpu", None) is not None and getattr(self, "ready", None) is not None:
            try:  # a prefetched tensor never unpacked: its block is not reused before the copy
                self.home.wait_event(self.ready)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass


class Offloader:
    """The pack/unpack pair of torch.autograd.graph.saved_tensors_hooks (see the module doc)."""

    def __init__(self, device=None, min_numel=None, max_inflight=None, prefetch=None,
                 host_limit=HOST_LIMIT_BYTES, weights=()):
        self.device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
        self.stream = torch.cuda.Stream(self.device)
        self.min_numel = _env_int("SOUP_SOLUTION_OFFLOAD_MIN_NUMEL", MIN_NUMEL) \
            if min_numel is None else min_numel
        layer = saved_per_layer()
        self.max_inflight = _env_int("SOUP_SOLUTION_OFFLOAD_INFLIGHT", layer) \
            if max_inflight is None else max_inflight
        self.prefetch = _env_int("SOUP_SOLUTION_OFFLOAD_PREFETCH", layer) \
            if prefetch is None else prefetch
        # storages of the model's parameters and buffers: never offloaded, whatever aliases them
        self.weight_storages = {t.untyped_storage().data_ptr() for t in weights}
        self.host = _HostStack(host_limit)
        self.inflight = collections.deque()
        self.reads = []  # events after H2D reads on a current stream, not yet waited for by side
        self.by_index = weakref.WeakValueDictionary()
        self.count = 0
        self.lock = threading.Lock()
        self.stats = collections.Counter()

    def pack(self, tensor):
        if (tensor.dtype not in HALF_DTYPES or not self.min_numel <= tensor.numel() <= MAX_NUMEL
                or not tensor.is_cuda or not tensor.is_contiguous()):
            return tensor
        storage = tensor.untyped_storage()
        if (storage.nbytes() != tensor.numel() * tensor.element_size()
                or storage.data_ptr() in self.weight_storages or _is_weight(tensor)):
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
        home = torch.cuda.current_stream(tensor.device)
        side.wait_event(home.record_event())
        for event in self.reads:
            side.wait_event(event)
        self.reads.clear()
        host = host.view(tensor.dtype).view(tensor.shape)
        with torch.cuda.stream(side):
            host.copy_(tensor, non_blocking=True)
        saved = _Saved()
        saved.shape, saved.dtype, saved.device = tensor.shape, tensor.dtype, tensor.device
        saved.slot, saved.host, saved.copied = slot, host, side.record_event()
        saved.gpu, saved.ready, saved.home = tensor, None, home
        saved.index = self.count
        self.count += 1
        self.by_index[saved.index] = saved
        inflight.append(saved)
        while len(inflight) > self.max_inflight:
            oldest = inflight.popleft()
            oldest.home.wait_event(oldest.copied)  # on the GPU: the CPU goes on
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
                if saved.home != current:
                    tensor.record_stream(current)
                saved.gpu = saved.ready = None
                self.stats["prefetch_hits"] += 1
            else:
                saved.home.wait_event(saved.copied)  # the D2H may still be reading it
                if saved.home != current:
                    tensor.record_stream(current)
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
            saved.gpu, saved.ready, saved.home = tensor, side.record_event(), current
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
        return (f"{s['offloaded']} tensors, {s['bytes'] / 2**30:.2f} GiB to host (windows: "
                f"{self.max_inflight} in flight, {self.prefetch} prefetched); back: "
                f"{s['prefetch_hits']} prefetched, {s['fetched']} on demand, "
                f"{s['still_on_gpu']} still on GPU; {s['waited']} dropped behind a stream wait; "
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
        trainer = getattr(self, "trainer", None)
        if not _real_cuda_trainer(trainer):
            return original(self, *args, **kwargs)
        model = trainer.model
        with offload_saved_tensors(weights=[*model.parameters(), *model.buffers()]) as offloader:
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


def _fused_mlp():
    """fused_proj when its Qwen2MLP patch is in place, else None."""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except Exception:  # noqa: BLE001
        return None
    if not getattr(modeling_qwen2.Qwen2MLP, "_soup_fused_proj", False):
        return None
    from . import fused_proj

    return fused_proj


def patch_mlp_memory(mode=None):
    """Apply SOUP_SOLUTION_MLP_MEMORY=recompute|offload|none (default DEFAULT_MODE), or `mode`
    when given. Returns the mode applied. Call it after mlp16's and fused_proj's MLP patches: with
    the fused MLP in place it sets what that MLP saves (fused_proj.set_mlp_recompute); without it,
    recompute replaces Qwen2MLP.forward. With offload, SOUP_SOLUTION_OFFLOAD_PREPIN_MB > 0 starts
    prepin() right away (off by default)."""
    if mode is None:
        mode = os.environ.get("SOUP_SOLUTION_MLP_MEMORY", "") or DEFAULT_MODE
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"SOUP_SOLUTION_MLP_MEMORY must be one of {MODES}, got {mode!r}")
    fused = _fused_mlp()
    if fused is not None:
        fused.set_mlp_recompute(mode == "recompute")
    if mode == "offload":
        patch_soup_train_offload()
        megabytes = _env_int("SOUP_SOLUTION_OFFLOAD_PREPIN_MB", 0)
        if megabytes > 0 and torch.cuda.is_available():
            prepin(megabytes)
    elif mode == "recompute" and fused is None:
        patch_qwen2_mlp_recompute()
    return mode
