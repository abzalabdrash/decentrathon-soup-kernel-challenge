"""The MLP's saved gate/up out of the GPU's peak: asynchronous offload to pinned host memory, or
recomputation in the backward. The rules allow both ("recomputation or offloading in your own code
(its time counts)"); Soup's own switches (activation_offloading, gradient_checkpointing) stay as the
task sets them.

With mlp16.py every layer keeps two fp16 tensors for the backward, gate and up (N x 8960, 14.8 MB
each at the 826 tokens of an average step after unpad.py), about 0.83 GB over 28 layers; with
fused_proj.py it is one, (gate|up) as (17920, N padded to 8). They are all alive at the end of the
forward, where the fused head adds its chunk of logits: the step's peak.

SOUP_SOLUTION_MLP_MEMORY picks one (patch_mlp_memory):

- recompute: with fused_proj.py's MLP, its autograd function saves only its half input x16 (N x
  1536, 2.25 MB at 767 tokens instead of 26 MB of (gate|up)) and its backward runs the (gate|up)
  GEMM again on the same operands, with the weights it rebuilds for the input gradient anyway (one
  GEMM more per layer, no dequantization more, bit-identical (gate|up)). Nothing crosses PCIe. On the
  T4 (main 4744723, diag runs of one commit) it costs ~50 ms of GPU per step (steps 3+: 24.86 s
  against 22.52 s with the offload) and does not lower the peak (1979 against 2003 MiB): the
  offload keeps one or two (gate|up) on the GPU, recompute keeps 28 x16. Worth it only when PCIe is
  saturated. Without the fused MLP: Qwen2MLP.forward saves only its fp16 input and recomputes gate
  and up (two 4-bit GEMMs) in the backward, same ops in the same order as mlp16.py.
- offload (default): saved-tensor hooks around the whole SFTTrainerWrapper.train. A saved half-precision CUDA
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

Residual offload (SOUP_SOLUTION_OFFLOAD_RESIDUAL=1, night task N1b): the decoder layers' fp32
inputs, which the input norms keep for their backward (6 KB per token per layer, 228 MiB over the
layers of a 1,440-token step once the attention recompute has removed the rest), go to host memory
too. rmsnorm.py marks the one tensor to take (`mark_residual`) before its save_for_backward, so no
other fp32 tensor is ever touched; each layer then saves two tensors (its input, then its gate|up),
both windows grow by one, and the input stays on the GPU after its first unpack until its last user
(the post-attention norm, the x16 rebuild and the input norm read it in the same layer's backward).

Knobs for experiments (environment): SOUP_SOLUTION_OFFLOAD_MIN_NUMEL, SOUP_SOLUTION_OFFLOAD_INFLIGHT,
SOUP_SOLUTION_OFFLOAD_PREFETCH, SOUP_SOLUTION_OFFLOAD_PREPIN_MB (see prepin). bench_offload.py
measures time and peak of the three modes and the pinning speed of the machine.
"""

import atexit
import collections
import contextlib
import os
import threading
import weakref

import torch

from .swiglu import swiglu

MODES = ("offload", "recompute", "none")
DEFAULT_MODE = "offload"

MIN_NUMEL = 4_000_000  # gate/up from 447 tokens on, (gate|up) from 224; nothing else saved is this big
MAX_NUMEL = 64_000_000  # (gate|up) up to 3571 tokens; the 151936 x 1536 embedding is far above
SLAB_BYTES = 256 << 20  # holds a MAX_NUMEL half tensor
# pinned ahead (patch_soup_setup_prepin) once Soup has built a real CUDA model; a larger step pins
# more on demand. On the T4 a slab takes 130 ms of cudaHostAlloc, which holds the forward up when the
# stack grows during training (the "stalls in torch.empty" of the d9f60bb trace). With the layer
# inputs offloaded too a step stacks ~42 KB per token per layer: 7 slabs at the public data's 1,427
# tokens (one pinned on demand with 6 ahead), 10 at the ~1,970-token steps of jury-style rows (four
# on demand: soup-night-ejury2 / fjury logs). 10 slabs cover steps up to ~2,100 tokens.
PREPIN_MB = 2560
ALIGN = 4096
HOST_LIMIT_BYTES = 6 << 30  # pinned memory cap; past it tensors stay on the GPU

HALF_DTYPES = (torch.float16, torch.bfloat16)
RESIDUAL_MIN_NUMEL = 256 * 1024  # an armed fp32 residual below this (170 tokens) stays on the GPU

# SOUP_SOLUTION_OFFLOAD_RESIDUAL (see the module doc): `marked` holds the data pointers of fp32
# layer inputs that a custom Function is saving (mark_residual); the pack hook takes the mark. A
# Python Function's saved tensors are packed after its forward returns, so the mark has to outlive
# the save_for_backward call; a mark nobody takes (no graph) only means one fp32 tensor at that
# address may later go to host too, which is still correct
_RESIDUAL = {"on": os.environ.get("SOUP_SOLUTION_OFFLOAD_RESIDUAL", "1").strip().lower()
             not in ("0", "false", "no", ""), "marked": {}}
# SOUP_SOLUTION_OFFLOAD_ATTN=1 (night task N1c): instead of fused_proj's attention recompute, the
# attention's saved q, compact k and v, output and log-sum-exp (unpad.py) and the post-attention
# norm's half branch (rmsnorm.py, split sum) are marked and go to host memory with the rest
ATTN = os.environ.get("SOUP_SOLUTION_OFFLOAD_ATTN", "0").strip().lower() not in ("0", "false", "no", "")
MARKED_MIN_BYTES = 64 << 10  # a marked tensor below this stays on the GPU


def mark_offload(tensor, keep=False):
    """An active Offloader takes `tensor` (about to be saved for the backward by a custom Function)
    to host memory whatever its dtype and size; `keep`: it stays on the GPU after its first unpack
    (several readers in the backward)."""
    marked = _RESIDUAL["marked"]
    if len(marked) > 256:  # marks never taken (no graph): forget them
        marked.clear()
    marked[tensor.data_ptr()] = keep


def mark_residual(tensor):
    """mark_offload for an fp32 layer input (SOUP_SOLUTION_OFFLOAD_RESIDUAL=1), kept after its first
    unpack for its readers in the layer's backward."""
    if _RESIDUAL["on"]:
        mark_offload(tensor, keep=True)

# Slabs pinned ahead of time by prepin(), taken by the stack before it pins one itself; while a
# prepin thread is still working the stack waits for its next slab instead of pinning one more.
_SPARE = []
_SPARE_LOCK = threading.Condition()
_PREPINNING = [0]  # prepin threads still running


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
    while it runs (measured on WSL2), so it only helps while the process does little CUDA work:
    imports, the model load. The thread is joined at exit: a daemon thread still inside
    cudaHostAlloc while Python finalizes can abort the process (a short pytest run)."""
    count = -(-int(megabytes) // (SLAB_BYTES >> 20))

    def work():
        try:
            for _ in range(count):
                try:
                    slab = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
                except (RuntimeError, MemoryError):
                    # page-locked memory can run out before the GPU does: the stack takes the
                    # slabs made so far, then keeps tensors on the GPU (codex/review-1, 611cade)
                    break
                with _SPARE_LOCK:
                    _SPARE.append(slab)
                    _SPARE_LOCK.notify_all()
        finally:
            with _SPARE_LOCK:
                _PREPINNING[0] -= 1
                _SPARE_LOCK.notify_all()

    with _SPARE_LOCK:
        _PREPINNING[0] += 1
    thread = threading.Thread(target=work, name="soup-offload-prepin", daemon=True)
    thread.start()
    atexit.register(thread.join)
    return thread


def release_host_memory():
    """Training is over (exit_release.py): the pinned slabs, spare and used, go back to the driver
    (cudaFreeHost through torch's host allocator, whose cache holds them once unreferenced). An
    offload that runs again pins new slabs on demand. Slabs still holding a live slot stay. Call it
    with the device synchronized. Returns the bytes released."""
    with _SPARE_LOCK:
        if _PREPINNING[0]:
            return 0  # a prepin thread is still pinning: leave everything as it is
        spares = list(_SPARE)
        _SPARE.clear()
    released = sum(slab.numel() for slab in spares)
    del spares
    offloader = _ACTIVE["offloader"]
    if offloader is not None:
        with offloader.lock:
            host = offloader.host
            if host.live() == 0:
                released += host.nbytes
                host.released = (host.nbytes, len(host.slabs))  # for the summary
                host.slabs, host.slots, host.nbytes = [], [], 0
    if released:
        torch._C._host_emptyCache()
    return released


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
        self.released = None  # (bytes, slabs) handed back by release_host_memory()

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
                while not _SPARE and _PREPINNING[0]:
                    _SPARE_LOCK.wait()  # the prepin thread pins this slab anyway: no second one
                spare = _SPARE.pop() if _SPARE else None
            if spare is None:
                try:
                    spare = torch.empty(SLAB_BYTES, dtype=torch.uint8, pin_memory=True)
                except (RuntimeError, MemoryError):
                    # no more page-locked memory: the cap from here on, so the pack hook keeps
                    # this and every later tensor on the GPU without trying to pin again per
                    # tensor (codex/review-1, 611cade)
                    self.limit = self.nbytes
                    return None, None
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
                 "home", "keep", "unpacked", "__weakref__")

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
        from .weight_stream import CurrentStream, SideStream

        # the stream switches and current-stream reads of weight_stream.py (SOUP_SOLUTION_FAST_STREAMS),
        # and one event, recorded again at every use, for the side stream's waits for the current one
        self._side = SideStream(self.stream)
        self._current = CurrentStream(self.device)
        self._order = torch.cuda.Event()
        self.min_numel = _env_int("SOUP_SOLUTION_OFFLOAD_MIN_NUMEL", MIN_NUMEL) \
            if min_numel is None else min_numel
        layer = saved_per_layer() + (1 if _RESIDUAL["on"] else 0) + (6 if ATTN else 0)
        self.max_inflight = _env_int("SOUP_SOLUTION_OFFLOAD_INFLIGHT", layer) \
            if max_inflight is None else max_inflight
        # with EARLY one tensor ahead: the unpack of a layer's (gate|up) brings back its input, the
        # input's unpack (post-attention norm) the previous layer's (gate|up), never two at once
        self.prefetch = _env_int("SOUP_SOLUTION_OFFLOAD_PREFETCH", 1 if EARLY else layer) \
            if prefetch is None else prefetch
        # storages of the model's parameters and buffers: never offloaded, whatever aliases them
        self.weight_storages = {t.untyped_storage().data_ptr() for t in weights}
        self.host = _HostStack(host_limit)
        self.inflight = collections.deque()
        self.early = {}  # data_ptr -> handle of a tensor whose copy offload_early started
        self.reads = []  # events after H2D reads on a current stream, not yet waited for by side
        self.by_index = weakref.WeakValueDictionary()
        self.count = 0
        self.lock = threading.Lock()
        self.stats = collections.Counter()

    def pack(self, tensor):
        if self.early:
            saved = self.early.pop(tensor.data_ptr(), None)
            if (saved is not None and saved.gpu is not None
                    and saved.gpu.data_ptr() == tensor.data_ptr()
                    and saved.shape == tensor.shape and saved.dtype == tensor.dtype):
                return saved  # the copy offload_early started for this very storage
        if _RESIDUAL["marked"]:
            saved = self._pack_marked(tensor)
            if saved is not None:
                return saved
        if (tensor.dtype not in HALF_DTYPES or not self.min_numel <= tensor.numel() <= MAX_NUMEL
                or not tensor.is_cuda or not tensor.is_contiguous()):
            return tensor
        storage = tensor.untyped_storage()
        if (storage.nbytes() != tensor.numel() * tensor.element_size()
                or storage.data_ptr() in self.weight_storages or _is_weight(tensor)):
            return tensor
        with self.lock:
            return self._offload(tensor)

    def _pack_marked(self, tensor):
        """The handle of a marked tensor (mark_offload), the tensor itself when it cannot go, or
        None when it is not marked (the size rule of the MLP's activations decides)."""
        keep = _RESIDUAL["marked"].pop(tensor.data_ptr(), None)
        if keep is None:
            return None
        nbytes = tensor.numel() * tensor.element_size()
        if (not tensor.is_cuda or not tensor.is_contiguous()
                or not MARKED_MIN_BYTES <= nbytes <= SLAB_BYTES):
            return tensor
        storage = tensor.untyped_storage()
        if (storage.nbytes() != nbytes or storage.data_ptr() in self.weight_storages
                or _is_weight(tensor)):
            return tensor
        with self.lock:
            saved = self._offload(tensor)
            if type(saved) is _Saved:
                saved.keep = keep
                self.stats["residuals" if keep else "marked"] += 1
            return saved

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
        home = self._current()
        self._order.record(home)
        side.wait_event(self._order)
        for event in self.reads:
            side.wait_event(event)
        self.reads.clear()
        host = host.view(tensor.dtype).view(tensor.shape)
        with self._side():
            host.copy_(tensor, non_blocking=True)
        saved = _Saved()
        saved.shape, saved.dtype, saved.device = tensor.shape, tensor.dtype, tensor.device
        saved.slot, saved.host, saved.copied = slot, host, side.record_event()
        saved.gpu, saved.ready, saved.home = tensor, None, home
        saved.keep, saved.unpacked = False, False
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
            current = self._current() if saved.device == self.device else torch.cuda.current_stream(saved.device)
            tensor = saved.gpu
            if saved.unpacked:  # a kept residual, unpacked before in this backward
                return tensor
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
            if saved.keep:  # its other readers in this layer's backward get the same tensor
                saved.gpu, saved.unpacked = tensor, True
            else:  # read: out of the prefetch window (a node unpacking several handles in
                # ascending order would otherwise copy the ones it already read back again)
                self.by_index.pop(saved.index, None)
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
                self._order.record(current)
                side.wait_event(self._order)
                waited = True
            with self._side():
                tensor.copy_(saved.host, non_blocking=True)
            saved.gpu, saved.ready, saved.home = tensor, side.record_event(), current
            self.stats["prefetched"] += 1

    def close(self):
        with self.lock:
            self.early.clear()
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
                f"pinned stack {(self.host.released or (self.host.nbytes,))[0] >> 20} MiB in "
                f"{(self.host.released or (0, len(self.host.slabs)))[1]} slabs ({self.host.prepinned} prepinned"
                f"{'; released after the last step' if self.host.released else ''}); kept on GPU over the cap {s['kept']}; "
                f"of them layer inputs {s['residuals']}, other marked {s['marked']}; "
                f"started early {s['early']}")


_ACTIVE = {"offloader": None}  # the Offloader of the running offload_saved_tensors, if any
# SOUP_SOLUTION_CLEAR_CUBLAS_WS=1: at the start of training, after the warm-up (which waits for its
# thread first), the cuBLAS workspaces are freed so the warm-up thread's is not kept for the run
CLEAR_CUBLAS = os.environ.get("SOUP_SOLUTION_CLEAR_CUBLAS_WS", "1").strip().lower() not in (
    "0", "false", "no", "")
# SOUP_SOLUTION_OFFLOAD_DRAIN_LAST=1: the last layer's MLP forward first lets the copies still in
# flight finish (drain_inflight), so the previous layer's (gate|up) is not alive next to its own
DRAIN_LAST = os.environ.get("SOUP_SOLUTION_OFFLOAD_DRAIN_LAST", "0").strip().lower() not in (
    "0", "false", "no", "")


# SOUP_SOLUTION_OFFLOAD_DRAIN_HEAD=1: at the final norm (head_stream.py's prefetch hook, before the
# head's ring and logits buffer are allocated: the run's peak on the T4 after EARLY, soup-night-wdiag)
# the copies in flight finish on the GPU and the last layer's (gate|up) (47 MiB at ~1,400 tokens)
# leaves; its backward fetches it back on demand
DRAIN_HEAD = os.environ.get("SOUP_SOLUTION_OFFLOAD_DRAIN_HEAD", "0").strip().lower() not in (
    "0", "false", "no", "")
# SOUP_SOLUTION_OFFLOAD_EARLY=1: the fused MLP starts the copy of its (gate|up) to the host right after
# the gate|up GEMM (offload_early) instead of when autograd packs it after the MLP's forward (after
# SwiGLU and the down GEMM), and every MLP forward first lets the copies in flight finish
# (drain_inflight, a wait on the GPU): the previous layer's (gate|up), copied while SwiGLU, the down
# GEMM and the next attention block ran, is given back before this layer allocates its own, instead
# of both being alive at the MLP (the forward's peak: 2 x 49 MiB at 1,440 tokens)
# T4 (soup-night-early, ABBA): peak 539 -> 499 MiB; with the backward's prefetch window of one
# tensor (the default with EARLY: the previous layer's (gate|up) starts coming back when this layer's
# MLP backward is done, under its attention backward) 479 MiB; +0.2..0.3 s
EARLY = os.environ.get("SOUP_SOLUTION_OFFLOAD_EARLY", "1").strip().lower() not in (
    "0", "false", "no", "")


def offload_early(tensor):
    """Starts the D2H copy of `tensor` now when the active Offloader would offload it on its pack
    (the MLP's (gate|up), still in the custom Function's forward); the pack of the same tensor right
    after the forward returns this handle."""
    offloader = _ACTIVE["offloader"]
    if offloader is None:
        return
    saved = offloader.pack(tensor)
    if saved is not tensor:
        with offloader.lock:
            if len(offloader.early) > 8:  # never packed (no graph after all): forget them
                offloader.early.clear()
            offloader.early[tensor.data_ptr()] = saved
            offloader.stats["early"] += 1


def drain_inflight():
    """The compute stream waits (on the GPU) for every D2H copy still in flight and their GPU
    tensors are dropped: their blocks go back to the allocator for what the stream does next."""
    offloader = _ACTIVE["offloader"]
    if offloader is None:
        return
    with offloader.lock:
        offloader.early.clear()  # any handle not packed by now is stale
        while offloader.inflight:
            oldest = offloader.inflight.popleft()
            oldest.home.wait_event(oldest.copied)
            oldest.gpu = None
            offloader.stats["drained"] += 1


@contextlib.contextmanager
def offload_saved_tensors(**knobs):
    """While active, large saved activations live in pinned host memory between the forward and
    the backward. Yields the Offloader (stats, summary())."""
    offloader = Offloader(**knobs)
    # a mark is taken by the pack that follows it inside the same forward, so a mark made before this
    # context began is stale (a forward without a graph, another context): left in place, it would
    # send an unrelated fp32 tensor that later gets its address to host memory too (the values stay
    # right, the counts do not: test_offload saw 13 offloaded tensors instead of 12 after other tests)
    _RESIDUAL["marked"].clear()
    previous = _ACTIVE["offloader"]
    _ACTIVE["offloader"] = offloader
    try:
        with torch.autograd.graph.saved_tensors_hooks(offloader.pack, offloader.unpack):
            yield offloader
    finally:
        _ACTIVE["offloader"] = previous
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


# SOUP_SOLUTION_EMPTY_CACHE=step: torch.cuda.empty_cache() at the start of every training step (the
# harness's step callback synchronizes right before), so the free blocks the caching allocator keeps
# from the previous step (reserved - allocated: ~33 MiB at the peak) are unmapped before the step
# maps its own; "0" (default) leaves the cache alone. The allocator's settings are not touched.
EMPTY_CACHE = os.environ.get("SOUP_SOLUTION_EMPTY_CACHE", "0").strip().lower()

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
        if CLEAR_CUBLAS:
            # the warm-up thread's cuBLAS workspace (8 MiB, one per thread and stream) is no
            # longer used; the training threads make theirs again at their next GEMM
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
        from .cuda_limits import lower_stack_limit

        changed = lower_stack_limit()  # SOUP_SOLUTION_STACK_LIMIT (cuda_limits.py)
        if changed is not None:
            print(f"[solution] CUDA stack limit {changed[0]} -> {changed[1]} bytes per thread",
                  flush=True)
        if EMPTY_CACHE == "step":
            from transformers import TrainerCallback

            class _EmptyCache(TrainerCallback):
                # after the harness's callback has synchronized at the step's start: the blocks
                # the allocator keeps free go back, the step maps what it needs again
                def on_step_begin(self, args, state, control, **kwargs):
                    torch.cuda.empty_cache()

            trainer.add_callback(_EmptyCache())
        from .adam_host import moments_on_host

        with offload_saved_tensors(weights=[*model.parameters(), *model.buffers()]) as offloader, \
                moments_on_host() as moments:
            result = original(self, *args, **kwargs)
        print(f"[solution] mlp offload: {offloader.summary()}", flush=True)
        if moments is not None:
            print(f"[solution] AdamW moments on the host between {moments.steps} steps "
                  f"({moments.moved / 2**20:.1f} MiB each way per step)", flush=True)
        return result

    cls.train = train
    cls._soup_mlp_offload = True
    cls._soup_offload_previous_train = original


def patch_soup_setup_prepin(megabytes):
    """prepin(megabytes) as soon as SFTTrainerWrapper._setup_transformers has built a model with CUDA
    parameters: the pinning (~0.8 s for 1.5 GB on the T4) runs during the trainer build and the start
    of train(), CPU work, and is done before step 0. Soup's tests (fake models) and CPU runs pin
    nothing: a pytest process that applies the solution must not hold 1.5 GB of page-locked memory
    (four of them at once ran a 31 GB host out of memory)."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_offload_prepin", False):
        return
    original = cls._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module) and not cls._soup_offload_prepin_started:
            param = next(model.parameters(), None)
            if param is not None and param.is_cuda:
                cls._soup_offload_prepin_started = True
                prepin(megabytes)
        return result

    cls._setup_transformers = _setup_transformers
    cls._soup_offload_prepin = True
    cls._soup_offload_prepin_started = False
    cls._soup_offload_prepin_previous = original


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
    """Apply SOUP_SOLUTION_MLP_MEMORY=offload|recompute|none (default DEFAULT_MODE), or `mode`
    when given. Returns the mode applied. Call it after mlp16's and fused_proj's MLP patches: with
    the fused MLP in place it sets what that MLP saves (fused_proj.set_mlp_recompute); without it,
    recompute replaces Qwen2MLP.forward. With offload, prepin() of SOUP_SOLUTION_OFFLOAD_PREPIN_MB
    (default PREPIN_MB, 0 = off) starts once Soup has built a real CUDA model."""
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
        megabytes = _env_int("SOUP_SOLUTION_OFFLOAD_PREPIN_MB", PREPIN_MB)
        if megabytes > 0:
            patch_soup_setup_prepin(megabytes)
    elif mode == "recompute" and fused is None:
        patch_qwen2_mlp_recompute()
    return mode
