"""The 4-bit weights of the decoder layers kept in pinned host memory and copied to the GPU one layer
ahead of the compute, so that only a ring of K layers (K = 2 or 3) holds them on the GPU instead of
all 28. Allowed by the rules: "recomputation or offloading in your own code (its time counts)";
Soup's own `stream_layers` switch stays off, nothing of it is used here.

What moves: for every bitsandbytes Linear4bit of a layer (q, k, v, o, gate, up, down) the packed NF4
weight (uint8), the 8-bit absmax codes of its blocks and, with double quant, the fp32 second-level
scales. On Qwen2.5-1.5B that is 46.8 M weights per layer, 24.1 MB with the scales, 675 MB over the
28 layers. The NF4 code table, the absmax code table, the offset and the biases are a few hundred
bytes and stay on the GPU.

The ring: slot s holds layer i when i % K == s, always the same slot for the same layer. The
Params4bit of layer i points (`param.data`) into slot i % K for good, and so do `quant_state.absmax`
and `quant_state.state2.absmax`; only the bytes in the slot change. This matters for the backward:
bitsandbytes' MatMul4Bit (and the patched one in nf4.py) saves `B.view(-1, 1)`, a view of the storage
the weight had during the forward, and reads it again in the backward. With a fixed slot per layer,
that view sees the layer's bytes again whenever the layer is loaded back into its slot.

Order of the copies (host -> GPU on a side stream, from pinned memory, so they overlap the compute):
- forward: before layer i runs, the compute stream waits for the copy of layer i, and the copies
  of layers i+1 .. i+K-1 are started; each copy first waits for everything the compute stream had
  queued when it was issued (the layer that last used that slot, i+d-K <= i-1, is queued by then);
- backward: a tiny identity autograd function on the layer's output fires when the gradient reaches
  the layer, i.e. right before the layer's own backward; it does the same in the other direction
  (wait for layer i, start i-1 .. i-K+1). Every weight rebuild in the backward of layer i+1
  (bitsandbytes' MatMul4Bit, nf4.py, fused_proj.py's qkv / o / gate|up / down) comes before it:
  their input gradients all flow into that gradient. Those rebuilds read `module.weight` and its
  quant state when they run, i.e. the layer's slot;
- the forward ends with the last K layers resident, so the backward starts without waiting, and
  the backward ends with layers 0 .. K-1 resident for the next step.
A layer called out of this order (an eval forward, generation) loads what it needs on the spot: the
result stays correct, it only waits for the copy.

The hooks are per instance (`register_forward_pre_hook` / `register_forward_hook` on each decoder
layer), so the class-level patches of rmsnorm.py and unpad.py's layer loop stay as they are. With
gradient checkpointing (a fixed Soup switch, off on this task) the layers would run again inside the
backward; streaming is not set up then.
"""

import atexit
import collections
import functools
import os
import time
import weakref

import torch

ALIGN = 256  # bytes; every tensor of a slot starts at this alignment (the allocator's is 512)

_STREAMERS = weakref.WeakKeyDictionary()  # decoder layer -> its WeightStreamer
# decoder layer -> _HostLayer: its 4-bit tensors moved to pinned host memory while transformers
# loaded the model (load_stream.py), waiting for a WeightStreamer to adopt them
_LOADED = weakref.WeakKeyDictionary()


class StreamSkip(Exception):
    """The model is not one this module streams; it is left untouched."""


def _unregister(buffer):
    try:
        torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
    except Exception:  # noqa: BLE001 - at interpreter exit CUDA may be gone already
        pass


def pinned_host_buffer(nbytes, owner):
    """`nbytes` of page-locked host memory as a uint8 tensor, unpinned when `owner` is collected (or
    at exit). A plain CPU tensor registered with the driver: torch's pinned allocator rounds the size
    up to a power of two (675 MB -> 1 GB) and pinning costs time per byte (WSL2, RTX 3050 Ti: 645 MB
    in 0.13-0.15 s this way, 1.4 s with `pin_memory=True`). Falls back to `pin_memory=True`.
    Returns (tensor, finalizer or None)."""
    buffer = torch.empty(nbytes, dtype=torch.uint8)
    buffer.fill_(0)  # fault the pages in with torch's parallel fill; the driver pins them faster
    try:
        cudart = torch.cuda.cudart()
        ok = cudart.cudaHostRegister(buffer.data_ptr(), nbytes, 0) == cudart.cudaError.success
    except Exception:  # noqa: BLE001 - a torch build without the binding
        ok = False
    if not ok or not buffer.is_pinned():
        if ok:
            _unregister(buffer)
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), None
    return buffer, weakref.finalize(owner, _unregister, buffer)


class PinnedBuffer:
    """A pinned host buffer that can be made ahead (load_stream.py pins them in a thread while the
    loader reads the checkpoint) and handed to whoever needs it; unpinned by `unpin()` or when the
    holder is collected."""

    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.buffer, self._finalizer = pinned_host_buffer(nbytes, self)

    def unpin(self):
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        self.buffer = None


def predicted_layer_bytes(layer, only=None):
    """The size of a layer's buffer in the streamers' layout, from the shapes of its (still meta)
    4-bit linear layers, as bitsandbytes' quantize_4bit sizes its outputs: the packed weight,
    the absmax of each block (uint8 codes with double quantization) and the fp32 second-level absmax."""
    import bitsandbytes as bnb

    def aligned(nbytes):
        return -(-nbytes // ALIGN) * ALIGN

    total = 0
    for name, module in layer.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit) or (only and not only.intersection(name.split("."))):
            continue
        weight = module.weight
        n = module.in_features * module.out_features
        item = torch.empty(0, dtype=getattr(weight, "quant_storage", torch.uint8)).element_size()
        blocks = -(-n // getattr(weight, "blocksize", 64))
        total += aligned((n + 1) // (2 * item) * item)
        if getattr(weight, "compress_statistics", True):
            total += aligned(blocks) + aligned(-(-blocks // 256) * 4)
        else:
            total += aligned(blocks * 4)
    return total


def _layer_tensors(layer, device, only=None):
    """[(owner, field, tensor)] of the big tensors of every 4-bit linear layer inside `layer`:
    (module, 'weight', packed weight), (quant_state, 'absmax', ...), (state2, 'absmax', ...).
    `only`: a set of module name parts; then only the layers whose dotted name inside `layer` has
    one of them ({'mlp'}: gate_proj, up_proj, down_proj)."""
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState

    found = []
    for name, module in layer.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        if only and not only.intersection(name.split(".")):
            continue
        weight = module.weight
        state = getattr(weight, "quant_state", None)
        if not isinstance(weight, bnb.nn.Params4bit) or not isinstance(state, QuantState):
            raise StreamSkip(f"{name}: no 4-bit weight with a quant state")
        if getattr(state, "packing_format_for_cpu", False):
            raise StreamSkip(f"{name}: CPU packing format")
        fields = [(module, "weight", weight.data), (state, "absmax", state.absmax)]
        if state.nested:
            if not isinstance(state.state2, QuantState):
                raise StreamSkip(f"{name}: nested quant state without state2")
            fields.append((state.state2, "absmax", state.state2.absmax))
        for owner, field, tensor in fields:
            if (not torch.is_tensor(tensor) or tensor.device != device or not tensor.is_contiguous()
                    or tensor.numel() == 0):
                raise StreamSkip(f"{name}.{field}: not a contiguous tensor on {device}")
            found.append((owner, field, tensor))
    if not found:
        raise StreamSkip("a layer without 4-bit linear layers")
    return found


def _plan_layer(layer, device, only, seen=None):
    """[(segment, tensor)] of a layer's 4-bit tensors laid out in one buffer (each at ALIGN), and the
    buffer's size."""
    offset, plan = 0, []
    for owner, field, tensor in _layer_tensors(layer, device, only):
        if seen is not None:
            if tensor.data_ptr() in seen:
                raise StreamSkip("a 4-bit tensor shared between layers")
            seen.add(tensor.data_ptr())
        plan.append((_Segment(owner, field, tensor, offset), tensor))
        offset += -(-tensor.numel() * tensor.element_size() // ALIGN) * ALIGN
    return plan, offset


class _HostLayer:
    """A decoder layer's 4-bit tensors in a pinned host buffer of its own, laid out as a
    WeightStreamer lays out a layer; the modules point at views of it."""

    def __init__(self, layer, device, only, pinned=None):
        plan, nbytes = _plan_layer(layer, device, only)
        self.device, self.only, self.nbytes = device, only, nbytes
        if pinned is not None and pinned.buffer is not None and pinned.nbytes >= nbytes:
            self._pinned = pinned  # pinned ahead (load_stream.py)
            self.buffer, self._unpin = pinned.buffer[:nbytes], pinned.unpin
        else:
            self._pinned = None
            self.buffer, self._unpin = pinned_host_buffer(nbytes, self)
        for segment, tensor in plan:
            segment.view(self.buffer).copy_(tensor, non_blocking=True)
        torch.cuda.synchronize(device)
        self.segments = [segment for segment, _ in plan]
        for segment in self.segments:
            segment.assign(segment.view(self.buffer))

    def restore(self):
        """The layer's tensors back on the GPU, in tensors of their own."""
        for segment in self.segments:
            tensor = torch.empty(segment.shape, dtype=segment.dtype, device=self.device)
            tensor.copy_(segment.view(self.buffer))
            segment.assign(tensor)
        torch.cuda.synchronize(self.device)
        self.unpin()

    def unpin(self):
        if self._unpin is not None:
            self._unpin()
            self._unpin = None
        self.buffer = None


def offload_loaded_layer(layer, device, only=None, pinned=None):
    """Called by load_stream.py once the loader has quantized every 4-bit linear layer of `layer`:
    its tensors go to pinned host memory now (into `pinned`, a PinnedBuffer made ahead, when it is
    big enough), for the WeightStreamer built before peft's kbit preparation to adopt (no second
    copy). Returns the bytes moved."""
    only = frozenset(only) if only else None
    if layer in _LOADED or layer in _STREAMERS:
        return 0
    host = _HostLayer(layer, device, only, pinned)
    _LOADED[layer] = host
    return host.nbytes


def restore_loaded_layers(model):
    """Puts back on the GPU the layers of `model` that the loader moved to the host and that no
    WeightStreamer adopted (streaming declined). Returns how many."""
    if not isinstance(model, torch.nn.Module) or not len(_LOADED):
        return 0
    restored = 0
    for layer in find_decoder_layers(model):
        host = _LOADED.pop(layer, None)
        if host is not None:
            host.restore()
            restored += 1
    return restored


class _Segment:
    __slots__ = ("owner", "field", "shape", "dtype", "offset", "nbytes")

    def __init__(self, owner, field, tensor, offset):
        self.owner, self.field = owner, field
        self.shape, self.dtype = tuple(tensor.shape), tensor.dtype
        self.offset, self.nbytes = offset, tensor.numel() * tensor.element_size()

    def view(self, buffer):
        """This tensor inside a flat uint8 buffer laid out like the layer."""
        return buffer[self.offset:self.offset + self.nbytes].view(self.dtype).view(self.shape)

    def assign(self, tensor):
        if self.field == "weight":
            self.owner.weight.data = tensor
        else:
            setattr(self.owner, self.field, tensor)


class _LayerExit(torch.autograd.Function):
    """Identity on a layer's output; its backward runs right before the layer's own backward and
    brings the layer's weights into its slot."""

    @staticmethod
    def forward(ctx, hidden, streamer, index):
        ctx.streamer, ctx.index = streamer, index
        return hidden.view_as(hidden)

    @staticmethod
    def backward(ctx, grad):
        ctx.streamer._use(ctx.index, -1)
        return grad, None, None


class WeightStreamer:
    """Moves the 4-bit weights of `layers` (called in this order) to pinned host memory and keeps
    `slots` of them on the GPU, loaded ahead of the compute. Construction does the move; `restore()`
    puts every layer back into tensors of its own on the GPU and removes the hooks. `only`: stream
    only the 4-bit layers with one of these name parts (e.g. {'mlp'}), the rest stays on the GPU.

    Timing (CUDA events, read without blocking as they complete, see `stats()`): every copy on the
    copy stream, and the time the compute stream spends in each wait for a layer's copy (an event
    right before the wait and one right after it), forward and backward apart."""

    def __init__(self, layers, slots=2, only=None):
        layers = list(layers)
        only = frozenset(only) if only else None
        if not torch.cuda.is_available():
            raise StreamSkip("no CUDA")
        if not isinstance(slots, int) or slots < 2:
            raise StreamSkip(f"slots={slots!r}: at least 2")
        if len(layers) <= slots:
            raise StreamSkip(f"{len(layers)} layers for {slots} slots: nothing to stream")
        for layer in layers:
            if layer in _STREAMERS:
                raise StreamSkip("a layer is streamed already")
            if getattr(layer, "gradient_checkpointing", False):
                raise StreamSkip("gradient checkpointing would rerun the layers in the backward")
        # layers the loader moved to the host already (load_stream.py) are adopted as they are
        adopted = []
        for layer in layers:
            host = _LOADED.get(layer)
            ok = host is not None and host.only == only and host.buffer is not None
            adopted.append(host if ok else None)
        self.device = next((host.device for host in adopted if host is not None), None)
        if self.device is None:
            self.device = _first_device(layers[0])
        if self.device.type != "cuda":
            raise StreamSkip(f"weights on {self.device}")

        plans, sizes, seen = [], [], set()
        for layer, host in zip(layers, adopted):
            if host is not None:
                plans.append([(segment, None) for segment in host.segments])
                sizes.append(host.nbytes)
            else:
                plan, size = _plan_layer(layer, self.device, only, seen)
                plans.append(plan)
                sizes.append(size)

        started = time.perf_counter()
        self.layers, self.slots, self.n, self.only = layers, slots, len(layers), only
        self.layer_bytes = sizes
        self.slot_bytes = max(sizes)
        self.adopted = sum(host is not None for host in adopted)
        # everything that can fail comes before the first change to the model
        fresh = [j for j, host in enumerate(adopted) if host is None]
        starts, total = {}, 0
        for j in fresh:
            starts[j] = total
            total += sizes[j]
        self._fresh, self._unpin = pinned_host_buffer(total, self) if fresh else (None, None)
        self._hosts = [host.buffer if host is not None else self._fresh[starts[j]:starts[j] + sizes[j]]
                       for j, host in enumerate(adopted)]
        self._adopted_hosts = [host for host in adopted if host is not None]  # they keep the pinning
        self._slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
                       for _ in range(slots)]
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._ready = [torch.cuda.Event() for _ in range(slots)]  # copy into the slot done
        self._free = [torch.cuda.Event() for _ in range(slots)]  # compute queued when a copy was issued
        for j in fresh:
            host = self._host_layer(j)
            for segment, tensor in plans[j]:
                segment.view(host).copy_(tensor, non_blocking=True)
        torch.cuda.synchronize(self.device)
        for layer, host in zip(layers, adopted):
            if host is not None:
                _LOADED.pop(layer, None)

        self._plans = [[segment for segment, _ in plan] for plan in plans]
        del plans
        for j, plan in enumerate(self._plans):
            slot = self._slots[j % slots]
            for segment in plan:
                segment.assign(segment.view(slot))
        self._owner = [-1] * slots
        self.loads = 0  # layer copies issued
        self.late_loads = 0  # of them, issued when the layer was already due (not prefetched)
        self._spans = collections.deque()  # (kind, begin event, end event, bytes) not yet read
        self._spare_events = []
        self._reset_timing()
        for j in range(slots):
            self._load(j)
        self._copy_stream.synchronize()
        self._harvest(block=True)
        # host -> GPU bandwidth of the first K layer copies, measured before anything else runs
        self.setup_copy_gbps = self.copy_bytes / self.copy_ms / 1e6 if self.copy_ms > 0 else None
        self._reset_timing()
        self._handles = []
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_pre_hook(functools.partial(self._before, i)))
            self._handles.append(layer.register_forward_hook(functools.partial(self._after, i)))
            _STREAMERS[layer] = self
        self.active = True
        torch.cuda.empty_cache()
        self.setup_seconds = time.perf_counter() - started

    @property
    def host_bytes(self):
        return sum(self.layer_bytes)

    def _host_layer(self, j):
        return self._hosts[j]

    def _reset_timing(self):
        self.copy_ms, self.copy_bytes, self.copies_timed = 0.0, 0, 0
        self.wait_ms = {1: 0.0, -1: 0.0}  # compute stream waiting for copies: forward, backward
        self.waits = {1: 0, -1: 0}

    def _event(self):
        return self._spare_events.pop() if self._spare_events else torch.cuda.Event(enable_timing=True)

    def _harvest(self, block=False):
        """Adds up the spans whose end event has completed (all of them with block=True)."""
        while self._spans:
            kind, begin, end, nbytes = self._spans[0]
            if block:
                end.synchronize()
            elif not end.query():
                break
            ms = begin.elapsed_time(end)
            if kind == "copy":
                self.copy_ms += ms
                self.copy_bytes += nbytes
                self.copies_timed += 1
            else:
                self.wait_ms[kind] += ms
                self.waits[kind] += 1
            self._spans.popleft()
            self._spare_events += (begin, end)

    def stats(self):
        """Copies and waits so far (synchronizes the device to read the last events)."""
        if self._spans:
            torch.cuda.synchronize(self.device)
            self._harvest(block=True)
        return {
            "loads": self.loads, "late_loads": self.late_loads,
            "copies_timed": self.copies_timed, "copy_ms": self.copy_ms,
            "copy_gbps": self.copy_bytes / self.copy_ms / 1e6 if self.copy_ms > 0 else None,
            "setup_copy_gbps": self.setup_copy_gbps,
            "wait_forward_ms": self.wait_ms[1], "wait_backward_ms": self.wait_ms[-1],
            "waits_forward": self.waits[1], "waits_backward": self.waits[-1],
        }

    def _load(self, j):
        """Starts the copy of layer j into its slot on the copy stream, after the compute queued so
        far (which includes every use of the slot's previous layer)."""
        s = j % self.slots
        self._free[s].record(torch.cuda.current_stream(self.device))
        begin, end = self._event(), self._event()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._free[s])
            begin.record(self._copy_stream)
            self._slots[s][:self.layer_bytes[j]].copy_(self._host_layer(j), non_blocking=True)
            end.record(self._copy_stream)
            self._ready[s].record(self._copy_stream)
        self._spans.append(("copy", begin, end, self.layer_bytes[j]))
        self._owner[s] = j
        self.loads += 1

    def _use(self, i, step):
        """Layer i is about to run (step +1: forward, -1: backward): the compute stream waits for its
        weights, and the next K-1 layers in that direction start loading."""
        if not self.active:
            return
        s = i % self.slots
        if self._owner[s] != i:
            self._load(i)
            self.late_loads += 1
        compute = torch.cuda.current_stream(self.device)
        begin, end = self._event(), self._event()
        begin.record(compute)
        compute.wait_event(self._ready[s])
        end.record(compute)
        self._spans.append((step, begin, end, 0))
        self._harvest()
        for d in range(1, self.slots):
            j = i + d * step
            if j < 0 or j >= self.n:
                break
            if self._owner[j % self.slots] != j:
                self._load(j)

    def _before(self, index, module, args):
        if getattr(module, "gradient_checkpointing", False) and module.training:
            raise RuntimeError("solution.weight_stream: gradient checkpointing was turned on after the "
                               "weights were streamed; the recomputed layers would read other slots")
        self._use(index, 1)

    def _after(self, index, module, args, output):
        if not self.active:
            return None
        if torch.is_tensor(output):
            return _LayerExit.apply(output, self, index) if output.requires_grad else None
        if (isinstance(output, tuple) and output and torch.is_tensor(output[0])
                and output[0].requires_grad):
            return (_LayerExit.apply(output[0], self, index),) + output[1:]
        return None

    def restore(self):
        """Every layer back in GPU tensors of its own, hooks off. Only between steps: an autograd
        graph built while streaming still points into the slots."""
        if not self.active:
            return
        self.active = False
        for handle in self._handles:
            handle.remove()
        self._handles = []
        torch.cuda.synchronize(self.device)
        for j, plan in enumerate(self._plans):
            host = self._host_layer(j)
            for segment in plan:
                tensor = torch.empty(segment.shape, dtype=segment.dtype, device=self.device)
                tensor.copy_(segment.view(host))
                segment.assign(tensor)
        torch.cuda.synchronize(self.device)
        for layer in self.layers:
            _STREAMERS.pop(layer, None)
        self._slots = []
        if self._unpin is not None:
            self._unpin()
        for host in self._adopted_hosts:
            host.unpin()
        self._fresh, self._hosts, self._adopted_hosts = None, [], []


def _first_device(layer):
    import bitsandbytes as bnb

    for module in layer.modules():
        if isinstance(module, bnb.nn.Linear4bit) and torch.is_tensor(module.weight):
            return module.weight.device
    raise StreamSkip("a layer without 4-bit linear layers")


def find_decoder_layers(model):
    """The Qwen2 decoder layers of `model` (a Qwen2 model, its causal LM or a PeftModel around it)
    in the order the decoder calls them."""
    from transformers.models.qwen2 import modeling_qwen2

    return [m for m in model.modules() if isinstance(m, modeling_qwen2.Qwen2DecoderLayer)]


def streamer_of(model):
    """The active WeightStreamer of this model's decoder layers, or None."""
    if not isinstance(model, torch.nn.Module):
        return None
    for layer in find_decoder_layers(model):
        streamer = _STREAMERS.get(layer)
        if streamer is not None and streamer.active:
            return streamer
    return None


def stream_model_weights(model, slots=2, log=print, only=None):
    """Streams the 4-bit weights of the decoder layers of a real CUDA Qwen2 model in NF4/FP4
    (bitsandbytes). Returns the WeightStreamer, or None when the model is not one (fake modules,
    CPU, not quantized, gradient checkpointing): then nothing is changed. A model streamed already
    returns its streamer as it is."""
    if not isinstance(model, torch.nn.Module) or not torch.cuda.is_available():
        return None
    existing = streamer_of(model)
    if existing is not None:
        return existing
    try:
        layers = find_decoder_layers(model)
        if not layers:
            return None
        if getattr(model, "is_gradient_checkpointing", False):
            raise StreamSkip("gradient checkpointing would rerun the layers in the backward")
        before = torch.cuda.memory_allocated()
        streamer = WeightStreamer(layers, slots, only)
    except StreamSkip as exc:
        log(f"[solution] weight streaming skipped: {exc}")
        return None
    freed = before - torch.cuda.memory_allocated()
    gbps = streamer.setup_copy_gbps
    log(f"[solution] 4-bit weights of {streamer.n} layers streamed from pinned host memory"
        f"{'' if not only else ' (only ' + ','.join(sorted(only)) + ')'}: "
        f"{streamer.slots} GPU slots x {streamer.slot_bytes / 2**20:.1f} MiB, "
        f"{freed / 2**20:.0f} MiB freed on the GPU, {streamer.adopted} layers already in host memory "
        f"from the load, setup {streamer.setup_seconds:.2f} s, "
        f"host -> GPU {'?' if gbps is None else f'{gbps:.1f}'} GB/s")
    atexit.register(_report, weakref.ref(streamer), log)
    return streamer


def _report(ref, log):
    streamer = ref()
    if streamer is None:
        return
    try:
        s = streamer.stats()
    except Exception as exc:  # noqa: BLE001 - CUDA may be gone at exit
        log(f"[solution] weight streaming: {streamer.loads} layer copies ({exc!r})")
        return
    copy = (f"{s['copy_ms'] / s['copies_timed']:.2f} ms per copy at {s['copy_gbps']:.1f} GB/s"
            if s["copies_timed"] and s["copy_gbps"] else "no timed copies")
    log(f"[solution] weight streaming: {s['loads']} layer copies, {s['late_loads']} of them not "
        f"prefetched; {copy}; compute stream waited for copies {s['wait_forward_ms']:.1f} ms in "
        f"{s['waits_forward']} forward layer calls, {s['wait_backward_ms']:.1f} ms in "
        f"{s['waits_backward']} backward ones")


def _stream_or_restore(model, slots, only, log=print):
    """Streams the model's 4-bit weights; when that is declined, the layers the loader moved to the
    host come back to the GPU (the model then is what it would have been without us)."""
    streamer = None
    try:
        streamer = stream_model_weights(model, slots, log=log, only=only)
    except Exception as exc:  # noqa: BLE001 - an optimisation must not break setup
        log(f"[solution] weight streaming skipped: {exc!r}")
    if streamer is None:
        restored = restore_loaded_layers(model)
        if restored:
            log(f"[solution] {restored} layers moved to host memory during the load put back on the GPU")
    return streamer


def _only_from_env():
    """SOUP_SOLUTION_WEIGHT_STREAM_ONLY: comma-separated module name parts to stream (e.g. 'mlp',
    or 'gate_proj,up_proj,down_proj'); empty or unset streams all 4-bit layers."""
    value = os.environ.get("SOUP_SOLUTION_WEIGHT_STREAM_ONLY", "")
    parts = {p.strip() for p in value.split(",") if p.strip()}
    return parts or None


def patch_weight_stream(slots=2, early=True, only=None, load=None):
    """Streams the model's 4-bit weights during Soup's setup.

    early=True: right before peft's `prepare_model_for_kbit_training` (when Soup calls it without
    gradient checkpointing), i.e. before it casts the embedding to fp32, so that the setup's peak
    does not hold all 4-bit weights either. In any case once more after
    `SFTTrainerWrapper._setup_transformers` returns (a no-op when the early hook already streamed).
    Fake modules in Soup's regression tests take the original paths. `only` (default from
    SOUP_SOLUTION_WEIGHT_STREAM_ONLY): see WeightStreamer. `load` (default: SOUP_SOLUTION_LOAD_STREAM,
    on): with `early`, every layer's 4-bit tensors go to pinned host memory as soon as transformers
    has quantized them during the load (load_stream.py), and the streamer adopts them."""
    from soup_cli.trainer import sft as soup_sft

    if only is None:
        only = _only_from_env()
    if load is None:
        load = os.environ.get("SOUP_SOLUTION_LOAD_STREAM", "1") not in ("0", "false", "no")
    if early and load:
        from .load_stream import patch_loader

        patch_loader(layers=True, only=only)
    cls = soup_sft.SFTTrainerWrapper
    if not getattr(cls, "_soup_weight_stream", False):
        original_setup = cls._setup_transformers

        def _setup_transformers(self, *args, **kwargs):
            result = original_setup(self, *args, **kwargs)
            model = getattr(self, "model", None)
            if isinstance(model, torch.nn.Module):
                _stream_or_restore(model, slots, only)
            return result

        cls._setup_transformers = _setup_transformers
        cls._soup_weight_stream = True

    if early:
        import peft

        original_prepare = peft.prepare_model_for_kbit_training
        if getattr(original_prepare, "_soup_weight_stream", False):
            return

        @functools.wraps(original_prepare)
        def prepare_model_for_kbit_training(model, *args, **kwargs):
            checkpointing = kwargs.get("use_gradient_checkpointing", args[0] if args else True)
            if isinstance(model, torch.nn.Module):
                if checkpointing is False:
                    _stream_or_restore(model, slots, only)
                elif restore_loaded_layers(model):
                    print("[solution] gradient checkpointing: the layers moved to host memory during "
                          "the load are back on the GPU")
            return original_prepare(model, *args, **kwargs)

        prepare_model_for_kbit_training._soup_weight_stream = True
        prepare_model_for_kbit_training._soup_original = original_prepare
        peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
