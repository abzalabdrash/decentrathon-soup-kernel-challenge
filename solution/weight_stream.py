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

The type ring (SOUP_SOLUTION_WEIGHT_RING=type; the default `layer` is the ring above). A ring of two
whole layers keeps 2 x 24.1 MB on the GPU in every phase of the step, though each MLP matrix of a
layer is read once per pass. Here the attention projections (q, k, v, o: 2.84 MB per layer) keep the
ring above with its per-layer hooks (K slots by layer: the attention recompute reads them again in
the backward), and each MLP matrix type (gate, up, down: 7.1 MB each) has ONE slot of its own, which
every layer's matrix of that type points into for good: 2 x 2.84 + 3 x 7.1 = 27 MB instead of 48.3.
fused_proj.py's MLP calls `acquire(module, step)` before it dequantizes one of them (the compute
stream waits for that matrix's copy; a slot holding another layer's matrix is loaded on the spot)
and `release(module, step)` once the dequantization is queued (the GEMMs read the shared half
buffer, not the slot): the copy of the same matrix of the next layer in the pass (i+1 forward, i-1
backward) starts on the copy stream behind that point. Each of gate, up and down is dequantized once
per pass (forward: gate, up, down; backward: down, gate, up; the recompute mode holds up across its
two reads), so one slot per type costs no copy more than the layer ring. An MLP that runs without
acquire() (its forward is not fused_proj's, or the fused path declines the call) gets its three
matrices loaded before it runs, and again when its layer's backward starts (the hook above). The
matrices' copies and waits are timed only with SOUP_SOLUTION_WEIGHT_RING_TIMING=1 (six more pairs of
CUDA events per layer and pass cost CPU time); the attention group's are, as in the layer ring.
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
# an MLP matrix's Linear4bit -> (its type-ring WeightStreamer, group, layer index)
_MATRICES = weakref.WeakKeyDictionary()

RINGS = ("layer", "type")
# the type ring's groups: the attention projections (every streamed 4-bit layer that is not an MLP
# matrix), then one group per MLP matrix type
_TYPE_GROUPS = ("attn", "gate_proj", "up_proj", "down_proj")


def ring_from_env():
    """SOUP_SOLUTION_WEIGHT_RING: `type` (the default since the night integration: the attention
    projections in a ring of layers, one slot per MLP matrix type; T4, soup-night-wring: nvidia-smi
    421 -> 401 MiB, time unchanged) or `layer` (the ring of whole layers); anything else is
    `layer`."""
    value = os.environ.get("SOUP_SOLUTION_WEIGHT_RING", "type").strip().lower()
    return value if value in RINGS else "layer"


def _env_on(name, default):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


# SOUP_SOLUTION_WEIGHT_RING_TIMING=1: the type ring also times its MLP matrices' copies and waits
# (CUDA events, as for the attention group; 6 matrix spans per layer and pass cost CPU time, so off).
# SOUP_SOLUTION_STREAM_TIMING=1: the attention group's (the layer ring: every layer's) copies and
# waits are timed as well, and so are head_stream.py's chunks; off by default since the CPU-side
# task C1 (two pairs of events per layer and pass and per head chunk, read back with queries: CPU
# time on every step for numbers only the logs show)
# SOUP_SOLUTION_RING_EARLY_RELEASE=1: the int8 MLP backward (int8_bwd.mlp_backward) releases each matrix's
# slot right after its decode, not after the whole backward, so the next layer's copy starts earlier
EARLY_RELEASE = _env_on("SOUP_SOLUTION_RING_EARLY_RELEASE", "1")
# SOUP_SOLUTION_RING_SLOTS_GATE / _UP / _DOWN (1 or 2): slots of that MLP matrix type in the type ring.
# With two, layer j's matrix sits in slot j % 2 and the release of layer i loads layer i + 2*step
# into the slot it frees: the copy starts a whole layer earlier, and the forward ends with the
# last two layers' matrices resident for the backward (+6.8 MiB per slot on Qwen2.5-1.5B)
MLP_SLOTS = {name: max(1, min(2, int(os.environ.get(f"SOUP_SOLUTION_RING_SLOTS_{name.upper()}", "1") or 1)))
             for name in ("gate", "up", "down")}
_RING = {"timing": _env_on("SOUP_SOLUTION_WEIGHT_RING_TIMING", "0"),
         "stream_timing": _env_on("SOUP_SOLUTION_STREAM_TIMING", "0")}

# SOUP_SOLUTION_FAST_STREAMS=0: the copies select their side stream with torch.cuda.stream() and read
# the current stream with torch.cuda.current_stream(), as before. On (the default) the side stream is
# selected by torch's own C setter (what torch.cuda.stream() ends in, after about fifteen Python calls:
# two current-stream objects, device compares) and the current stream is read as its id triple, its
# Stream object reused while the triple stays the same. Laptop: 5.3 -> 0.3 us per stream switch,
# 1.8 -> 0.06 us per current stream; the same CUDA calls in the same order
_STREAMS = {"fast": _env_on("SOUP_SOLUTION_FAST_STREAMS", "1")}
_current_ids = torch._C._cuda_getCurrentStream
_select_stream = torch._C._cuda_setStream
_current_device = torch._C._cuda_getDevice


def _device_index(device):
    device = torch.device(device) if not isinstance(device, torch.device) else device
    return device.index if device.index is not None else torch.cuda.current_device()


class CurrentStream:
    """torch.cuda.current_stream(device) for one device (of the calling thread, as torch's); the
    Stream object is reused while the device's current stream stays the same."""

    __slots__ = ("index", "cached")

    def __init__(self, device):
        self.index = _device_index(device)
        self.cached = (None, None)  # (id triple, Stream), swapped as one

    def __call__(self):
        if not _STREAMS["fast"]:
            return torch.cuda.current_stream(self.index)
        ids = _current_ids(self.index)
        cached = self.cached
        if cached[0] != ids:
            cached = (ids, torch.cuda.Stream(stream_id=ids[0], device_index=ids[1], device_type=ids[2]))
            self.cached = cached
        return cached[1]


class _Selected:
    __slots__ = ("stream", "ids", "prev", "ctx")

    def __init__(self, stream, ids):
        self.stream, self.ids, self.prev, self.ctx = stream, ids, None, None

    def __enter__(self):
        if _STREAMS["fast"] and _current_device() == self.ids[1]:
            self.prev = _current_ids(self.ids[1])
            _select_stream(*self.ids)
        else:
            self.ctx = torch.cuda.stream(self.stream)
            self.ctx.__enter__()
        return self.stream

    def __exit__(self, *exc):
        if self.prev is not None:
            _select_stream(*self.prev)
        else:
            self.ctx.__exit__(*exc)
        return False


class SideStream:
    """`with side():` makes `stream` the current stream until the block ends, as
    `with torch.cuda.stream(stream):` does (a stream of another device than the current one goes
    through torch.cuda.stream())."""

    __slots__ = ("stream", "ids")

    def __init__(self, stream):
        self.stream = stream
        self.ids = (stream.stream_id, stream.device_index, stream.device_type)

    def __call__(self):
        return _Selected(self.stream, self.ids)


class StreamSkip(Exception):
    """The model is not one this module streams; it is left untouched."""


def _unregister(buffer):
    try:
        torch.cuda.cudart().cudaHostUnregister(buffer.data_ptr())
    except Exception:  # noqa: BLE001 - at interpreter exit CUDA may be gone already
        pass


_PINNED = []  # the finalizers of the buffers pinned_host_buffer registered (dead ones are skipped)


def unpin_all():
    """Every buffer pinned_host_buffer registered goes back to pageable memory now (exit_release.py,
    once training is over): its values stay, a copy from it stays correct, only slower; its finalizer
    is spent, so nothing unregisters it again at exit. Returns the bytes unregistered."""
    total = 0
    while _PINNED:
        finalizer = _PINNED.pop()
        info = finalizer.peek()
        if info is not None:
            total += info[2][0].numel()
            finalizer()
    return total


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
        del buffer  # the fallback must not hold two full host copies at once (codex/review-1)
        return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True), None
    finalizer = weakref.finalize(owner, _unregister, buffer)
    _PINNED.append(finalizer)
    return buffer, finalizer


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

    def __init__(self, layer, device, only, pinned=None, sync=True):
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
        if sync:  # else the caller synchronizes once before anything reads the host copies (load_stream.py)
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


def offload_loaded_layer(layer, device, only=None, pinned=None, sync=True):
    """Called by load_stream.py once the loader has quantized every 4-bit linear layer of `layer`:
    its tensors go to pinned host memory now (into `pinned`, a PinnedBuffer made ahead, when it is
    big enough), for the WeightStreamer built before peft's kbit preparation to adopt (no second
    copy). Returns the bytes moved."""
    only = frozenset(only) if only else None
    if layer in _LOADED or layer in _STREAMERS:
        return 0
    host = _HostLayer(layer, device, only, pinned, sync=sync)
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

    def view(self, buffer, base=0):
        """This tensor inside a flat uint8 buffer laid out like the layer (from byte `base` of the
        layer's layout on: a slot of the type ring holds one part of it)."""
        start = self.offset - base
        return buffer[start:start + self.nbytes].view(self.dtype).view(self.shape)

    def assign(self, tensor):
        if self.field == "weight":
            self.owner.weight.data = tensor
        else:
            setattr(self.owner, self.field, tensor)


class _Group:
    """The slots of one part of every layer: `count` slots from `first` in the streamer's flat slot
    list (layer j in slot first + j % count); `ranges[j]` = (start, nbytes) of the part in layer j's
    layout (host buffer); `nbytes` = the slot size. The layer ring has one group, the whole layer."""

    __slots__ = ("name", "count", "ranges", "nbytes", "first")

    def __init__(self, name, count, ranges):
        self.name, self.count, self.ranges = name, count, ranges
        self.nbytes = max(nbytes for _, nbytes in ranges)
        self.first = 0


def _type_layout(layers, plans):
    """How the type ring splits each layer's layout (`plans`: its segments in buffer order):
    (ranges, groups, matrices) with ranges[g][j] = (start, nbytes) of group g of _TYPE_GROUPS, a
    contiguous run of segments, in layer j; groups[j][k] = the group of segment k of layer j;
    matrices[j] = {g: the Linear4bit of MLP matrix group g} for layer j. Raises StreamSkip when a
    layer does not split that way (then the layer ring streams it)."""
    ranges = [[] for _ in _TYPE_GROUPS]
    groups_of, matrices = [], []
    for layer, plan in zip(layers, plans):
        names = {id(module): name for name, module in layer.named_modules()}
        groups, modules, group = [], {}, None
        for segment in plan:
            if segment.field == "weight":  # a module's segments: weight, absmax(, state2 absmax)
                name = names.get(id(segment.owner))
                if name is None:
                    raise StreamSkip("type ring: a streamed weight outside its layer")
                parts = name.split(".")
                group = next((g for g in range(1, len(_TYPE_GROUPS)) if _TYPE_GROUPS[g] in parts), 0)
                if group and modules.setdefault(group, segment.owner) is not segment.owner:
                    raise StreamSkip(f"type ring: two {_TYPE_GROUPS[group]} in a layer")
            elif group is None:
                raise StreamSkip("type ring: a quant state before its weight")
            groups.append(group)
        for g, name in enumerate(_TYPE_GROUPS):
            where = [k for k, owner in enumerate(groups) if owner == g]
            if not where or where != list(range(where[0], where[-1] + 1)):
                raise StreamSkip(f"type ring: {name} missing or not contiguous in a layer's layout")
            first, last = plan[where[0]], plan[where[-1]]
            ranges[g].append((first.offset, last.offset + last.nbytes - first.offset))
        groups_of.append(groups)
        matrices.append(modules)
    return ranges, groups_of, matrices


def _ring_aware(mlp):
    """Whether this MLP module's forward brackets every read of its matrices with acquire() and
    release() (fused_proj.py's patched forward, which calls `mlp_unmanaged` when it falls back)."""
    if mlp is None or "forward" in vars(mlp):
        return False
    return getattr(type(mlp).forward, "_soup_weight_ring", False) is True


def _layer_backward(streamer, index):
    """The gradient reached layer `index`'s output: its own backward comes next."""
    streamer._use(index, -1)


class _LayerExit(torch.autograd.Function):
    """Identity on a layer's output; its backward runs right before the layer's own backward and
    brings the layer's weights into its slot (SOUP_SOLUTION_LAYER_EXIT_HOOK=0)."""

    @staticmethod
    def forward(ctx, hidden, streamer, index):
        ctx.streamer, ctx.index = streamer, index
        return hidden.view_as(hidden)

    @staticmethod
    def backward(ctx, grad):
        _layer_backward(ctx.streamer, ctx.index)
        return grad, None, None


# SOUP_SOLUTION_LAYER_EXIT_HOOK=1 (the default since task C1): the layer's backward marker is a hook
# on its output tensor instead of _LayerExit. autograd calls a tensor's hooks when that tensor's
# gradient is complete, right before the node that made it runs its backward, which is where
# _LayerExit's backward ran (it consumed exactly that gradient and handed it on): the same order of
# work, one Python autograd node less per layer each way
_EXIT = {"hook": _env_on("SOUP_SOLUTION_LAYER_EXIT_HOOK", "1")}


def _exit_hook(streamer, index, grad):
    _layer_backward(streamer, index)


class WeightStreamer:
    """Moves the 4-bit weights of `layers` (called in this order) to pinned host memory and keeps
    `slots` of them on the GPU, loaded ahead of the compute. Construction does the move; `restore()`
    puts every layer back into tensors of its own on the GPU and removes the hooks. `only`: stream
    only the 4-bit layers with one of these name parts (e.g. {'mlp'}), the rest stays on the GPU.

    Timing (CUDA events, read without blocking as they complete, see `stats()`): every copy on the
    copy stream, and the time the compute stream spends in each wait for a layer's copy (an event
    right before the wait and one right after it), forward and backward apart.

    `ring`: "layer" (rings of whole layers) or "type" (see the module doc: the attention projections
    in `slots` slots by layer, one slot per MLP matrix type; a layer that does not split that way or
    `only` given: the layer ring, with the reason in `ring_note`). `timing`: time the copies and
    waits of the attention group (the layer ring: of every layer); None: SOUP_SOLUTION_STREAM_TIMING."""

    def __init__(self, layers, slots=2, only=None, ring="layer", timing=None):
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
        # the type ring's split of the layers (read only: before the first change to the model)
        self.ring, self.ring_note, layout = "layer", None, None
        if ring == "type":
            try:
                if only:
                    raise StreamSkip("type ring: it streams every 4-bit layer, `only` is set")
                layout = _type_layout(layers, [[segment for segment, _ in plan] for plan in plans])
                self.ring = "type"
            except StreamSkip as exc:
                self.ring_note = str(exc)
        elif ring != "layer":
            self.ring_note = f"unknown ring {ring!r}"
        if layout is None:
            groups = [_Group("layer", slots, [(0, size) for size in sizes])]
        else:
            groups = [_Group("attn", slots, layout[0][0])] + [
                _Group(name.split("_")[0], MLP_SLOTS[name.split("_")[0]], layout[0][g])
                for g, name in enumerate(_TYPE_GROUPS) if g]
        first = 0
        for group in groups:
            group.first, first = first, first + group.count

        started = time.perf_counter()
        self.layers, self.slots, self.n, self.only = layers, slots, len(layers), only
        self.layer_bytes = sizes
        self._groups = groups
        self.slot_bytes = groups[0].nbytes  # the layer ring: max(sizes)
        self.ring_bytes = sum(group.count * group.nbytes for group in groups)
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
        # the slots of every group in one flat list (the layer ring: K slots of a whole layer)
        self._slots = [torch.empty(group.nbytes, dtype=torch.uint8, device=self.device)
                       for group in groups for _ in range(group.count)]
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._side = SideStream(self._copy_stream)
        self._compute = CurrentStream(self.device)
        self._ready = [torch.cuda.Event() for _ in self._slots]  # copy into the slot done
        self._free = [torch.cuda.Event() for _ in self._slots]  # compute queued when a copy was issued
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
            for k, segment in enumerate(plan):
                group = groups[0 if layout is None else layout[1][j][k]]
                slot = self._slots[group.first + j % group.count]
                segment.assign(segment.view(slot, group.ranges[j][0]))
        self._owner = [-1] * len(self._slots)
        self.loads = 0  # layer (type ring: group) copies issued
        self.late_loads = 0  # of them, issued when the layer was already due (not prefetched)
        self.group_loads = [0] * len(groups)
        self._spans = collections.deque()  # (kind, begin event, end event, bytes) not yet read
        self._spare_events = []
        self._reset_timing()
        # type ring: group and layer of every MLP matrix (acquire/release), the layers' MLP modules
        # (is their forward fused_proj's?), the layers whose MLP ran without acquire() in a forward
        # with grad (their matrices are loaded again when their backward starts)
        self._matrices = {}
        if layout is not None:
            for j, modules in enumerate(layout[2]):
                for g, module in modules.items():
                    self._matrices[module] = (g, j, groups[g].first + j % groups[g].count)
        self._matrix_groups = tuple(range(1, len(groups)))
        self.matrix_timing = _RING["timing"]
        self.timing = _RING["stream_timing"] if timing is None else bool(timing)
        # every copy's destination (its slot's first bytes) and source (the part of the layer's
        # pinned buffer), made once: views cost CPU time on every copy
        self._copies = [[(self._slots[group.first + j % group.count][:nbytes],
                          self._hosts[j][start:start + nbytes])
                         for j, (start, nbytes) in enumerate(group.ranges)] for group in groups]
        self._mlps = [getattr(layer, "mlp", None) for layer in layers]
        self._held = set()
        for g, group in enumerate(groups):
            for j in range(group.count):
                self._load(j, g, timed=g == 0 or self.matrix_timing)  # the setup's bandwidth
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
        for module, (g, j, _) in self._matrices.items():
            _MATRICES[module] = (self, g, j)
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
        # compute stream waiting for copies: forward, backward (1, -1: a layer's, or the attention
        # group's in the type ring; 2, -2: an MLP matrix's in the type ring)
        self.wait_ms = {1: 0.0, -1: 0.0, 2: 0.0, -2: 0.0}
        self.waits = {1: 0, -1: 0, 2: 0, -2: 0}

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
            "group_loads": {group.name: n for group, n in zip(self._groups, self.group_loads)},
            "copies_timed": self.copies_timed, "copy_ms": self.copy_ms,
            "copy_gbps": self.copy_bytes / self.copy_ms / 1e6 if self.copy_ms > 0 else None,
            "setup_copy_gbps": self.setup_copy_gbps,
            "wait_forward_ms": self.wait_ms[1] + self.wait_ms[2],
            "wait_backward_ms": self.wait_ms[-1] + self.wait_ms[-2],
            "waits_forward": self.waits[1] + self.waits[2],
            "waits_backward": self.waits[-1] + self.waits[-2],
            # of them, the waits for MLP matrices (type ring, timed with SOUP_SOLUTION_WEIGHT_RING_TIMING)
            "timing": self.timing, "matrix_timing": self.matrix_timing,
            "matrix_wait_forward_ms": self.wait_ms[2], "matrix_wait_backward_ms": self.wait_ms[-2],
            "matrix_waits_forward": self.waits[2], "matrix_waits_backward": self.waits[-2],
        }

    def _load(self, j, g=0, timed=None):
        """Starts the copy of layer j's part of group g (the layer ring: the whole layer) into its
        slot on the copy stream, after the compute queued so far (which includes every use of the
        slot's previous layer)."""
        group = self._groups[g]
        s = group.first + j % group.count
        dst, src = self._copies[g][j]
        if timed is None:
            timed = self.matrix_timing if g else self.timing
        copy_stream = self._copy_stream
        self._free[s].record(self._compute())
        with self._side():
            copy_stream.wait_event(self._free[s])
            if timed:
                begin, end = self._event(), self._event()
                begin.record(copy_stream)
            dst.copy_(src, non_blocking=True)
            if timed:
                end.record(copy_stream)
            self._ready[s].record(copy_stream)
        if timed:
            self._spans.append(("copy", begin, end, dst.numel()))
        self._owner[s] = j
        self.loads += 1
        self.group_loads[g] += 1

    def _wait(self, s, kind):
        """The compute stream waits for the last copy into slot s (timed as `kind`: the attention
        group's or a layer's, kind +-1, with SOUP_SOLUTION_STREAM_TIMING=1, an MLP matrix's, kind
        +-2, with SOUP_SOLUTION_WEIGHT_RING_TIMING=1)."""
        compute = self._compute()
        if not (self.matrix_timing if kind in (2, -2) else self.timing):
            compute.wait_event(self._ready[s])
            return
        begin, end = self._event(), self._event()
        begin.record(compute)
        compute.wait_event(self._ready[s])
        end.record(compute)
        self._spans.append((kind, begin, end, 0))
        self._harvest()

    def _use(self, i, step):
        """Layer i is about to run (step +1: forward, -1: backward): the compute stream waits for its
        weights (the type ring: its attention projections), and the next K-1 layers in that
        direction start loading."""
        if not self.active:
            return
        group = self._groups[0]
        s = group.first + i % group.count
        if self._owner[s] != i:
            self._load(i)
            self.late_loads += 1
        self._wait(s, step)
        for d in range(1, group.count):
            j = i + d * step
            if j < 0 or j >= self.n:
                break
            if self._owner[group.first + j % group.count] != j:
                self._load(j)
        if self._matrix_groups:
            if step > 0:
                if not _ring_aware(self._mlps[i]):  # no acquire() will come: its matrices now
                    self._hold_matrices(i, step, torch.is_grad_enabled())
            elif i in self._held:
                self._held.discard(i)
                self._hold_matrices(i, step, False)

    def _hold_matrices(self, i, step, keep):
        """Type ring, an MLP that reads its matrices without acquire(): the three matrices of layer
        i in their slots (loaded on the spot when needed) and the compute stream waiting for them;
        `keep`: again when the layer's backward starts."""
        for g in self._matrix_groups:
            s = self._groups[g].first + i % self._groups[g].count
            if self._owner[s] != i:
                self._load(i, g)
                self.late_loads += 1
            self._wait(s, 2 * step)
        if keep:
            self._held.add(i)

    def acquire(self, module, step):
        """Type ring: `module` (an MLP matrix's Linear4bit) is about to be dequantized in a pass
        (step +1: forward, -1: backward): the compute stream waits for its copy, issued now when its
        slot holds another layer's matrix. Every read of the matrix comes between this call and
        `release`. A no-op for a module this streamer does not hold that way."""
        entry = self._matrices.get(module)
        if entry is None or not self.active:
            return
        g, i, s = entry
        if self._owner[s] != i:
            self._load(i, g)
            self.late_loads += 1
        self._wait(s, 2 * step)

    def release(self, module, step):
        """Type ring: the last read of `module` in this pass is queued (its dequantization; the
        GEMMs read the dequantized copy): the copy of the same matrix of the layer that uses this slot
        next in the pass (i + step, or i + 2*step with two slots of the type) starts behind it."""
        entry = self._matrices.get(module)
        if entry is None or not self.active:
            return
        g, i, s = entry
        j = i + step * self._groups[g].count
        if 0 <= j < self.n and self._owner[s] == i:
            self._load(j, g)

    def _before(self, index, module, args):
        if getattr(module, "gradient_checkpointing", False) and module.training:
            raise RuntimeError("solution.weight_stream: gradient checkpointing was turned on after the "
                               "weights were streamed; the recomputed layers would read other slots")
        self._use(index, 1)

    def _after(self, index, module, args, output):
        if not self.active:
            return None
        if _EXIT["hook"]:
            target = output if torch.is_tensor(output) else (
                output[0] if isinstance(output, tuple) and output and torch.is_tensor(output[0])
                else None)
            if target is not None and target.requires_grad:
                target.register_hook(functools.partial(_exit_hook, self, index))
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
        for module in self._matrices:
            _MATRICES.pop(module, None)
        self._matrices, self._held = {}, set()
        self._slots, self._copies = [], []
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


def mlp_ring(*modules):
    """For fused_proj.py's MLP: the active type-ring WeightStreamer that holds these MLP matrices
    (all of one layer, each in its slot; bracket their reads with its acquire() and release()),
    None when none of them is held that way, False when only some are (do not read them)."""
    if not len(_MATRICES):
        return None
    entries = [_MATRICES.get(module) for module in modules]
    if all(entry is None for entry in entries):
        return None
    if any(entry is None for entry in entries):
        return False
    streamer, _, layer = entries[0]
    if not streamer.active or any(s is not streamer or i != layer for s, _, i in entries):
        return False
    return streamer


def mlp_unmanaged(mlp):
    """For fused_proj.py's MLP when it falls back to a forward that reads its matrices without
    acquire(): with a type ring, the three matrices of the layer in their slots before that forward
    runs, and again when the layer's backward starts. A no-op otherwise."""
    if not len(_MATRICES) or not isinstance(mlp, torch.nn.Module):
        return
    for module in mlp.modules():
        entry = _MATRICES.get(module)
        if entry is not None:
            streamer, _, i = entry
            if streamer.active:
                streamer._hold_matrices(i, 1, torch.is_grad_enabled())
            return


def stream_model_weights(model, slots=2, log=print, only=None, ring=None):
    """Streams the 4-bit weights of the decoder layers of a real CUDA Qwen2 model in NF4/FP4
    (bitsandbytes). Returns the WeightStreamer, or None when the model is not one (fake modules,
    CPU, not quantized, gradient checkpointing): then nothing is changed. A model streamed already
    returns its streamer as it is. `ring`: "layer" or "type" (default: SOUP_SOLUTION_WEIGHT_RING)."""
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
        streamer = WeightStreamer(layers, slots, only, ring=ring_from_env() if ring is None else ring)
    except StreamSkip as exc:
        log(f"[solution] weight streaming skipped: {exc}")
        return None
    freed = before - torch.cuda.memory_allocated()
    gbps = streamer.setup_copy_gbps
    if streamer.ring_note:
        log(f"[solution] weight streaming: {streamer.ring_note}; the ring of whole layers instead")
    if streamer.ring == "type":
        slots_text = ("type ring " + " + ".join(f"{g.count} x {g.name} {g.nbytes / 2**20:.2f}"
                                                 for g in streamer._groups)
                      + f" = {streamer.ring_bytes / 2**20:.1f} MiB on the GPU")
    else:
        slots_text = f"{streamer.slots} GPU slots x {streamer.slot_bytes / 2**20:.1f} MiB"
    log(f"[solution] 4-bit weights of {streamer.n} layers streamed from pinned host memory"
        f"{'' if not only else ' (only ' + ','.join(sorted(only)) + ')'}: {slots_text}, "
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
    if not s["timing"] and not s["matrix_timing"]:
        log(f"[solution] weight streaming ({streamer.ring} ring): {s['loads']} copies"
            + (f" ({', '.join(f'{name} {n}' for name, n in s['group_loads'].items())})"
               if streamer.ring == "type" else "")
            + f", {s['late_loads']} of them not prefetched; copies and waits untimed "
            f"(SOUP_SOLUTION_STREAM_TIMING=1 times them)")
        return
    if streamer.ring == "type":
        loads = ", ".join(f"{name} {n}" for name, n in s["group_loads"].items())
        if s["matrix_timing"]:
            fwd = f" (MLP matrices {s['matrix_wait_forward_ms']:.1f} ms in {s['matrix_waits_forward']})"
            bwd = f" (MLP matrices {s['matrix_wait_backward_ms']:.1f} ms in {s['matrix_waits_backward']})"
            timed = ""
        else:
            fwd = bwd = ""
            timed = " (attention group only: SOUP_SOLUTION_WEIGHT_RING_TIMING=1 times the MLP matrices)"
        log(f"[solution] weight streaming (type ring): {s['loads']} copies ({loads}), "
            f"{s['late_loads']} of them not prefetched; {copy}; compute stream waited for copies "
            f"{s['wait_forward_ms']:.1f} ms in {s['waits_forward']} forward waits{fwd}, "
            f"{s['wait_backward_ms']:.1f} ms in {s['waits_backward']} backward ones{bwd}{timed}")
        return
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
