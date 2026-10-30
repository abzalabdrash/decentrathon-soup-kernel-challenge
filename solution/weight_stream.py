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
  (wait for layer i, start i-1 .. i-K+1). The backward of every 4-bit matmul of layer i+1 comes
  before it: their input gradients all flow into that gradient;
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
import functools
import time
import weakref

import torch

ALIGN = 256  # bytes; every tensor of a slot starts at this alignment (the allocator's is 512)

_STREAMERS = weakref.WeakKeyDictionary()  # decoder layer -> its WeightStreamer


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


def _layer_tensors(layer, device):
    """[(owner, field, tensor)] of the big tensors of every 4-bit linear layer inside `layer`:
    (module, 'weight', packed weight), (quant_state, 'absmax', ...), (state2, 'absmax', ...)."""
    import bitsandbytes as bnb
    from bitsandbytes.functional import QuantState

    found = []
    for name, module in layer.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
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
    puts every layer back into tensors of its own on the GPU and removes the hooks."""

    def __init__(self, layers, slots=2):
        layers = list(layers)
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
        self.device = _first_device(layers[0])
        if self.device.type != "cuda":
            raise StreamSkip(f"weights on {self.device}")

        plans, sizes, seen = [], [], set()
        for layer in layers:
            offset, plan = 0, []
            for owner, field, tensor in _layer_tensors(layer, self.device):
                if tensor.data_ptr() in seen:
                    raise StreamSkip("a 4-bit tensor shared between layers")
                seen.add(tensor.data_ptr())
                plan.append((_Segment(owner, field, tensor, offset), tensor))
                offset += -(-tensor.numel() * tensor.element_size() // ALIGN) * ALIGN
            plans.append(plan)
            sizes.append(offset)

        started = time.perf_counter()
        self.layers, self.slots, self.n = layers, slots, len(layers)
        self.layer_bytes = sizes
        self.slot_bytes = max(sizes)
        self._starts = [0]
        for size in sizes:
            self._starts.append(self._starts[-1] + size)
        # everything that can fail comes before the first change to the model
        self._host, self._unpin = pinned_host_buffer(self._starts[-1], self)
        self._slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device)
                       for _ in range(slots)]
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._ready = [torch.cuda.Event() for _ in range(slots)]  # copy into the slot done
        self._free = [torch.cuda.Event() for _ in range(slots)]  # compute queued when a copy was issued
        for j, plan in enumerate(plans):
            host = self._host_layer(j)
            for segment, tensor in plan:
                segment.view(host).copy_(tensor, non_blocking=True)
        torch.cuda.synchronize(self.device)

        self._plans = [[segment for segment, _ in plan] for plan in plans]
        del plans
        for j, plan in enumerate(self._plans):
            slot = self._slots[j % slots]
            for segment in plan:
                segment.assign(segment.view(slot))
        self._owner = [-1] * slots
        self.loads = 0  # layer copies issued
        self.late_loads = 0  # of them, issued when the layer was already due (not prefetched)
        for j in range(slots):
            self._load(j)
        self._copy_stream.synchronize()
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
        return self._starts[-1]

    def _host_layer(self, j):
        return self._host[self._starts[j]:self._starts[j + 1]]

    def _load(self, j):
        """Starts the copy of layer j into its slot on the copy stream, after the compute queued so
        far (which includes every use of the slot's previous layer)."""
        s = j % self.slots
        self._free[s].record(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._free[s])
            self._slots[s][:self.layer_bytes[j]].copy_(self._host_layer(j), non_blocking=True)
            self._ready[s].record(self._copy_stream)
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
        torch.cuda.current_stream(self.device).wait_event(self._ready[s])
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
        self._host = None


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


def stream_model_weights(model, slots=2, log=print):
    """Streams the 4-bit weights of the decoder layers of a real CUDA Qwen2 model in NF4/FP4
    (bitsandbytes). Returns the WeightStreamer, or None when the model is not one (fake modules,
    CPU, not quantized, gradient checkpointing): then nothing is changed."""
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
        streamer = WeightStreamer(layers, slots)
    except StreamSkip as exc:
        log(f"[solution] weight streaming skipped: {exc}")
        return None
    freed = before - torch.cuda.memory_allocated()
    log(f"[solution] 4-bit weights of {streamer.n} layers streamed from pinned host memory: "
        f"{streamer.slots} GPU slots x {streamer.slot_bytes / 2**20:.1f} MiB, "
        f"{freed / 2**20:.0f} MiB freed on the GPU, setup {streamer.setup_seconds:.2f} s")
    atexit.register(_report, weakref.ref(streamer), log)
    return streamer


def _report(ref, log):
    streamer = ref()
    if streamer is not None:
        log(f"[solution] weight streaming: {streamer.loads} layer copies, "
            f"{streamer.late_loads} of them not prefetched")


def patch_weight_stream(slots=2, early=True):
    """Streams the model's 4-bit weights during Soup's setup.

    early=True: right before peft's `prepare_model_for_kbit_training` (when Soup calls it without
    gradient checkpointing), i.e. before it casts the embedding to fp32, so that the setup's peak
    does not hold all 4-bit weights either. In any case once more after
    `SFTTrainerWrapper._setup_transformers` returns (a no-op when the early hook already streamed).
    Fake modules in Soup's regression tests take the original paths."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if not getattr(cls, "_soup_weight_stream", False):
        original_setup = cls._setup_transformers

        def _setup_transformers(self, *args, **kwargs):
            result = original_setup(self, *args, **kwargs)
            model = getattr(self, "model", None)
            if isinstance(model, torch.nn.Module):
                try:
                    stream_model_weights(model, slots)
                except Exception as exc:  # noqa: BLE001 - an optimisation must not break setup
                    print(f"[solution] weight streaming skipped: {exc!r}")
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
            if checkpointing is False and isinstance(model, torch.nn.Module):
                try:
                    stream_model_weights(model, slots)
                except Exception as exc:  # noqa: BLE001
                    print(f"[solution] weight streaming skipped: {exc!r}")
            return original_prepare(model, *args, **kwargs)

        prepare_model_for_kbit_training._soup_weight_stream = True
        prepare_model_for_kbit_training._soup_original = original_prepare
        peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
