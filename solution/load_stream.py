"""The model's big weights go to pinned host memory while transformers loads them (Soup's setup
only), so that the setup never holds them all on the GPU.

Measured on the T4 (T14 diag, 30 Oct): between the load and peft's kbit preparation the GPU held all
28 layers' 4-bit weights (599 MiB) and the fp16 embedding (445 MiB): 1,100 MiB allocated, 1,176 MiB
reserved, a setup peak of 1,146 MiB, which becomes the run's peak once the training peak falls below
it. transformers' `caching_allocator_warmup` also reserves the model's whole size in one allocation
before the first weight is read.

With on-the-fly bitsandbytes quantization transformers 5.16 loads sequentially in the main thread
(`core_model_loading.convert_and_load_state_dict_in_model`): each checkpoint tensor is read to its
device (`spawn_materialize`, run when its converter runs), quantized there for a 4-bit linear layer
(`Bnb4bitQuantize` -> `Params4bit.to` -> `bnb.functional.quantize_4bit`), and set into its module
(`set_param_for_module`), in the natural order of the keys: the embedding, layer 0 .. 27, the final
norm. Here, inside Soup's `SFTTrainerWrapper._setup_transformers` and for a CUDA Qwen2ForCausalLM
loaded with 4-bit on-the-fly quantization on one GPU:

- `caching_allocator_warmup` is skipped: it reserves 1.1 GB for weights that no longer stay;
- the input embedding is read to the host (`get_device` answers "cpu" for its key), copied into
  pinned memory by head_stream.py and set as its 4-byte placeholder Parameter: the GPU never holds it
  (head_stream.load_embedding; peft's cast to fp32 is then avoided as before);
- when the last 4-bit linear layer of a decoder layer has been set, the layer's packed weights and
  scales go to pinned host memory in weight_stream.py's layout (`offload_loaded_layer`), and the
  WeightStreamer built right before the kbit preparation adopts them without copying them again.

The GPU then holds one checkpoint tensor (at most 27.5 MB in fp16) and one layer's 4-bit tensors at
a time. If streaming is declined later, everything comes back to the GPU as it would have been
(weight_stream.restore_loaded_layers, head_stream's restore). Any other load takes the original code.
Switch: SOUP_SOLUTION_LOAD_STREAM=0. With SOUP_SOLUTION_LOAD_PINNED=1 the checkpoint -> GPU copies of
the loads handled here go through pinned buffers filled by a thread (load_pinned.py).

SOUP_SOLUTION_EMBED_ASYNC=1: the embedding's job of the loop (`_materialize_copy(slice, "cpu",
float16)`: `slice[...]`, a view of the safetensors mmap since safetensors 0.8, then `.to(float16)`
into a fresh 467 MB tensor) took 0.59 s of the main thread on the T4, and head_stream.py then copied
the result into its pinned buffer (0.08 s). Here the job returns an fp16 view of that pinned buffer
(pinned ahead by the thread below, as before) and a thread fills it with `copy_` from the same
`slice[...]`: the cast kernel `.to()` runs, into memory that is already mapped, without the GIL
(locally 0.03 s with torch's 4 CPU threads against 0.18 s for `.to()`), while the main thread goes on
with the layers. head_stream.py adopts the buffer without copying. Nothing reads the values before
the fill is joined: when the load returns (or raises), and before head_stream.py's fallback copies
them to the GPU. If the thread fails, the main thread fills the buffer the same way at the join. Taken
only for the embedding of a load handled here (its key read to the CPU, a safetensors slice of the
embedding's shape, a float dtype, no thread pool or sharding) when its pinned buffer is there.

SOUP_SOLUTION_LOAD_CODE_TABLE=1: bitsandbytes' `quantize_4bit` (functional.py:936) asks
`get_4bit_type` for the 16-value NF4 table of every weight it quantizes (196 here), and that builds
it again each time: `torch.tensor(list, device=cuda)`, a copy from pageable memory that makes the
CPU wait until the GPU has finished the weight's own copy and cast, then three small kernels
(functional.py:854-855). On the T4 those two lines were 0.44 s of the main thread in the tensor loop
(soup-s1-diag0 profile), waits and GIL handoffs to the warm-up threads. While a load handled here
runs, the first call per (type, CUDA device, block size) is the original one and keeps a copy of its
result; later calls return a `clone()` of that copy: the same float32 values, a new tensor for each
weight's quant state as before, one device copy queued on the current stream instead of a wait.
Other devices, and anything outside these loads, take the original function.

SOUP_SOLUTION_LOAD_PREAD=1: load_pinned.py's thread reads each tensor's bytes from its file with
`preadv` instead of copying them out of the safetensors mmap (see there); here the files Soup's setup
opens with `safe_open` are noted and, for a load handled here, the slices of its state_dict are
mapped to their byte ranges (`_FileRanges`).
"""

import functools
import inspect
import os
import threading
import time

import torch

_STATE = {"depth": 0, "layers": False, "only": None, "embedding": False,
          "opened": []}  # (path, handle) of the safetensors files opened in Soup's setup (LOAD_PREAD)
_ACTIVE = {"load": None}  # the load in progress, when it is one this module handles
LAST = {"load": None}  # the last load handled (counters, for the log and the tests)


def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


class _Pinner:
    """Pins host buffers in a thread, in the order the load will need them, while the main thread
    reads the checkpoint and quantizes (torch's fill and cudaHostRegister release the GIL): the
    pinning (0.77 s of the setup on the T4 when it was done in line, T15) overlaps the load."""

    def __init__(self, requests, device):
        from .weight_stream import PinnedBuffer

        self._requests = list(requests)  # [(key, nbytes)]
        self._ready = {key: threading.Event() for key, _ in self._requests}
        self._made = {}
        self._stop = False
        self.seconds = 0.0

        def run():
            started = time.perf_counter()
            try:
                torch.cuda.set_device(device)
                for key, nbytes in self._requests:
                    if self._stop:
                        break
                    try:
                        self._made[key] = PinnedBuffer(nbytes)
                    except Exception:  # noqa: BLE001 - the consumer pins it itself
                        self._made[key] = None
                    self._ready[key].set()
            finally:
                for event in self._ready.values():
                    event.set()
                self.seconds = time.perf_counter() - started

        self._thread = threading.Thread(target=run, name="soup-pinner", daemon=True)
        self._thread.start()

    def take(self, key):
        """The buffer pinned for `key` (waits for it), or None."""
        event = self._ready.get(key)
        if event is None:
            return None
        event.wait()
        return self._made.pop(key, None)

    def close(self):
        """Stops the thread; buffers nobody took are unpinned."""
        self._stop = True
        self._thread.join()
        for buffer in self._made.values():
            if buffer is not None:
                buffer.unpin()
        self._made.clear()


class _Load:
    """One `convert_and_load_state_dict_in_model` call this module takes part in."""

    def __init__(self, model, device, locate=None):
        from .weight_stream import find_decoder_layers

        self.model, self.device = model, device
        self.only = _STATE["only"]
        self.embed_key = _embedding_key(model) if _STATE["embedding"] else None
        self.layer_of, self.remaining = {}, {}
        if _STATE["layers"]:
            import bitsandbytes as bnb

            for layer in find_decoder_layers(model):
                ids = set()
                for name, module in layer.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit) and (
                            not self.only or self.only.intersection(name.split("."))):
                        self.layer_of[id(module)] = layer
                        ids.add(id(module))
                if ids:
                    self.remaining[id(layer)] = ids
        self.failed = self.embedding_done = False
        self.layers_moved = self.layer_bytes = self.embedding_bytes = 0
        self.layer_seconds = self.embedding_seconds = 0.0  # spent here (copies to the host, pinning)
        self.locate = locate  # LOAD_PREAD: slice -> its byte range in the file
        self.embedding_pinned = None  # the embedding's pinned buffer once its job took it
        self.embedding_fill = None  # EMBED_ASYNC: (thread, host view, slice, outcome)
        self.pinner = _Pinner(self._requests(), device)
        from .load_pinned import new_copier

        self.copier = new_copier(device, locate=locate)  # load_pinned.py (SOUP_SOLUTION_LOAD_PINNED=1), else None

    def _requests(self):
        """The pinned buffers this load will need, in the order it needs them."""
        from .weight_stream import find_decoder_layers, predicted_layer_bytes

        requests = []
        if self.embed_key is not None:
            weight = self.model.get_input_embeddings().weight  # meta: shape and dtype of the load
            requests.append(("embedding", weight.numel() * weight.element_size()))
        for layer in find_decoder_layers(self.model):
            if id(layer) in self.remaining:
                requests.append((id(layer), predicted_layer_bytes(layer, self.only)))
        return requests

    def embedding_job(self, job, spawn, args, kwargs):
        """SOUP_SOLUTION_EMBED_ASYNC (see the module docstring): the job for the embedding's checkpoint
        slice, or None for any other tensor (or when switched off)."""
        if self.embed_key is None or self.embedding_fill is not None or not _flag("SOUP_SOLUTION_EMBED_ASYNC"):
            return None
        try:
            bound = inspect.signature(spawn).bind(*args, **kwargs).arguments
            source, device, dtype = bound.get("tensor"), bound.get("device"), bound.get("dtype")
            if bound.get("thread_pool") is not None or bound.get("sharding_op") is not None:
                return None
            if torch.is_tensor(source) or not all(hasattr(source, n) for n in ("get_shape", "get_dtype", "__getitem__")):
                return None
            if not isinstance(device, (str, torch.device)) or torch.device(device).type != "cpu":
                return None
            weight = self.model.get_input_embeddings().weight  # meta: the load's shape and dtype
            shape = tuple(int(size) for size in source.get_shape())
            dtype = weight.dtype if dtype is None else dtype
            if (shape != tuple(weight.shape) or dtype not in (torch.float16, torch.bfloat16, torch.float32)
                    or source.get_dtype() not in ("BF16", "F16", "F32")):
                return None
        except Exception:  # noqa: BLE001 - the loop's own job
            return None
        self.embedding_fill = ()  # one embedding per load
        return functools.partial(self._embedding_async, job, source, shape, dtype)

    def _embedding_async(self, job, source, shape, dtype):
        """The loop's call of the embedding's job: a `dtype` view of its pinned buffer, filled by a
        thread from the slice; the loop's own job when the buffer is not there."""
        pinned = self.pinner.take("embedding")
        self.embedding_pinned = pinned
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        if pinned is None or pinned.buffer is None or pinned.nbytes < nbytes:
            return job()
        host = pinned.buffer[:nbytes].view(dtype).view(shape)
        outcome = {"error": None, "seconds": None, "waited": 0.0}

        where = None
        if self.locate is not None:
            try:
                found = self.locate(source)
                src_dtype = _SAFETENSORS_DTYPES.get(found[3]) if found is not None else None
                if (src_dtype is not None and found[3] == source.get_dtype() and tuple(found[4]) == shape
                        and found[2] == torch.Size(shape).numel() * src_dtype.itemsize and len(shape) == 2):
                    where = (found[0], found[1], src_dtype)
            except Exception:  # noqa: BLE001 - from the slice
                where = None

        def fill():
            started = time.perf_counter()
            try:
                if where is not None:
                    _cast_from_file(host, *where)
                else:
                    host.copy_(source[...])
            except BaseException as exc:  # noqa: BLE001 - the join fills it in the main thread
                outcome["error"] = exc
            outcome["seconds"] = time.perf_counter() - started

        thread = threading.Thread(target=fill, name="soup-embedding-fill", daemon=True)
        self.embedding_fill = (thread, host, source, outcome)
        thread.start()
        return host

    def finish_embedding(self):
        """Waits for the embedding's fill (a no-op without one, or once done); fills the buffer in
        this thread if the fill failed. Returns the seconds this call waited."""
        fill = self.embedding_fill
        if not fill:
            return 0.0
        thread, host, source, outcome = fill
        started = time.perf_counter()
        thread.join()
        if outcome["error"] is not None:
            print(f"[solution] embedding fill thread failed ({outcome['error']!r}); filled in the main thread")
            outcome["error"] = None
            host.copy_(source[...])
        waited = time.perf_counter() - started
        outcome["waited"] += waited
        return waited

    def after_set(self, target_name):
        """A parameter was set: when it completes a decoder layer, move the layer to the host."""
        if self.failed or not self.remaining:
            return
        module_path, _, field = target_name.rpartition(".")
        if field != "weight":
            return
        try:
            module = self.model.get_submodule(module_path)
        except AttributeError:
            return
        layer = self.layer_of.get(id(module))
        if layer is None:
            return
        pending = self.remaining.get(id(layer))
        if pending is None:
            return
        pending.discard(id(module))
        if pending:
            return
        del self.remaining[id(layer)]
        from .weight_stream import offload_loaded_layer

        started = time.perf_counter()
        try:
            pinned = self.pinner.take(id(layer))
            self.layer_bytes += offload_loaded_layer(layer, self.device, self.only, pinned)
            self.layers_moved += 1
        except Exception as exc:  # noqa: BLE001 - the rest of the load stays on the GPU
            self.failed = True
            print(f"[solution] load streaming of the layers stopped: {exc!r}")
        self.layer_seconds += time.perf_counter() - started


def _embedding_key(model):
    """The state-dict key of a tied plain embedding of a model with the fused head, else None."""
    from transformers.models.qwen2 import modeling_qwen2

    if not getattr(type(model), "_soup_fused_ce", False):
        return None
    if not getattr(model.config, "tie_word_embeddings", False):
        return None
    embedding = model.get_input_embeddings()
    if type(embedding) is not torch.nn.Embedding or embedding.max_norm is not None:
        return None
    if not isinstance(model, modeling_qwen2.Qwen2ForCausalLM):
        return None
    for name, module in model.named_modules():
        if module is embedding:
            return f"{name}.weight" if name else "weight"
    return None


def _single_cuda_device(values):
    devices = set()
    for value in values:
        try:
            device = torch.device(value if not isinstance(value, int) else f"cuda:{value}")
        except (RuntimeError, TypeError):
            return None
        if device.type != "cuda":
            return None
        devices.add(torch.device("cuda", device.index if device.index is not None else
                                 torch.cuda.current_device()))
    return devices.pop() if len(devices) == 1 else None


def _handled(model, hf_quantizer, device_values):
    """The device to load to when this module handles the load, else None."""
    if _STATE["depth"] <= 0 or not (_STATE["layers"] or _STATE["embedding"]):
        return None
    if not torch.cuda.is_available() or hf_quantizer is None:
        return None
    if type(hf_quantizer).__name__ != "Bnb4BitHfQuantizer" or getattr(hf_quantizer, "pre_quantized", True):
        return None
    from transformers.models.qwen2 import modeling_qwen2

    if not isinstance(model, modeling_qwen2.Qwen2ForCausalLM):
        return None
    return _single_cuda_device(list(device_values))


_SAFETENSORS_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
FILL_CHUNK_BYTES = 32 << 20  # LOAD_PREAD's scratch for the embedding fill


def _cast_from_file(host, fd, offset, src_dtype):
    """The 2-D `host` filled from the file range at `offset` holding the same shape in `src_dtype`:
    read in chunks of rows into one scratch tensor (preadv), each cast into `host` with `copy_`,
    the kernel `.to()` runs (no overlapping copies)."""
    from .load_pinned import _read_into

    rows, cols = host.shape
    row_bytes = cols * src_dtype.itemsize
    step = max(1, FILL_CHUNK_BYTES // row_bytes)
    scratch = torch.empty(min(step, rows) * cols, dtype=src_dtype)
    raw = scratch.view(torch.uint8)
    for first in range(0, rows, step):
        count = min(step, rows - first)
        _read_into(fd, offset + first * row_bytes, raw, count * row_bytes)
        host[first:first + count].copy_(scratch[:count * cols].view(count, cols))


def _code_table(original):
    """bitsandbytes' get_4bit_type for SOUP_SOLUTION_LOAD_CODE_TABLE (see the module docstring)."""
    tables = {}

    def get_4bit_type(typename, device=None, blocksize=64):
        try:
            where = torch.device("cuda" if device is None else device)
            if where.type != "cuda":
                return original(typename, device=device, blocksize=blocksize)
            if where.index is None:
                where = torch.device("cuda", torch.cuda.current_device())
            key = (typename, where, blocksize)
            table = tables.get(key)
            if table is not None:
                return table.clone()
            made = original(typename, device=device, blocksize=blocksize)
            if isinstance(made, torch.Tensor) and made.device == where:
                tables[key] = made.clone()
            return made
        except (TypeError, ValueError, RuntimeError):
            return original(typename, device=device, blocksize=blocksize)

    get_4bit_type._soup_original = original
    return get_4bit_type


class _FileRanges:
    """SOUP_SOLUTION_LOAD_PREAD (load_pinned.py): where each checkpoint slice of a load lies in its
    file. `state_dict` maps the checkpoint's keys to their slices (transformers' loader), `opened`
    the files Soup's setup opened with safetensors' safe_open; each file's header (an 8-byte little
    endian length, then JSON with dtype, shape and data offsets per key) gives the byte range."""

    def __init__(self, state_dict, opened):
        self.key_of = {id(value): key for key, value in state_dict.items()}
        self.path_of = {}
        for path, handle in opened:
            for key in handle.keys():
                self.path_of.setdefault(key, path)
        self.files = {}  # path -> (fd, data start, header)

    def _file(self, path):
        import json
        import struct

        found = self.files.get(path)
        if found is None:
            fd = os.open(path, os.O_RDONLY)
            try:
                size = struct.unpack("<Q", os.pread(fd, 8, 0))[0]
                header = json.loads(os.pread(fd, size, 8))
            except Exception:
                os.close(fd)
                raise
            found = self.files[path] = (fd, 8 + size, header)
        return found

    def __call__(self, source):
        """(fd, offset, nbytes, dtype name, shape) of the slice `source`, or None."""
        key = self.key_of.get(id(source))
        path = self.path_of.get(key) if key is not None else None
        if path is None:
            return None
        fd, start, header = self._file(path)
        info = header.get(key)
        if not isinstance(info, dict):
            return None
        first, last = info["data_offsets"]
        return fd, start + first, last - first, info["dtype"], info["shape"]

    def close(self):
        for fd, _, _ in self.files.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.files.clear()


def _bind(fn, args, kwargs):
    try:
        return inspect.signature(fn).bind(*args, **kwargs).arguments
    except TypeError:
        return None


def _wrap(owner, name, make):
    """owner.name = make(owner.name), unless the function there is one of ours already (a marker,
    not a flag of this module: tests and other patches replace and restore these attributes)."""
    current = getattr(owner, name)
    if getattr(current, "_soup_load_stream", False):
        return
    wrapper = make(current)
    wrapper._soup_load_stream = True
    setattr(owner, name, wrapper)


def _scoped_setup(original):
    @functools.wraps(original)
    def _setup_transformers(self, *args, **kwargs):
        _STATE["depth"] += 1
        try:
            return original(self, *args, **kwargs)
        finally:
            _STATE["depth"] -= 1
            if _STATE["depth"] <= 0:
                _STATE["opened"] = []  # files of loads this module did not handle

    return _setup_transformers


def _warmup(original):
    @functools.wraps(original)
    def caching_allocator_warmup(model, expanded_device_map, hf_quantizer, *args, **kwargs):
        if _handled(model, hf_quantizer, (expanded_device_map or {}).values()) is not None:
            return None  # the weights do not stay on the GPU: nothing to reserve
        return original(model, expanded_device_map, hf_quantizer, *args, **kwargs)

    return caching_allocator_warmup


def _loader(original):
    @functools.wraps(original)
    def convert_and_load_state_dict_in_model(*args, **kwargs):
        bound = _bind(original, args, kwargs) or {}
        model, config = bound.get("model"), bound.get("load_config")
        device_map = getattr(config, "device_map", None) or {}
        device = _handled(model, getattr(config, "hf_quantizer", None), device_map.values()) \
            if model is not None and device_map else None
        if device is None or _ACTIVE["load"] is not None:
            return original(*args, **kwargs)
        ranges = None
        opened, _STATE["opened"] = _STATE["opened"], []
        if _flag("SOUP_SOLUTION_LOAD_PREAD") and isinstance(bound.get("state_dict"), dict) and opened:
            try:
                ranges = _FileRanges(bound["state_dict"], opened)
            except Exception:  # noqa: BLE001 - staged from the slices
                ranges = None
        try:
            load = _Load(model, device, locate=ranges)
        except Exception as exc:  # noqa: BLE001 - load as usual
            print(f"[solution] load streaming skipped: {exc!r}")
            return original(*args, **kwargs)
        _ACTIVE["load"] = load
        started = time.perf_counter()
        functional = table = None
        if _flag("SOUP_SOLUTION_LOAD_CODE_TABLE"):
            try:
                import bitsandbytes.functional as functional

                plain = functional.get_4bit_type
                if getattr(plain, "_soup_original", None) is None:
                    table = _code_table(plain)
                    functional.get_4bit_type = table
            except Exception:  # noqa: BLE001 - bitsandbytes' own tables then
                table = None
        try:
            return original(*args, **kwargs)
        finally:
            if table is not None and functional.get_4bit_type is table:
                functional.get_4bit_type = table._soup_original
            _ACTIVE["load"] = None
            LAST["load"] = load
            load.finish_embedding()  # EMBED_ASYNC: the embedding's values are complete from here on
            if load.copier is not None:
                load.copier.close()
            load.pinner.close()
            if ranges is not None:
                ranges.close()
            if load.layer_bytes + load.embedding_bytes:
                print(f"[solution] load streaming: {load.layers_moved} layers' 4-bit weights "
                      f"({load.layer_bytes / 2**20:.0f} MiB, {load.layer_seconds:.2f} s) and the embedding "
                      f"({load.embedding_bytes / 2**20:.0f} MiB, {load.embedding_seconds:.2f} s) to pinned "
                      f"host memory as they were loaded; buffers pinned by a thread alongside the load in "
                      f"{load.pinner.seconds:.2f} s; tensor loop {time.perf_counter() - started:.2f} s"
                      + (f"; the embedding cast into its pinned buffer by a thread in "
                         f"{load.embedding_fill[3]['seconds']:.2f} s (the load waited "
                         f"{load.embedding_fill[3]['waited']:.2f} s for it)"
                         if load.embedding_fill and load.embedding_fill[3]["seconds"] is not None else ""))

    return convert_and_load_state_dict_in_model


def _device(original):
    @functools.wraps(original)
    def get_device(device_map, param_name, valid_torch_device=False):
        load = _ACTIVE["load"]
        if load is not None and load.embed_key is not None and param_name == load.embed_key:
            return torch.device("cpu") if valid_torch_device else "cpu"
        return original(device_map, param_name, valid_torch_device=valid_torch_device)

    return get_device


def _setter(original):
    @functools.wraps(original)
    def set_param_for_module(model, target_name, param_value, *args, **kwargs):
        load = _ACTIVE["load"]
        if load is None or model is not load.model:
            return original(model, target_name, param_value, *args, **kwargs)
        if target_name == load.embed_key and not load.embedding_done:
            from .head_stream import load_embedding

            load.embedding_done = True
            started = time.perf_counter()
            pinned = load.embedding_pinned if load.embedding_fill is not None else load.pinner.take("embedding")
            param_value, nbytes = load_embedding(model, param_value, load.device, pinned=pinned,
                                                 ready=load.finish_embedding)
            load.embedding_bytes += nbytes
            load.embedding_seconds += time.perf_counter() - started
            return original(model, target_name, param_value, *args, **kwargs)
        result = original(model, target_name, param_value, *args, **kwargs)
        load.after_set(target_name)
        return result

    return set_param_for_module


def _opener(original):
    @functools.wraps(original)
    def safe_open(filename, *args, **kwargs):
        handle = original(filename, *args, **kwargs)
        if _STATE["depth"] > 0:
            try:
                _STATE["opened"].append((os.fspath(filename), handle))
            except TypeError:
                pass
        return handle

    return safe_open


def _spawner(original):
    @functools.wraps(original)
    def spawn_materialize(*args, **kwargs):
        job = original(*args, **kwargs)
        load = _ACTIVE["load"]
        if load is None:
            return job
        job = load.embedding_job(job, original, args, kwargs) or job  # EMBED_ASYNC: the embedding's
        if load.copier is None:
            return job
        return load.copier.add(job, original, args, kwargs)  # load_pinned.py: the job the loop calls

    return spawn_materialize


def patch_loader(layers=None, only=None, embedding=None):
    """Turns on the load-time moves (layers: weight_stream.py; embedding: head_stream.py) and
    installs the loader hooks (once per function); with SOUP_SOLUTION_LOAD_PINNED=1 also the hook
    of load_pinned.py's copies."""
    if layers is not None:
        _STATE["layers"] = bool(layers)
        _STATE["only"] = frozenset(only) if only else None
    if embedding is not None:
        _STATE["embedding"] = bool(embedding)
    from soup_cli.trainer import sft as soup_sft
    from transformers import core_model_loading, modeling_utils

    _wrap(soup_sft.SFTTrainerWrapper, "_setup_transformers", _scoped_setup)
    _wrap(modeling_utils, "caching_allocator_warmup", _warmup)
    _wrap(modeling_utils, "convert_and_load_state_dict_in_model", _loader)
    _wrap(core_model_loading, "get_device", _device)
    _wrap(core_model_loading, "set_param_for_module", _setter)
    from .load_pinned import enabled as load_pinned_enabled

    if _flag("SOUP_SOLUTION_LOAD_PREAD") and hasattr(modeling_utils, "safe_open"):
        _wrap(modeling_utils, "safe_open", _opener)
    if (load_pinned_enabled() or _flag("SOUP_SOLUTION_EMBED_ASYNC")) and hasattr(core_model_loading,
                                                                                 "spawn_materialize"):
        _wrap(core_model_loading, "spawn_materialize", _spawner)
