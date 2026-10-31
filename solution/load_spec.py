"""The layers' NF4 quantization done ahead of transformers' loop by a thread (SOUP_SOLUTION_LOAD_SPEC=1,
off by default until measured; the organizers allow reading and quantizing the weights in a thread
before from_pretrained when the finished tensors are handed over only on a match, answer 13 of the
second wave). Needs SOUP_SOLUTION_LOAD_EARLY (the checkpoint's bytes in pinned memory).

In Soup's model setup transformers quantizes each of the 196 linear weights in its loop: the job copies
the bf16 checkpoint tensor to the GPU as fp16 (load_pinned.py), bitsandbytes' `Params4bit._quantize`
calls `bnb.functional.quantize_4bit` on it (the NF4 kernel, the per-weight mean of the block maxima,
the 8-bit double quantization of those maxima), and the result goes into the module.

Here, once apply()'s imports are done, a thread takes the layers' linear weights in the loop's order
(natural key order) from load_early.py's pinned chunks and, for each, does what the loop would do on
the GPU: the same raw copy and in-place cast to the run's half dtype as load_pinned.py's GPU cast, then
`bnb.functional.quantize_4bit` with the run's parameters (NF4, blocks of 64, double quantization, uint8
storage), one weight at a time (the double quantization's statistics are per weight). At most WINDOW
bytes of finished weights wait on the GPU; the loop's hand-over frees room.

The hand-over is checked tensor by tensor against what the loop is about to do: the same checkpoint
file, key, shape and dtypes, the module a bitsandbytes Linear4bit whose Params4bit has the same
blocksize, double quantization, quant type and storage, the same device. Then load_pinned.py's job
returns a two-byte stand-in (a stride-0 view of the right shape and dtype) instead of the fp16 tensor,
and `Params4bit._quantize` on that stand-in sets what `_quantize` sets (`data`, `quant_state`, the
module's `quant_state`, `bnb_quantized`) from the thread's `quantize_4bit` result: the same function on
the same values, so the packed weight and the whole quant state are the loop's. Any mismatch, a thread
error or a missing result: the speculation stops and that tensor and every later one take the loop's
own path. When the load returns the thread is stopped and joined before the pinned chunks are released.
"""

import os
import re
import threading
import time

import torch

WINDOW = 96 << 20  # bytes of finished weights waiting on the GPU (~4 layers of Qwen2.5-1.5B)
QUANT = {"blocksize": 64, "compress_statistics": True, "quant_type": "nf4", "quant_storage": torch.uint8}
_LINEAR = re.compile(r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$")
_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
LAST = {"made": 0, "handed": 0, "dropped": None, "seconds": None, "waited": 0.0, "first_wait": None}


def enabled():
    return os.environ.get("SOUP_SOLUTION_LOAD_SPEC", "0").strip().lower() not in ("0", "false", "no", "")


def _natural(key):
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in key.split("."))


def _quantize_4bit(A, code):
    """bitsandbytes 0.50.2's `functional.quantize_4bit(A, blocksize=64, compress_statistics=True,
    quant_type="nf4", quant_storage=torch.uint8)` step by step (functional.py:917-971), with the NF4
    table `code` made once instead of `get_4bit_type` per call: that one builds it with
    `torch.tensor(list)` under the process's default dtype, which transformers switches to the load's
    dtype while it builds the model skeleton on the main thread (`local_torch_dtype`), so a call from this
    thread in that window would make an fp16 table. Every other step is bitsandbytes' own (the CUDA ops
    allocate with explicit dtypes; the dynamic map of `quantize_blockwise` is built in float32)."""
    import bitsandbytes.functional as F

    input_shape = A.shape
    _out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(A, QUANT["blocksize"], QUANT["quant_type"],
                                                                 QUANT["quant_storage"])
    offset = _absmax.mean()
    qabsmax, state2 = F.quantize_blockwise(_absmax - offset, blocksize=256)
    del _absmax
    state = F.QuantState(absmax=qabsmax, shape=input_shape, dtype=A.dtype, blocksize=QUANT["blocksize"],
                         code=code.clone(), quant_type=QUANT["quant_type"], offset=offset, state2=state2)
    return _out, state


class _Spec:
    def __init__(self, files, half, device, code):
        self.half, self.device, self.code = half, device, code
        self.cond = threading.Condition()
        self.results = {}  # (path, key) -> (w4, quant_state, nbytes)
        self.held = 0
        self.stop = False
        self.error = None
        self.done = False
        items = []
        for entry in files:
            for key, (dtype, shape, first, end) in entry.tensors.items():
                if _LINEAR.match(key) and len(shape) == 2 and dtype in _DTYPES and _DTYPES[dtype].itemsize == half.itemsize:
                    items.append((_natural(key), entry.path, key, _DTYPES[dtype], shape, entry.data_start + first, end - first))
        self.items = [item[1:] for item in sorted(items)]
        self.keys = {(item[0], item[1]) for item in self.items}  # (path, key) of every weight it quantizes
        self.thread = threading.Thread(target=self._run, name="soup-load-spec", daemon=True)

    def _run(self):
        from .load_early import view

        started = time.perf_counter()
        try:
            torch.cuda.set_device(self.device)
            for path, key, src_dtype, shape, offset, nbytes in self.items:
                packed = nbytes // 4  # the packed weight (two 4-bit codes per byte of a half value)
                with self.cond:
                    while not self.stop and self.held and self.held + packed > WINDOW:
                        self.cond.wait()
                    if self.stop:
                        return
                staged = view(path, offset, nbytes)
                if staged is None:
                    raise RuntimeError(f"{key}: not in the checkpoint read ahead")
                w = torch.empty(shape, dtype=self.half, device=self.device)
                raw = w.view(src_dtype)  # the same element size: cast in place, as load_pinned.py's GPU cast
                raw.copy_(staged.view(src_dtype).view(shape), non_blocking=True)
                if src_dtype != self.half:
                    w.copy_(raw)
                w4, state = _quantize_4bit(w, self.code)
                del w, raw
                with self.cond:
                    self.results[(path, key)] = (w4, state, packed)
                    self.held += packed
                    LAST["made"] += 1
                    self.cond.notify_all()
        except BaseException as exc:  # noqa: BLE001 - the loop quantizes what is left itself
            self.error = exc
            LAST["dropped"] = f"thread: {exc!r}"[:300]
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()
            LAST["seconds"] = round(time.perf_counter() - started, 3)

    def take(self, path, key):
        """The thread's (w4, quant_state) for `key` of `path` (waits while the thread is on its way to
        it), or None."""
        started = time.perf_counter()
        with self.cond:
            while not self.stop and (path, key) not in self.results and not self.done and self.error is None:
                self.cond.wait()
            found = self.results.pop((path, key), None)
            if found is not None:
                self.held -= found[2]
                self.cond.notify_all()
        waited = time.perf_counter() - started
        LAST["waited"] += waited
        if LAST["first_wait"] is None:
            LAST["first_wait"] = round(waited, 3)
        return None if found is None else found[:2]

    def close(self):
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        self.thread.join()
        self.results.clear()


_STATE = {"spec": None, "handover": {}}
_LOCK = threading.Lock()


def start(half=None):
    """Starts the thread (once) when the switch is on and load_early.py is reading the checkpoint. `half`:
    the dtype the run loads in (default: the run's half dtype, fp16 on the T4; the hand-over checks it)."""
    if not enabled() or not torch.cuda.is_available():
        return None
    from . import load_early
    from .warmup import training_half_dtype

    import bitsandbytes as bnb

    with _LOCK:
        if _STATE["spec"] is not None or load_early._STATE["thread"] is None or not load_early._STATE["files"]:
            return _STATE["spec"]
        if bnb.__version__ != "0.50.2" or torch.get_default_dtype() != torch.float32:
            return None  # _quantize_4bit follows that release; the table must be made under float32
        device = torch.cuda.current_device()
        code = bnb.functional.get_4bit_type(QUANT["quant_type"], device=torch.device("cuda", device))
        spec = _Spec(list(load_early._STATE["files"].values()), half or training_half_dtype(), device, code)
        if not spec.items:
            return None
        _STATE["spec"] = spec
        spec.thread.start()
        return spec


def drop(reason):
    """No more hand-overs: the loop quantizes every later tensor itself."""
    spec = _STATE["spec"]
    if spec is not None and not spec.stop:
        LAST["dropped"] = LAST["dropped"] or reason
        spec.close()


def stop():
    """When the load returns (or raises): the thread is stopped and joined (before the pinned chunks it
    reads from are released); results nobody took are dropped."""
    spec = _STATE["spec"]
    if spec is not None:
        spec.close()
    _STATE["handover"].clear()


def stand_in(path, key, shape, dtype, device, model):
    """For the loop's job of checkpoint tensor `key`: a stand-in carrying the thread's result when every
    check passes, else None (the job runs as usual)."""
    spec = _STATE["spec"]
    if spec is None or spec.stop:
        return None
    from .load_early import _covered

    read = _covered(path)  # the file read ahead that `path` is (transformers opens the cache's symlinks)
    path = read.path if read is not None else path
    if (path, key) not in spec.keys:
        return None  # not one of the weights it quantizes (norms, biases, the embedding): the job as usual
    try:
        import bitsandbytes as bnb

        module_name, _, field = key.rpartition(".")
        module = model.get_submodule(module_name)
        param = getattr(module, field)
        attrs = (getattr(param, "blocksize", None), getattr(param, "compress_statistics", None),
                 getattr(param, "quant_type", None), getattr(param, "quant_storage", None))
        if (not isinstance(module, bnb.nn.Linear4bit) or not isinstance(param, bnb.nn.Params4bit)
                or getattr(param, "bnb_quantized", True) or dtype != spec.half or torch.device(device) != torch.device("cuda", spec.device)
                or attrs != (QUANT["blocksize"], QUANT["compress_statistics"], QUANT["quant_type"], QUANT["quant_storage"])
                or tuple(param.shape) != tuple(shape)):
            drop(f"{key}: not the speculated quantization")
            return None
    except Exception as exc:  # noqa: BLE001 - the loop's own path
        drop(f"{key}: {exc!r}")
        return None
    found = spec.take(path, key)
    if found is None:
        drop(f"{key}: no result" + (f" ({spec.error!r})" if spec.error is not None else ""))
        return None
    w4, state = found
    if tuple(state.shape) != tuple(shape) or state.dtype != dtype:
        drop(f"{key}: result of another shape or dtype")
        return None
    placeholder = torch.empty(1, dtype=dtype, device=device).expand(*shape)
    _STATE["handover"][placeholder.data_ptr()] = (w4, state)
    LAST["handed"] += 1
    return placeholder


def _quantize(original):
    def quantize(self, device):
        data = self.data
        found = None
        if data.dim() and all(stride == 0 for stride in data.stride()):
            found = _STATE["handover"].pop(data.data_ptr(), None)
        if found is None:
            return original(self, device)
        w4, state = found  # what original computes: bnb.functional.quantize_4bit(self.data ...) of the same values
        self.data = w4
        self.quant_state = state
        if self.module is not None:
            self.module.quant_state = state
        self.bnb_quantized = True
        return self

    quantize._soup_original = original
    return quantize


def install_handover():
    """Params4bit._quantize takes the stand-ins (while a handled load runs). Returns the undo."""
    import bitsandbytes as bnb

    cls = bnb.nn.Params4bit
    current = cls.__dict__.get("_quantize")
    if current is None or getattr(current, "_soup_original", None) is not None:
        return lambda: None
    patched = _quantize(current)
    cls._quantize = patched

    def undo():
        if cls.__dict__.get("_quantize") is patched:
            cls._quantize = current

    return undo


def report():
    if not LAST["made"] and not LAST["dropped"]:
        return ""
    return (f"[solution] load spec: {LAST['made']} weights quantized ahead by a thread in "
            f"{LAST['seconds'] if LAST['seconds'] is not None else float('nan'):.2f} s, {LAST['handed']} handed over, "
            f"the loop waited {LAST['waited']:.2f} s (first {LAST['first_wait']})"
            + (f"; dropped: {LAST['dropped']}" if LAST["dropped"] else ""))
