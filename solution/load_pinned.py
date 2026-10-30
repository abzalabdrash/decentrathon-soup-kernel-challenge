"""The checkpoint -> GPU copies of the model load go through pinned host buffers that a thread fills
ahead of transformers' loader (Soup's setup only, as part of load_stream.py's loads).

Measured on the T4 (session soup-t12): transformers' tensor loop
(`core_model_loading.convert_and_load_state_dict_in_model`) takes 3.2-3.5 s, about 1.75 s of it in
`_materialize_copy`'s `tensor.to(cuda, float16)` of each checkpoint tensor. With on-the-fly 4-bit
quantization the loop is sequential: its first pass only creates one deferred job per checkpoint
tensor (`spawn_materialize`), its second pass calls them one by one, and each `.to()` converts the
bf16 slice of the safetensors mmap to fp16 on the CPU into a fresh pageable tensor (a blocking copy
converts on the CPU: aten/src/ATen/native/cuda/Copy.cu, copy_kernel_cuda), then copies that to the
GPU from pageable memory and waits, while the GPU and the copy engine idle in between.

Here, for the loads load_stream.py handles (a CUDA Qwen2ForCausalLM quantized to 4 bits on the fly
by bitsandbytes on one GPU, inside Soup's `_setup_transformers`):
- `spawn_materialize` records every checkpoint slice that goes to this GPU, in the order the second
  pass will ask for them (the order the first pass creates their jobs);
- when the second pass begins, a thread fills a ring of K pinned host buffers
  (SOUP_SOLUTION_LOAD_PINNED_SLOTS, default 4, each the size of the largest tensor: 26 MiB for
  Qwen2.5-1.5B) with the next tensors, from the same `slice[...]`;
- the bf16 -> fp16 cast: with SOUP_SOLUTION_LOAD_PINNED_CAST=gpu (default) the thread copies the raw
  bf16 bytes (one memmove in one thread) and the GPU converts them in place in the tensor after its
  copy (torch's CUDA cast; both it and torch's CPU cast are IEEE round-to-nearest-even: the tests
  check all 65,536 bf16 values and every tensor of the real checkpoint bit for bit); with =cpu the
  thread converts into an fp16 view of the buffer with `copy_`, the kernel the original `.to()`
  runs, so the bytes are the original ones by construction, but that conversion is slow on the T4
  host (1.2 GB/s in the thread: the loop then waits for it);
- the job of a tensor allocates its GPU tensor where `.to()` does (current stream, same shape and
  dtype) and copies the buffer into it with `non_blocking=True`; an event after each copy tells the
  thread when the buffer may be refilled (no record_stream). With SOUP_SOLUTION_LOAD_PINNED_WINDOW=W
  (default 0, up to 2) the next W tensors are copied ahead on a side stream, ordered by events: the side
  stream first waits for the default stream's past (the allocator may hand out a block the default
  stream's work freed), the default stream waits for the copy before the tensor is handed over, so
  every later use and the free are ordered after it. The load's GPU peak grows by at most W tensors;
  W=0 allocates exactly what the original does, and on the T4 W=1 was not faster (the loop's own
  Python and quantization work is what is left);
- the rest of the loop is untouched: bitsandbytes' Bnb4bitQuantize (quantize_4bit per weight, the
  double-quant offset from that weight's own absmax) on the same fp16 tensor, so the packed
  weights, absmax and quant states are bit for bit those of the original load.
T4 (session soup-night-load, ABBA, 2 rounds each): tensor loop 3.91 s -> 2.24 s (CPU cast 3.12 s),
Soup's model setup 5.33 -> 3.67 s, peak 539 MiB unchanged; the first step comes only 0.1-0.2 s
earlier, because it then waits 0.7-0.9 s for the warm-up thread's kernel compilation (7.2 s).
The embedding stays on load_stream.py's path (read to the host, its job is not ours). Anything
unexpected (a tensor asked out of order or from another thread, a non-default current stream:
bitsandbytes launches its kernels on the legacy default stream, a slice that is not what was
planned, an error) takes the original job for that tensor, and for every later one unless only
that tensor's staging failed. Switch: SOUP_SOLUTION_LOAD_PINNED (on by default since the night
integration; =0 takes the original loader).
"""

import ctypes
import functools
import inspect
import os
import threading
import time

import torch

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
_FLOATS = (torch.float16, torch.bfloat16, torch.float32)
MAX_TENSOR_BYTES = 256 << 20  # larger tensors take the original path (the ring holds K of the largest)
LAST = {"copier": None}  # the last copier closed (counters, for the log and the tests)


def enabled():
    return os.environ.get("SOUP_SOLUTION_LOAD_PINNED", "1") not in ("0", "false", "no", "")


def _env_int(name, default, low, high):
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        return default
    return min(max(value, low), high)


def new_copier(device, log=print):
    """A PinnedCopier for a load that load_stream.py handles (to `device`), or None when switched off
    or when one cannot be made (the load then runs as before)."""
    if not enabled():
        return None
    try:
        cast = os.environ.get("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu").strip().lower()
        return PinnedCopier(device, window=_env_int("SOUP_SOLUTION_LOAD_PINNED_WINDOW", 0, 0, 2),
                            slots=_env_int("SOUP_SOLUTION_LOAD_PINNED_SLOTS", 4, 2, 16),
                            cast="cpu" if cast == "cpu" else "gpu", log=log)
    except Exception as exc:  # noqa: BLE001 - load as usual
        log(f"[solution] pinned load copies skipped: {exc!r}")
        return None


def _as_device(value):
    try:
        if isinstance(value, int) and not isinstance(value, bool):
            return torch.device("cuda", value)
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError):
        return None
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


class _Entry:
    """One checkpoint tensor of the plan and where it is: staged in a buffer, copied (ahead) to the GPU."""

    __slots__ = ("index", "source", "shape", "src_dtype", "dtype", "stage_dtype", "nbytes", "job", "ready",
                 "released", "staged", "error", "out", "event")

    def __init__(self, index, source, shape, src_dtype, dtype, stage_dtype, job):
        self.index, self.source, self.shape = index, source, shape
        self.src_dtype, self.dtype, self.job = src_dtype, dtype, job
        self.stage_dtype = stage_dtype  # in the buffer and copied: `dtype`, or `src_dtype` cast on the GPU
        numel = 1
        for size in shape:
            numel *= size
        self.nbytes = numel * stage_dtype.itemsize
        self.ready = threading.Event()  # staged in its buffer (or staging gave up on it)
        self.released = threading.Event()  # its copy out of the buffer is issued (event set) or skipped
        self.staged = self.error = self.out = self.event = None


class PinnedCopier:
    """One load's checkpoint -> GPU copies through a ring of pinned host buffers (see the module
    docstring). `add` is called from transformers' spawn_materialize (load_stream.py's hook) during
    the loop's first pass and returns the job the second pass calls; `close` ends it when the load
    returns or raises. Device operations are small methods, so that the CPU tests can replace them."""

    def __init__(self, device, window=0, slots=4, cast="gpu", log=print):
        self.device = torch.device(device)
        self.window, self.slots, self.log = max(0, int(window)), max(1, int(slots)), log
        self.cast = cast
        self._owner = threading.get_ident()
        self._entries = []
        self._next = 0  # index of the entry the loop must ask for next
        self._started = self._closed = self._stop = self._broken = False
        self._filler = None
        self._buffers = []
        self._signature = None
        self.slot_bytes = self.planned = 0
        self.reason = None  # why the copies stopped early, if they did
        self.copied = self.copied_ahead = self.fallbacks = self.copied_bytes = 0
        self.wait_seconds = 0.0  # the loop waiting for a tensor to be staged
        self.fill_seconds = 0.0  # the thread staging (slice and conversion into a buffer)
        self.slot_wait_seconds = 0.0  # the thread waiting for a buffer's copy to finish
        self.thread_seconds = 0.0
        self.created = time.perf_counter()
        self._side = self._make_side_stream() if self.window > 0 else None

    # --- device operations (CUDA); the CPU tests replace these -------------------------------------

    def _make_side_stream(self):
        return torch.cuda.Stream(self.device)

    def _thread_setup(self):
        torch.cuda.set_device(self.device)

    def _pin(self, nbytes):
        from .weight_stream import PinnedBuffer

        return PinnedBuffer(nbytes)

    def _on_default_stream(self):
        return (torch.cuda.current_stream(self.device).cuda_stream
                == torch.cuda.default_stream(self.device).cuda_stream)

    def _copy(self, entry):
        """The tensor allocated as `.to()` allocates it and copied now on the current (default)
        stream; returns it and an event recorded after the copy."""
        stream = torch.cuda.current_stream(self.device)
        out = torch.empty(entry.shape, dtype=entry.dtype, device=self.device)
        self._copy_into(out, entry)
        event = torch.cuda.Event(blocking=True)  # the thread sleeps in synchronize() instead of spinning
        event.record(stream)
        return out, event

    def _copy_ahead(self, entry):
        """The tensor allocated on the default stream and copied on the side stream once the default
        stream's work so far is done (its block may come from a tensor that work still reads)."""
        out = torch.empty(entry.shape, dtype=entry.dtype, device=self.device)
        allocated = torch.cuda.Event()
        allocated.record(torch.cuda.current_stream(self.device))
        self._side.wait_event(allocated)
        with torch.cuda.stream(self._side):
            self._copy_into(out, entry)
        event = torch.cuda.Event(blocking=True)
        event.record(self._side)
        return out, event

    @staticmethod
    def _copy_into(out, entry):
        """The staged bytes into `out` on the current stream; staged in the checkpoint's dtype, they
        land in `out`'s own memory and are cast there in place (same element size, full overlap)."""
        if entry.stage_dtype == entry.dtype:
            out.copy_(entry.staged, non_blocking=True)
        else:
            raw = out.view(entry.stage_dtype)
            raw.copy_(entry.staged, non_blocking=True)
            out.copy_(raw)

    def _hand_over(self, event):
        """The default stream waits for a copy made ahead before the tensor is used (and freed) on it."""
        torch.cuda.current_stream(self.device).wait_event(event)

    def _synchronize(self):
        torch.cuda.synchronize(self.device)

    # --- the plan (first pass) ----------------------------------------------------------------------

    def add(self, job, spawn, args, kwargs):
        """The job the loop will call for the tensor `spawn(*args, **kwargs)` was asked to load:
        ours when it is a checkpoint slice going to this GPU and the plan is still open, else the
        loop's own `job` (wrapped, while the plan is open, so that the second pass's first call of
        any job starts the thread)."""
        try:
            if self._started or self._closed or threading.get_ident() != self._owner or not callable(job):
                return job
            if self._signature is None:
                self._signature = inspect.signature(spawn)
            bound = self._signature.bind(*args, **kwargs).arguments
            entry = None
            if bound.get("thread_pool") is None and bound.get("sharding_op") is None:
                entry = self._plan(bound.get("tensor"), bound.get("device"), bound.get("dtype"), job)
        except Exception:  # noqa: BLE001 - this tensor the original way
            return job
        if entry is None:
            return functools.partial(self._other, job)
        self._entries.append(entry)
        return functools.partial(self._take, entry)

    def _plan(self, source, device, dtype, job):
        if torch.is_tensor(source) or not all(hasattr(source, name) for name in ("get_shape", "get_dtype",
                                                                                  "__getitem__")):
            return None  # not a safetensors slice
        if _as_device(device) != self.device:
            return None
        src_dtype = _DTYPES.get(source.get_dtype())
        target = src_dtype if dtype is None else dtype
        if src_dtype is None or target not in _FLOATS:
            return None
        stage = src_dtype if self.cast == "gpu" and src_dtype.itemsize == target.itemsize else target
        entry = _Entry(len(self._entries), source, tuple(int(size) for size in source.get_shape()),
                       src_dtype, target, stage, job)
        return entry if 0 < entry.nbytes <= MAX_TENSOR_BYTES else None

    # --- the jobs (second pass) ---------------------------------------------------------------------

    def _start(self):
        """The second pass began: the plan is closed and the thread starts staging."""
        self._started = True
        if self._entries and not self._closed:
            self.slot_bytes = max(entry.nbytes for entry in self._entries)
            self._filler = threading.Thread(target=self._fill, name="soup-load-pinned", daemon=True)
            self._filler.start()

    def _other(self, job):
        if not self._started and threading.get_ident() == self._owner:
            self._start()
        return job()

    def _take(self, entry):
        """The job of a planned tensor: the GPU tensor `_materialize_copy` would return."""
        if threading.get_ident() != self._owner:
            return entry.job()
        if not self._started:
            self._start()
        if self._broken or self._closed:
            return entry.job()
        if entry.index != self._next:
            self._abandon(f"tensor {entry.index} asked when {self._next} was due")
            return entry.job()
        if not self._on_default_stream():
            self._abandon("the current stream is not the default stream")
            return entry.job()
        self._next += 1
        try:
            out = self._get(entry)
        except Exception as exc:  # noqa: BLE001 - this tensor and the rest the original way
            self._abandon(repr(exc))
            return entry.job()
        if out is None:  # staging gave up on this tensor only
            self.fallbacks += 1
            return entry.job()
        try:
            self._ahead(entry.index + 1)
        except Exception as exc:  # noqa: BLE001 - `out` is complete; the rest the original way
            self._abandon(repr(exc))
        return out

    def _get(self, entry):
        if entry.out is not None:  # copied ahead on the side stream
            out, entry.out = entry.out, None
            self._hand_over(entry.event)
            return out
        started = time.perf_counter()
        entry.ready.wait()
        self.wait_seconds += time.perf_counter() - started
        if entry.staged is None:
            entry.released.set()
            return None
        out, entry.event = self._copy(entry)
        entry.staged = None
        entry.released.set()
        self.copied += 1
        self.copied_bytes += entry.nbytes
        return out

    def _ahead(self, start):
        """Copies the next tensors on the side stream, up to the window, if they are staged already
        (one that is not is copied when it is asked for)."""
        if self._side is None:
            return
        for index in range(start, min(start + self.window, len(self._entries))):
            entry = self._entries[index]
            if entry.event is not None:
                continue  # copied ahead already
            if not entry.ready.is_set() or entry.staged is None:
                return
            entry.out, entry.event = self._copy_ahead(entry)
            entry.staged = None
            entry.released.set()
            self.copied += 1
            self.copied_ahead += 1
            self.copied_bytes += entry.nbytes

    # --- the staging thread -------------------------------------------------------------------------

    def _fill(self):
        started = time.perf_counter()
        try:
            self._thread_setup()
            for entry in self._entries:
                if self._stop:
                    break
                slot = entry.index % self.slots
                if entry.index >= self.slots:  # the buffer's previous tensor must be out of it
                    previous = self._entries[entry.index - self.slots]
                    waited = time.perf_counter()
                    previous.released.wait()
                    if self._stop:
                        break
                    if previous.event is not None:
                        previous.event.synchronize()
                    self.slot_wait_seconds += time.perf_counter() - waited
                if slot == len(self._buffers):
                    self._buffers.append(self._pin(self.slot_bytes))
                filled = time.perf_counter()
                try:
                    entry.staged = self._stage(entry, self._buffers[slot].buffer)
                except Exception as exc:  # noqa: BLE001 - this tensor takes the original path
                    entry.error = exc
                self.fill_seconds += time.perf_counter() - filled
                entry.ready.set()
        except Exception as exc:  # noqa: BLE001 - the tensors not staged take the original path
            self.reason = self.reason or f"staging thread: {exc!r}"
        finally:
            for entry in self._entries:
                entry.ready.set()
            self.thread_seconds = time.perf_counter() - started

    @staticmethod
    def _stage(entry, buffer):
        """The checkpoint slice (the same `slice[...]` as `_materialize_copy`) in the buffer: converted
        as its `.to(device, dtype)` converts it on the CPU before the copy to the GPU (the same
        kernel), or its raw bytes when the GPU casts them (or when there is nothing to cast)."""
        source = entry.source[...]
        if (not torch.is_tensor(source) or source.device.type != "cpu" or source.dtype != entry.src_dtype
                or tuple(source.shape) != entry.shape):
            raise ValueError(f"unexpected slice: {type(source).__name__} {getattr(source, 'dtype', None)} "
                             f"{tuple(getattr(source, 'shape', ()))}")
        staged = buffer[:entry.nbytes].view(entry.stage_dtype).view(entry.shape)
        if entry.stage_dtype == source.dtype and source.is_contiguous():
            ctypes.memmove(staged.data_ptr(), source.data_ptr(), entry.nbytes)  # one thread, no GIL
        else:
            staged.copy_(source)
        return staged

    # --- the end ------------------------------------------------------------------------------------

    def _stop_filler(self):
        self._stop = True
        for entry in self._entries:
            entry.released.set()
        if self._filler is not None:
            self._filler.join()

    def _drop_ahead(self):
        """Copies made ahead that nobody took: the default stream waits for them before their memory
        can be reused, then they are dropped."""
        for entry in self._entries:
            if entry.out is not None:
                try:
                    self._hand_over(entry.event)
                except Exception:  # noqa: BLE001 - CUDA is broken already
                    pass
                entry.out = None

    def _abandon(self, reason):
        """This tensor and every later one take the original path."""
        if self._broken:
            return
        self._broken = True
        self.reason = self.reason or reason
        self._stop_filler()
        self._drop_ahead()

    def close(self):
        """Called by load_stream.py when the load returns or raises: the thread stops, the copies in
        flight finish, the buffers are unpinned; one line for the log."""
        if self._closed:
            return
        self._closed = True
        self._stop_filler()
        self._drop_ahead()
        buffers, self._buffers = self._buffers, []
        try:
            if buffers or self.copied:
                self._synchronize()  # every copy out of the buffers is done before they are unpinned
        except Exception as exc:  # noqa: BLE001 - CUDA is broken already
            self.reason = self.reason or f"close: {exc!r}"
        for buffer in buffers:
            buffer.unpin()
        for entry in self._entries:
            entry.staged = None
        self.planned, self._entries = len(self._entries), []  # the slices (and the mmap) are not kept
        LAST["copier"] = self
        if self.planned:
            self.log(f"[solution] pinned load copies: {self.copied} of {self.planned} checkpoint tensors "
                     f"({self.copied_bytes / 2**20:.0f} MiB, {self.copied_ahead} copied ahead on a side stream, "
                     f"window {self.window}, cast on the {self.cast.upper()}) through {len(buffers)} pinned buffers of "
                     f"{self.slot_bytes / 2**20:.1f} MiB; staging {self.fill_seconds:.2f} s in a thread "
                     f"(+{self.slot_wait_seconds:.2f} s waiting for buffers), the loop waited "
                     f"{self.wait_seconds:.2f} s for staging; {self.fallbacks} tensors the original way"
                     + (f"; stopped early: {self.reason}" if self.reason else ""))
