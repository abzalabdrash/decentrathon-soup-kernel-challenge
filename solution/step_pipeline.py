"""The training step's host work moved off the GPU's critical path (task B1).

On the T4 trace of main (a6c3fe3, steps 8-9 under the profiler) the GPU idled ~31 ms at every step
boundary while the CPU worked alone. After the backward is queued, transformers' nan/inf check of
the step's loss (`torch.isnan(loss) or torch.isinf(loss)`, a host read) waited 63 ms for the GPU to
finish the backward, and from there the CPU did everything else with the GPU empty: the FLOP
counter's walk over the model (4.9 ms, boundary.py), the gradient clip's launches (5 ms), AdamW
(2.8 ms), the log and Soup's callbacks (4.6 ms), the DataLoader's fetch and collate of the batch
after next (2.7 ms), the next step's start (3.4 ms) and the forward's own host reads (the unpad
plan's `nonzero` and `tolist`, the cu_seqlens copy: 2.7 ms). Inside the step, the fused head's
labelled-token count and `num_items_in_batch` are read from the GPU at the head (the CPU waited
70 ms there for the forward).

Here, each behind its switch, for the real CUDA trainer only (anything else runs the original):

- SOUP_SOLUTION_NAN_FILTER=gpu: the filter is applied on the GPU. transformers adds the step's loss
  to its running `_tr_loss`, or, when the loss is nan or inf, `_tr_loss / (1 + steps since the last
  log)`; `training_step` here returns `torch.where(isfinite(loss), loss, _tr_loss / k)` and the
  host check is switched off for the epoch (`args.logging_nan_inf_filter`, restored afterwards and
  around any checkpoint save), so `_tr_loss` gets the same addend bit for bit and no read waits for
  the backward. The CPU goes on queueing the clip while the GPU still runs the backward; the
  scaler's `found_inf.item()` in the optimizer step stays (the step decision needs it) and is now
  the boundary's only wait besides the harness's own.
- SOUP_SOLUTION_PREFETCH_BATCH=1: accelerate's DataLoaderShard fetches one batch ahead of the one
  it yields; the batch after that is fetched (sampler, dataset rows, TRL's collator, pinning) right
  after the backward is queued, while the CPU would only wait for the GPU, and handed to the
  DataLoaderShard when it asks. Same batches in the same order; the sampler's indices reach Soup's
  rewind FIFO earlier but in the same order (it pops them per batch in compute_loss).
- SOUP_SOLUTION_HOST_PLAN=1: the unpad plan of a batch (packed ids, positions and indices of the
  real tokens, the row spans, the attention sequences with the alignment tail and their int32
  cu_seqlens), the number of labelled packed tokens and the label counts behind
  `num_items_in_batch` are computed from the batch's CPU tensors when the DataLoader hands it over
  (the same integer operations as unpad.pack_batch on the GPU), and copied to the GPU in one
  non-blocking copy next to the batch. The forward (fused_ce.py) and the head take them by the
  identity of the batch's GPU tensors; any other tensor takes the GPU path.
- SOUP_SOLUTION_BWD_SINGLE_THREAD=1: the trainer's `accelerator.backward(loss)` runs under
  `torch.autograd.set_multithreading_enabled(False)`: the autograd engine then evaluates every node
  on the calling thread instead of handing the CUDA nodes to its device thread (the same nodes, the
  same order, the same streams: the engine sets each node's stream itself). The main thread's cuBLAS
  handle, and so its workspace (8 MiB per handle and stream in torch 2.11), then serves the backward
  too, instead of a second handle made by the autograd thread; and no GIL handoff between the two
  threads per Python node of our autograd functions.
- SOUP_SOLUTION_BOUNDARY_TIMING=1 (diagnostics): host time of the callbacks per event, the log,
  Soup's tracker commit and display, the DataLoader fetch, printed once per epoch.
"""

import functools
import os
import time
import weakref
from collections.abc import Mapping

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _env(name, default):
    return os.environ.get(name, default).strip().lower()


def _flag(name, default="0"):
    return _env(name, default) not in ("0", "false", "no", "")


_STATE = {
    "nan_gpu": _env("SOUP_SOLUTION_NAN_FILTER", "host") == "gpu",
    "prefetch": _flag("SOUP_SOLUTION_PREFETCH_BATCH"),
    "plan": _flag("SOUP_SOLUTION_HOST_PLAN"),
    "timing": _flag("SOUP_SOLUTION_BOUNDARY_TIMING"),
    "bwd_single": _flag("SOUP_SOLUTION_BWD_SINGLE_THREAD"),
    "patched": False,
}
_STATS = {"prefetched": 0, "planned": 0, "plans_used": 0, "known_used": 0, "fetch_s": 0.0, "plan_s": 0.0}
_TIMES = {}  # label -> [seconds, calls] (SOUP_SOLUTION_BOUNDARY_TIMING)


def _timed(label, started):
    entry = _TIMES.setdefault(label, [0.0, 0])
    entry[0] += time.perf_counter() - started
    entry[1] += 1


# ---- values the host knows for GPU tensors ---------------------------------------------------

_KNOWN = {}  # id(tensor) -> (weakref to the tensor, value)


def remember(tensor, value):
    """Records that `tensor` (a GPU tensor) holds `value`, as long as that tensor object lives."""
    key = id(tensor)

    def drop(ref, key=key):
        entry = _KNOWN.get(key)
        if entry is not None and entry[0] is ref:
            del _KNOWN[key]

    _KNOWN[key] = (weakref.ref(tensor, drop), value)


def known(tensor):
    """The value recorded for this very tensor object, or None."""
    if not torch.is_tensor(tensor):
        return None
    entry = _KNOWN.get(id(tensor))
    if entry is None or entry[0]() is not tensor:
        return None
    _STATS["known_used"] += 1
    return entry[1]


# ---- the unpad plan from the batch's CPU tensors ----------------------------------------------


class HostPlan:
    """unpad.pack_batch's outputs and the counts of one batch, made on the host. `ids`, `pos` and
    `idx` are GPU views of one copied buffer; `cu` is the int32 cu_seqlens of `bounds`."""

    __slots__ = ("ids", "pos", "idx", "spans", "bounds", "cu", "n_labelled", "items_shifted",
                 "items_all", "_refs", "__weakref__")

    def matches(self, input_ids, attention_mask, labels):
        refs = self._refs
        return refs[0]() is input_ids and refs[1]() is attention_mask and refs[2]() is labels


def _cpu_plan(batch):
    """The plan's CPU part for a collated batch, or None for anything unpad.pack_batch would not see
    the same way (missing keys, other ranks, shapes or dtypes)."""
    if not isinstance(batch, Mapping):
        return None
    ids, mask, labels = batch.get("input_ids"), batch.get("attention_mask"), batch.get("labels")
    tensors = (ids, mask, labels)
    if not all(torch.is_tensor(t) and t.device.type == "cpu" and t.dim() == 2 for t in tensors):
        return None
    if ids.shape != mask.shape or labels.shape != ids.shape or "shift_labels" in batch:
        return None
    if ids.dtype != torch.int64 or labels.dtype != torch.int64 or ids.numel() == 0:
        return None
    from . import unpad

    m = mask.to(torch.bool)
    lengths = m.sum(dim=1)
    idx = m.flatten().nonzero(as_tuple=False).squeeze(1)
    pos = (torch.cumsum(m.to(torch.int64), dim=1) - 1).flatten()[idx]
    ends = torch.cumsum(lengths, dim=0).tolist()
    spans = [(end - length, end) for end, length in zip(ends, lengths.tolist()) if length > 0]
    n = idx.numel()
    if n == 0:
        return None
    shifted = F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:]  # fused_ce.shift_labels
    host = torch.empty(3 * n, dtype=torch.int64, pin_memory=True)
    host[:n] = ids.flatten()[idx]
    host[n:2 * n] = pos
    host[2 * n:] = idx
    bounds = unpad.sequences(spans, n + unpad._pad_count(n))
    cu = torch.tensor(bounds, dtype=torch.int32).pin_memory() if bounds is not None else None
    return {"host": host, "n": n, "spans": spans, "bounds": bounds, "cu": cu,
            "n_labelled": int((shifted.flatten()[idx] != IGNORE_INDEX).sum()),
            "items_shifted": int((labels[..., 1:] != IGNORE_INDEX).sum()),
            "items_all": int((labels != IGNORE_INDEX).sum())}


_PLANS = {}  # id(GPU attention_mask) -> HostPlan
_CPU_PLANS = {}  # id(CPU batch) -> (batch, its CPU plan): from the fetch to send_to_device


def _attach(cpu, gpu_batch):
    """Copies a CPU plan next to its batch's GPU tensors and registers it for them."""
    ids, mask, labels = (gpu_batch.get(k) for k in ("input_ids", "attention_mask", "labels"))
    if not all(torch.is_tensor(t) and t.is_cuda for t in (ids, mask, labels)):
        return
    n = cpu["n"]
    buf = cpu["host"].to(mask.device, non_blocking=True)
    plan = HostPlan()
    plan.ids, plan.pos, plan.idx = buf[:n].unsqueeze(0), buf[n:2 * n].unsqueeze(0), buf[2 * n:]
    plan.spans, plan.bounds = cpu["spans"], cpu["bounds"]
    plan.cu = cpu["cu"].to(mask.device, non_blocking=True) if cpu["cu"] is not None else None
    plan.n_labelled, plan.items_shifted, plan.items_all = (
        cpu["n_labelled"], cpu["items_shifted"], cpu["items_all"])
    key = id(mask)

    def drop(ref, key=key):
        entry = _PLANS.get(key)
        if entry is not None and entry._refs[1] is ref:
            del _PLANS[key]

    plan._refs = (weakref.ref(ids), weakref.ref(mask, drop), weakref.ref(labels))
    _PLANS[key] = plan


def plan_for(input_ids, attention_mask, labels):
    """The host plan of the batch these GPU tensors came in, or None (taken once)."""
    if not _STATE["plan"] or not torch.is_tensor(attention_mask):
        return None
    plan = _PLANS.get(id(attention_mask))
    if plan is None or not plan.matches(input_ids, attention_mask, labels):
        return None
    del _PLANS[id(attention_mask)]
    _STATS["plans_used"] += 1
    return plan


def _plan_for_labels(labels):
    for plan in _PLANS.values():
        if plan._refs[2]() is labels:
            return plan
    return None


# ---- the DataLoader iterator one batch further ahead ----------------------------------------


class _Prefetching:
    """Wraps the torch DataLoader iterator that accelerate's DataLoaderShard reads: `prefetch()`
    fetches the next batch now, `__next__` hands it over (or fetches as usual)."""

    def __init__(self, inner):
        self.inner = inner
        self.ahead = None  # ("batch", batch) | ("stop", None) | ("error", exception)
        self.done = False

    def __iter__(self):
        return self

    def _fetch(self):
        started = time.perf_counter()
        batch = next(self.inner)
        fetched = time.perf_counter()
        _STATS["fetch_s"] += fetched - started
        if _STATE["plan"]:
            try:
                cpu = _cpu_plan(batch)
            except Exception:  # noqa: BLE001 - the GPU path plans this batch instead
                cpu = None
            if cpu is not None:
                _CPU_PLANS[id(batch)] = (batch, cpu)
                _STATS["planned"] += 1
            _STATS["plan_s"] += time.perf_counter() - fetched
        return batch

    def __next__(self):
        ahead, self.ahead = self.ahead, None
        if ahead is not None:
            kind, value = ahead
            if kind == "batch":
                return value
            if kind == "stop":
                raise StopIteration
            raise value
        if self.done:
            raise StopIteration
        try:
            return self._fetch()
        except StopIteration:
            self.done = True
            raise

    def prefetch(self):
        if self.ahead is not None or self.done:
            return
        try:
            self.ahead = ("batch", self._fetch())
            _STATS["prefetched"] += 1
        except StopIteration:
            self.done = True
            self.ahead = ("stop", None)
        except Exception as exc:  # noqa: BLE001 - raised where the DataLoader would have raised it
            self.ahead = ("error", exc)


def _send_to_device(original, tensor, *args, **kwargs):
    entry = _CPU_PLANS.pop(id(tensor), None) if _CPU_PLANS else None
    out = original(tensor, *args, **kwargs)
    if entry is not None and entry[0] is tensor and isinstance(out, Mapping):
        _attach(entry[1], out)
    return out


def _install_iterator(trainer, train_dataloader):
    """Puts _Prefetching between the DataLoaderShard and its torch DataLoader for one epoch; returns
    the undo function or None when this is not the plain single-process layout."""
    base = getattr(train_dataloader, "base_dataloader", None)
    if (not isinstance(base, torch.utils.data.DataLoader) or base.num_workers != 0
            or getattr(train_dataloader, "use_stateful_dataloader", False)
            or "__iter__" in vars(base) or type(train_dataloader).__name__ != "DataLoaderShard"):
        return None
    original_iter = type(base).__iter__

    def open_iterator():
        wrapper = _Prefetching(original_iter(base))
        trainer.__dict__["_soup_prefetcher"] = wrapper if _STATE["prefetch"] else None
        return wrapper

    base.__iter__ = open_iterator

    def undo():
        base.__dict__.pop("__iter__", None)
        trainer.__dict__.pop("_soup_prefetcher", None)
        _CPU_PLANS.clear()

    return undo


# ---- the trainer patches ---------------------------------------------------------------------


def gpu_filtered_loss(trainer, loss):
    """What transformers adds to `_tr_loss` for this step's loss: `_tr_loss / (1 + global_step -
    last_logged)` when the loss is nan or inf, the loss otherwise; chosen on the GPU."""
    tr_loss = trainer._tr_loss
    k = 1 + trainer.state.global_step - trainer._globalstep_last_logged
    if torch.is_tensor(loss) and loss.dim() == 0 and loss.device == tr_loss.device:
        return torch.where(torch.isfinite(loss), loss, tr_loss / k)
    if torch.isnan(loss) or torch.isinf(loss):  # anything else: transformers' own host check
        return tr_loss / k
    return loss


def _real_cuda_trainer(trainer):
    from .offload import _real_cuda_trainer as real

    return real(trainer)


def patch_step_pipeline():
    """Trainer._run_epoch / training_step / _get_num_items_in_batch / _save_checkpoint and accelerate's
    send_to_device (see the module doc); nothing unless one of the switches is on."""
    if _STATE["patched"] or not (_STATE["nan_gpu"] or _STATE["prefetch"] or _STATE["plan"]
                                 or _STATE["timing"] or _STATE["bwd_single"]):
        return
    import accelerate.data_loader as adl
    from transformers import Trainer

    run_epoch = Trainer._run_epoch
    training_step = Trainer.training_step
    num_items = Trainer._get_num_items_in_batch
    save_checkpoint = Trainer._save_checkpoint
    send = adl.send_to_device

    @functools.wraps(run_epoch)
    def _run_epoch(self, *args, **kwargs):
        if not _real_cuda_trainer(self):
            return run_epoch(self, *args, **kwargs)
        from transformers.utils import is_torch_xla_available

        # transformers 5.16: _run_epoch(model, epoch, train_dataloader, ...), called by keyword
        train_dataloader = kwargs.get("train_dataloader", args[2] if len(args) > 2 else None)
        undo = None
        if _STATE["prefetch"] or _STATE["plan"]:
            undo = _install_iterator(self, train_dataloader)
        flipped = (_STATE["nan_gpu"] and bool(getattr(self.args, "logging_nan_inf_filter", False))
                   and not is_torch_xla_available())
        if flipped:
            self.args.logging_nan_inf_filter = False
            self.__dict__["_soup_nan_gpu"] = True
        restore_timing = _start_timing(self) if _STATE["timing"] else None
        single = _STATE["bwd_single"] and _single_thread_backward(self.accelerator)
        try:
            return run_epoch(self, *args, **kwargs)
        finally:
            if single:
                self.accelerator.__dict__.pop("backward", None)
            if flipped:
                self.args.logging_nan_inf_filter = True
                self.__dict__.pop("_soup_nan_gpu", None)
            if undo is not None:
                undo()
            from .trainer_patch import flush_deferred

            flush_deferred(self)
            _report(self, flipped, undo is not None, single)
            if restore_timing is not None:
                restore_timing()

    @functools.wraps(training_step)
    def _training_step(self, model, inputs, *args, **kwargs):
        loss = training_step(self, model, inputs, *args, **kwargs)
        if self.__dict__.get("_soup_nan_gpu"):
            loss = gpu_filtered_loss(self, loss)
        prefetcher = self.__dict__.get("_soup_prefetcher")
        if prefetcher is not None and getattr(model, "training", False):
            prefetcher.prefetch()  # the backward is queued: the CPU would only wait for the GPU now
        return loss

    @functools.wraps(num_items)
    def _get_num_items_in_batch(self, batch_samples, device):
        result = num_items(self, batch_samples, device)
        if (_STATE["plan"] and torch.is_tensor(result) and result.dim() == 0 and result.is_cuda
                and getattr(self.args, "world_size", 1) == 1 and getattr(self.args, "n_gpu", 1) <= 1
                and getattr(self.accelerator, "parallelism_config", None) is None):
            plans = [_plan_for_labels(b.get("labels")) if isinstance(b, Mapping) else None
                     for b in batch_samples]
            if plans and all(p is not None for p in plans):
                shifts = getattr(self, "_loss_shifts_labels", None)
                if isinstance(shifts, bool):
                    remember(result, sum(p.items_shifted if shifts else p.items_all for p in plans))
        return result

    @functools.wraps(save_checkpoint)
    def _save_checkpoint(self, *args, **kwargs):
        if not self.__dict__.get("_soup_nan_gpu"):
            return save_checkpoint(self, *args, **kwargs)
        self.args.logging_nan_inf_filter = True  # the saved arguments are the run's own
        try:
            return save_checkpoint(self, *args, **kwargs)
        finally:
            self.args.logging_nan_inf_filter = False

    wrappers = (_run_epoch, _training_step, _get_num_items_in_batch, _save_checkpoint,
                functools.partial(_send_to_device, send))
    (Trainer._run_epoch, Trainer.training_step, Trainer._get_num_items_in_batch,
     Trainer._save_checkpoint, adl.send_to_device) = wrappers
    _STATE["patched"] = (run_epoch, training_step, num_items, save_checkpoint, send)
    _STATE["wrappers"] = wrappers


def unpatch_step_pipeline():
    """Puts the originals back (tests), unless another patch wraps them since."""
    if not _STATE["patched"]:
        return
    import accelerate.data_loader as adl
    from transformers import Trainer

    current = (Trainer._run_epoch, Trainer.training_step, Trainer._get_num_items_in_batch,
               Trainer._save_checkpoint, adl.send_to_device)
    if any(a is not b for a, b in zip(current, _STATE["wrappers"])):
        return
    (Trainer._run_epoch, Trainer.training_step, Trainer._get_num_items_in_batch,
     Trainer._save_checkpoint, adl.send_to_device) = _STATE["patched"]
    _STATE["patched"] = False


def _single_thread_backward(accelerator):
    """accelerator.backward on the calling thread for one epoch (an instance attribute); False when
    something else already put one there."""
    if accelerator is None or "backward" in vars(accelerator):
        return False
    backward = accelerator.backward

    @functools.wraps(backward)
    def single_thread_backward(*args, **kwargs):
        with torch.autograd.set_multithreading_enabled(False):
            return backward(*args, **kwargs)

    accelerator.backward = single_thread_backward
    return True


def _report(trainer, flipped, iterator, single=False):
    parts = []
    if single:
        parts.append(f"backward on the calling thread (allocated at the epoch end "
                     f"{torch.cuda.memory_allocated() / 2**20:.1f} MiB)")
    if flipped:
        parts.append("nan/inf filter on the GPU")
    if iterator:
        parts.append(f"{_STATS['prefetched']} batches prefetched "
                     f"(fetch {_STATS['fetch_s'] * 1e3:.0f} ms in all)")
        if _STATE["plan"]:
            parts.append(f"{_STATS['planned']} host plans ({_STATS['plan_s'] * 1e3:.0f} ms), "
                         f"{_STATS['plans_used']} used, {_STATS['known_used']} host counts read")
    if parts:
        print("[solution] step pipeline: " + "; ".join(parts), flush=True)
    if _TIMES:
        rows = sorted(_TIMES.items(), key=lambda item: -item[1][0])
        print("[solution] boundary timing (host ms total / calls): " + ", ".join(
            f"{label} {seconds * 1e3:.1f}/{calls}" for label, (seconds, calls) in rows), flush=True)
        _TIMES.clear()


# ---- diagnostics: where the host time between the GPU's work goes -----------------------------


def _start_timing(trainer):
    """Wraps the callback handler's events, the log, Soup's tracker and display for one epoch."""
    from transformers import Trainer
    from transformers.trainer_callback import CallbackHandler

    call_event = CallbackHandler.call_event
    log_save = Trainer._maybe_log_save_evaluate
    undo = []

    def timed_call_event(self, event, args, state, control, **kwargs):
        if event not in ("on_step_begin", "on_step_end", "on_log", "on_pre_optimizer_step",
                         "on_optimizer_step", "on_substep_end"):
            return call_event(self, event, args, state, control, **kwargs)
        for callback in self.callbacks:
            started = time.perf_counter()
            result = getattr(callback, event)(
                args, state, control, model=self.model, processing_class=self.processing_class,
                optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader, eval_dataloader=self.eval_dataloader,
                **kwargs)
            if result is not None:
                control = result
            _timed(f"{type(callback).__name__}.{event}", started)
            if event == "on_log":
                _wrap_soup_sinks(callback, undo)
        return control

    def timed_log_save(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return log_save(self, *args, **kwargs)
        finally:
            _timed("Trainer._maybe_log_save_evaluate", started)

    CallbackHandler.call_event = timed_call_event
    Trainer._maybe_log_save_evaluate = timed_log_save
    wrapped = []
    for name in ("floating_point_ops", "_clip_grad_norm", "_get_grad_norm", "training_step",
                 "get_batch_samples", "_get_learning_rate"):
        wrapped.append((Trainer, name, getattr(Trainer, name)))
    for owner, name, fn in wrapped:
        setattr(owner, name, _timed_method(f"{owner.__name__}.{name}", fn))
    from . import boundary

    builds = boundary._STATE["builds"]

    def restore():
        CallbackHandler.call_event = call_event
        Trainer._maybe_log_save_evaluate = log_save
        for owner, name, fn in wrapped:
            setattr(owner, name, fn)
        for fn in undo:
            fn()
        _TIMES["boundary.py list builds"] = [0.0, boundary._STATE["builds"] - builds]
        for (kind, name), count in boundary._STATE["bumps"].items():
            _TIMES[f"boundary.py dropped by {kind}.{name}"] = [0.0, count]

    return restore


def _timed_method(label, fn):
    @functools.wraps(fn)
    def timed(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _timed(label, started)

    return timed


def _wrap_soup_sinks(callback, undo):
    """Times Soup's tracker commit and live display (instance attributes, removed at the end)."""
    for owner_name, method in (("tracker", "log_metrics"), ("display", "update")):
        owner = getattr(callback, owner_name, None)
        if owner is None or method in vars(owner) or not callable(getattr(owner, method, None)):
            continue
        bound = getattr(owner, method)
        label = f"{type(owner).__name__}.{method}"

        def timed(*args, _bound=bound, _label=label, **kwargs):
            started = time.perf_counter()
            try:
                return _bound(*args, **kwargs)
            finally:
                _timed(_label, started)

        setattr(owner, method, timed)
        undo.append(lambda owner=owner, method=method: owner.__dict__.pop(method, None))
