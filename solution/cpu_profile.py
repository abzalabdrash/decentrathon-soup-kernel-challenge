"""Diagnostics of the CPU side of the training steps: SOUP_SOLUTION_CPU_PROFILE=1 (off by default;
nothing in this file runs otherwise). Lines in the training log start with `[solution] cpu-profile`.

Steps are 0-based training steps; SOUP_SOLUTION_CPU_PROFILE_STEPS=a-b (default 5-12) is split in two
halves:
- every step of the run: its packed token count (the first decoder layer's input), the wall time
  from step begin to step end (both after a device synchronize), the issue time (step begin to the
  end of the optimizer step, before the synchronize: how long the CPU took to queue the step) and
  the main thread's CPU time; a step whose issue time is close to its wall time is CPU-bound;
- first half: cProfile (deterministic: exact call counts; every Python call costs ~1 us more, so
  times are inflated, shares are what to read): the functions by self and by inclusive time per
  step, self time by package (solution, torch's Python and its C ops, transformers, peft,
  bitsandbytes, trl, triton, accelerate, the harness), and counters per step: Triton launches, CUDA
  events made / recorded / waited for, bitsandbytes' Params4bit.__torch_function__, autograd
  Functions, saved-tensor hooks;
- second half: a sampler thread reads the main thread's stack every INTERVAL s (torch releases the
  GIL around every op, so the ticks land between ops) and charges the time since its previous tick
  to it: self time by the innermost frame and its package, inclusive time by function, and the time
  inside each function of solution/ (the innermost frame of ours on the stack);
- the step after the window: torch.cuda.set_sync_debug_mode("warn"), every synchronizing CUDA call
  (`.item()`, `.tolist()`, `nonzero`, blocking copies, synchronize) with its Python call site.
"""

import cProfile
import collections
import os
import pstats
import sys
import threading
import time
import traceback
import warnings

ENV = "SOUP_SOLUTION_CPU_PROFILE"
INTERVAL = 0.001
MAX_DEPTH = 200
TOP = 45
PACKAGES = ("transformers", "peft", "bitsandbytes", "trl", "triton", "accelerate", "soup_cli",
            "datasets", "torch")


def enabled():
    return os.environ.get(ENV, "0").strip().lower() not in ("0", "false", "no", "")


def _window():
    text = os.environ.get(ENV + "_STEPS", "5-12").strip()
    try:
        first, last = (int(x) for x in text.split("-"))
    except ValueError:
        first, last = 5, 12
    last = max(last, first + 1)
    middle = first + (last - first + 1) // 2
    return first, middle, last  # cProfile first..middle-1, sampler middle..last


def _log(text):
    print(f"[solution] cpu-profile {text}", flush=True)


def _package(filename, name=""):
    f = (filename or "").replace("\\", "/")
    if f == "~" or not f:
        return "torch C" if "torch" in name else "builtin C"
    if "/solution/" in f:
        return "solution"
    if "/harness/" in f:
        return "harness"
    for package in PACKAGES:
        if f"/{package}/" in f:
            return package
    return "python/other"


def _short(filename):
    f = (filename or "").replace("\\", "/")
    for marker in ("/site-packages/", "/solution/", "/harness/", "/lib/python"):
        i = f.rfind(marker)
        if i >= 0:
            return f[i + 1:] if marker in ("/solution/", "/harness/") else f[i + len(marker):]
    return f.rsplit("/", 2)[-1] if "/" in f else f


# counters read from cProfile's call counts: label -> predicate on (file, name)
_COUNTERS = (
    ("Triton launches (JITFunction.run)", lambda f, n: n == "run" and ("/triton/runtime/jit" in f
                                                                    or "/harness/" in f)),
    ("CUDA Event made", lambda f, n: n == "__new__" and f.endswith("torch/cuda/streams.py")),
    ("CUDA Event.record", lambda f, n: n == "record" and f.endswith("torch/cuda/streams.py")),
    ("CUDA Event.query", lambda f, n: n == "query" and f.endswith("torch/cuda/streams.py")),
    ("CUDA Event.elapsed_time", lambda f, n: n == "elapsed_time" and f.endswith("torch/cuda/streams.py")),
    ("Stream.wait_event", lambda f, n: n == "wait_event" and f.endswith("torch/cuda/streams.py")),
    ("Stream.record_event", lambda f, n: n == "record_event" and f.endswith("torch/cuda/streams.py")),
    ("torch.cuda.stream() contexts", lambda f, n: n == "__enter__" and f.endswith("torch/cuda/__init__.py")),
    ("current_stream", lambda f, n: n == "current_stream" and f.endswith("torch/cuda/__init__.py")),
    ("Params4bit.__torch_function__", lambda f, n: n == "__torch_function__" and "bitsandbytes" in f),
    ("any __torch_function__", lambda f, n: n == "__torch_function__"),
    ("autograd Function.apply", lambda f, n: n == "apply" and f.endswith("torch/autograd/function.py")),
    ("offload pack", lambda f, n: n == "pack" and f.endswith("solution/offload.py")),
    ("offload unpack", lambda f, n: n == "unpack" and f.endswith("solution/offload.py")),
    ("Module._call_impl", lambda f, n: n == "_call_impl"),
    ("item()", lambda f, n: "method 'item'" in n),
    ("tolist()", lambda f, n: "method 'tolist'" in n),
    ("synchronize", lambda f, n: n == "synchronize" or "synchronize" in n),
)


def _report_cprofile(profile, steps):
    stats = pstats.Stats(profile)
    rows = []  # (tottime, cumtime, calls, file, line, name)
    for (filename, line, name), (cc, nc, tt, ct, _callers) in stats.stats.items():
        rows.append((tt, ct, nc, filename, line, name))
    total = sum(r[0] for r in rows)
    _log(f"cProfile over {steps} steps: {total / steps * 1e3:.1f} ms self time per step in "
         f"{sum(r[2] for r in rows) / steps:.0f} calls per step (inflated by the profiler)")
    by_package = collections.Counter()
    calls_by_package = collections.Counter()
    for tt, _ct, nc, filename, _line, name in rows:
        by_package[_package(filename, name)] += tt
        calls_by_package[_package(filename, name)] += nc
    for package, tt in by_package.most_common():
        _log(f"  self by package: {package:14s} {tt / steps * 1e3:8.2f} ms/step "
             f"{calls_by_package[package] / steps:9.0f} calls/step")
    for label, test in _COUNTERS:
        n = sum(nc for _tt, _ct, nc, f, _l, name in rows if test((f or "").replace("\\", "/"), name))
        _log(f"  count per step: {label:38s} {n / steps:8.1f}")
    _log("  top self (ms/step, calls/step, function):")
    for tt, _ct, nc, filename, line, name in sorted(rows, key=lambda r: -r[0])[:TOP * 2]:
        _log(f"    self {tt / steps * 1e3:8.2f} {nc / steps:8.0f}  {_package(filename, name):12s} "
             f"{_short(filename)}:{line} {name}")
    _log("  top inclusive (ms/step, calls/step, function):")
    for _tt, ct, nc, filename, line, name in sorted(rows, key=lambda r: -r[1])[:TOP * 2]:
        _log(f"    incl {ct / steps * 1e3:8.2f} {nc / steps:8.0f}  {_package(filename, name):12s} "
             f"{_short(filename)}:{line} {name}")
    ours = [r for r in rows if _package(r[3], r[5]) == "solution"]
    _log("  solution/ functions by inclusive time (ms/step, self ms/step, calls/step):")
    for tt, ct, nc, filename, line, name in sorted(ours, key=lambda r: -r[1])[:TOP]:
        _log(f"    ours {ct / steps * 1e3:8.2f} {tt / steps * 1e3:8.2f} {nc / steps:7.0f}  "
             f"{_short(filename)}:{line} {name}")


class _Sampler:
    """The main thread's stack every INTERVAL s while `active` (see the module doc)."""

    def __init__(self, ident):
        self.ident = ident
        self.active = False
        self.stop = threading.Event()
        self.codes = {}  # id(code) -> code (kept alive)
        self.self_t = collections.Counter()
        self.incl = collections.Counter()
        self.ours = collections.Counter()
        self.package_self = collections.Counter()
        self.ticks = 0
        self.seconds = 0.0
        self.thread = threading.Thread(target=self._run, name="soup-cpu-profile", daemon=True)
        self.thread.start()

    def _run(self):
        last = time.perf_counter()
        while not self.stop.is_set():
            time.sleep(INTERVAL)
            frame = sys._current_frames().get(self.ident)
            now = time.perf_counter()
            dt, last = now - last, now
            if not self.active or frame is None:
                continue
            self.ticks += 1
            self.seconds += dt
            seen = set()
            leaf = None
            mine = None
            depth = 0
            while frame is not None and depth < MAX_DEPTH:
                code = frame.f_code
                key = id(code)
                if key not in self.codes:
                    self.codes[key] = code
                if leaf is None:
                    leaf = key
                if mine is None and "solution" in code.co_filename and (
                        "/solution/" in code.co_filename.replace("\\", "/")):
                    mine = key
                seen.add(key)
                frame = frame.f_back
                depth += 1
            del frame
            self.self_t[leaf] += dt
            self.package_self[_package(self.codes[leaf].co_filename)] += dt
            for key in seen:
                self.incl[key] += dt
            self.ours[mine] += dt

    def _name(self, key):
        if key is None:
            return "(outside solution/)"
        code = self.codes[key]
        return f"{_package(code.co_filename):12s} {_short(code.co_filename)}:{code.co_firstlineno} {code.co_name}"

    def report(self, steps):
        self.stop.set()
        _log(f"sampler over {steps} steps: {self.ticks} ticks, {self.seconds / steps * 1e3:.1f} ms per "
             f"step sampled ({self.seconds / max(self.ticks, 1) * 1e3:.2f} ms per tick)")
        for package, t in self.package_self.most_common():
            _log(f"  sampled self by package of the innermost Python frame: {package:14s} "
                 f"{t / steps * 1e3:8.2f} ms/step")
        _log("  sampled self (ms/step, innermost Python frame; includes the C calls it makes):")
        for key, t in self.self_t.most_common(TOP):
            _log(f"    self {t / steps * 1e3:8.2f}  {self._name(key)}")
        _log("  sampled inclusive (ms/step):")
        for key, t in self.incl.most_common(TOP * 2):
            _log(f"    incl {t / steps * 1e3:8.2f}  {self._name(key)}")
        _log("  sampled time by the innermost solution/ function on the stack (ms/step):")
        for key, t in self.ours.most_common(TOP):
            _log(f"    ours {t / steps * 1e3:8.2f}  {self._name(key)}")


class _SyncWatch:
    """torch.cuda.set_sync_debug_mode('warn') for one step; the call sites of the warnings."""

    def __init__(self):
        self.sites = collections.Counter()
        self.kinds = collections.Counter()
        self._cw = None

    def start(self):
        import torch

        self._cw = warnings.catch_warnings()
        self._cw.__enter__()
        warnings.simplefilter("always")
        warnings.showwarning = self._show
        torch.cuda.set_sync_debug_mode("warn")

    def _show(self, message, category, filename, lineno, file=None, line=None):
        stack = traceback.extract_stack()[:-1]
        frames = [f for f in stack if "/warnings.py" not in f.filename.replace("\\", "/")
                  and not f.filename.replace("\\", "/").endswith("solution/cpu_profile.py")]
        site = " <- ".join(f"{_short(f.filename)}:{f.lineno} {f.name}" for f in reversed(frames[-4:]))
        self.sites[site] += 1
        self.kinds[str(message).split(".")[0][:90]] += 1

    def stop(self, step):
        import torch

        torch.cuda.set_sync_debug_mode("default")
        if self._cw is not None:
            self._cw.__exit__(None, None, None)
            self._cw = None
        _log(f"synchronizing CUDA calls in step {step}: {sum(self.sites.values())}")
        for kind, n in self.kinds.most_common():
            _log(f"  sync kind {n:4d}x {kind}")
        for site, n in self.sites.most_common(40):
            _log(f"  sync site {n:4d}x {site}")


def make_callback(trainer):
    import torch
    import transformers

    first, middle, last = _window()

    class CpuProfileCallback(transformers.TrainerCallback):
        def __init__(self):
            self.step = -1
            self.rows = []
            self.tokens = None
            self.profile = None
            self.sampler = None
            self.sync = None
            self.hook = None

        def _tokens_hook(self, module, args, kwargs=None):
            hidden = args[0] if args else (kwargs or {}).get("hidden_states")
            if torch.is_tensor(hidden) and hidden.dim() == 3 and torch.is_grad_enabled():
                self.tokens = int(hidden.shape[1])

        def on_train_begin(self, args, state, control, **kwargs):
            try:
                from .weight_stream import find_decoder_layers

                layers = find_decoder_layers(trainer.model)
                if layers:
                    self.hook = layers[0].register_forward_pre_hook(self._tokens_hook)
            except Exception as exc:  # noqa: BLE001 - diagnostics only
                _log(f"token hook not installed: {exc!r}")
            _log(f"window: cProfile steps {first}-{middle - 1}, sampler {middle}-{last}, "
                 f"sync watch step {last + 1}")

        def on_step_begin(self, args, state, control, **kwargs):
            self.step += 1
            torch.cuda.synchronize()
            step = self.step
            if step == first:
                self.profile = cProfile.Profile()
                self.profile.enable()
            elif step == middle:
                self.sampler = _Sampler(threading.get_ident())
                self.sampler.active = True
            elif step == last + 1:
                self.sync = _SyncWatch()
                self.sync.start()
            self.tokens = None
            self.cpu0 = time.thread_time()
            self.t0 = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            issued = time.perf_counter()
            cpu = time.thread_time() - self.cpu0
            step = self.step
            if step == middle - 1 and self.profile is not None:
                self.profile.disable()
            if step == last and self.sampler is not None:
                self.sampler.active = False
            torch.cuda.synchronize()
            end = time.perf_counter()
            self.rows.append((step, self.tokens, end - self.t0, issued - self.t0, cpu))
            if step == middle - 1 and self.profile is not None:
                _report_cprofile(self.profile, middle - first)
                self.profile = None
            if step == last and self.sampler is not None:
                self.sampler.report(last - middle + 1)
                self.sampler = None
            if step == last + 1 and self.sync is not None:
                self.sync.stop(step)
                self.sync = None

        def on_train_end(self, args, state, control, **kwargs):
            if self.hook is not None:
                self.hook.remove()
            _log("per step: step, tokens, wall ms, issue ms (CPU queued the step), main-thread CPU ms")
            for step, tokens, wall, issue, cpu in self.rows:
                tag = " profiled" if first <= step <= last + 1 else ""
                _log(f"  step {step:3d} {tokens if tokens is not None else '?':>5} {wall * 1e3:8.1f} "
                     f"{issue * 1e3:8.1f} {cpu * 1e3:8.1f}{tag}")
            plain = [r for r in self.rows if not (first <= r[0] <= last + 1) and r[0] >= 2]
            if plain:
                short = [r for r in plain if r[1] is not None and r[1] < 600]
                for label, rows in (("steps 2+ outside the window", plain), ("of them < 600 tokens", short)):
                    if rows:
                        n = len(rows)
                        _log(f"  {label}: {n} steps, mean wall {sum(r[2] for r in rows) / n * 1e3:.1f} ms, "
                             f"issue {sum(r[3] for r in rows) / n * 1e3:.1f} ms, "
                             f"CPU {sum(r[4] for r in rows) / n * 1e3:.1f} ms")

    return CpuProfileCallback()


def patch_soup_cpu_profile():
    """Installs the callback above, first in the trainer's list (its step end runs before the
    harness's synchronize), on a real CUDA trainer; Soup's tests with fakes are untouched."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_cpu_profile", False):
        return
    original_train = cls.train

    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        try:
            import torch
            from transformers import Trainer

            if isinstance(trainer, Trainer) and torch.cuda.is_available():
                trainer.callback_handler.callbacks.insert(0, make_callback(trainer))
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            _log(f"callback not installed: {exc!r}")
        return original_train(self, *args, **kwargs)

    cls.train = train
    cls._soup_cpu_profile = True
