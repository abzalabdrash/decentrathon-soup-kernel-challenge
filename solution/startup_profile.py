"""Diagnostics of the time before the first training step: SOUP_SOLUTION_STARTUP_PROFILE=1 (off by
default; nothing in this file runs otherwise).

What it prints to the training log (lines start with `[solution] startup-profile`):
- at every phase mark (phases.mark), the wall time since `apply begin` and, since the previous mark,
  the CPU time of the main thread and of the rest of the process (warm-up workers, triton's hash,
  pinning threads, the tokenizer's and torch's native threads), and how late the sampler's ticks ran;
- extra marks from the end of Soup's model setup to the end of the first step (`install_marks`):
  the tokenization of the training rows, TRL's SFTTrainer and transformers' Trainer __init__, the end
  of Soup's setup, the Trainer's train() and its inner loop up to on_train_begin, the first step;
- the os.stat cost of each sys.path entry, and the import system's stat calls per sys.path entry
  (count, time) from `apply begin` to the first step;
- a sampling profile from `apply begin` to the end of the first step: a daemon thread reads
  sys._current_frames() every INTERVAL s and charges the time since its previous tick to the Python
  stack of the main thread and of every other thread. At the end of training it prints, per window
  between the main marks, the functions with the most time, inclusive (on the stack) and self (the
  innermost Python frame: that includes the C calls it makes and any wait for the GIL after them).

Why sampling and not cProfile: since Python 3.12 cProfile runs on sys.monitoring, which is per
interpreter, so a profile enabled in the main thread also records the warm-up threads' calls, on one
stack; and it makes Python code about twice as slow, which changes the GIL contention the profile is
meant to show. The sampler needs the GIL for each tick, so how late a tick runs after its due time
measures how long a thread that wants the GIL waits for it (the `late` figures).
"""

import os
import sys
import threading
import time

ENV = "SOUP_SOLUTION_STARTUP_PROFILE"
INTERVAL = 0.004  # seconds between ticks
MAX_DEPTH = 160
# the windows of the report: (label, mark that opens it, mark that closes it)
WINDOWS = (
    ("apply()", "apply begin", "apply end"),
    ("Soup CLI up to the model setup", "apply end", "setup_transformers begin"),
    ("model setup", "setup_transformers begin", "setup_transformers end"),
    ("trainer build (rest of Soup's setup)", "setup_transformers end", "setup end"),
    ("Soup CLI up to train (incl. the warm-up wait)", "setup end", "train begin"),
    ("train() up to the first step", "train begin", "first step begin"),
    ("first step", "first step begin", "first step end"),
)
TOP_INCLUSIVE, TOP_SELF, TOP_WARMUP, TOP_OTHER = 40, 20, 16, 6

_STATE = {"sampler": None, "marks": [], "reported": False, "installed": False, "t0": None}


def enabled():
    return os.environ.get(ENV, "0").strip().lower() not in ("0", "false", "no", "")


def _log(text):
    print(f"[solution] startup-profile {text}", flush=True)


def _cpu_of(ident):
    try:
        return time.clock_gettime(time.pthread_getcpuclockid(ident))
    except Exception:  # noqa: BLE001 - not Linux, or the thread just ended
        return None


def _kind(thread, main_ident):
    if thread is None:
        return "other:?"
    if thread.ident == main_ident:
        return "main"
    if thread.name.startswith("solution-warmup"):
        return "warmup"
    return "other:" + thread.name


class _Sampler(threading.Thread):
    def __init__(self, interval):
        super().__init__(name="solution-startup-profile", daemon=True)
        self.interval = interval
        self.main = threading.main_thread().ident
        self.halt = threading.Event()
        self.samples = []  # (t, kind, stack id, weight seconds)
        self.ticks = []  # (t, late seconds)
        self.stacks = {}  # (code ids, leaf line) -> id
        self.stack_list = []
        self.codes = {}  # id -> code object
        self.warmup_seen = None
        self.warmup_gone = None

    def _stack_id(self, frame):
        # keyed by the ids of the code objects: hashing a code object hashes its bytecode and
        # constants every time, which made the first version of this sampler slow the main thread
        # down by a third during the imports. The code objects are kept in `codes` (alive).
        codes = []
        line = frame.f_lineno
        while frame is not None and len(codes) < MAX_DEPTH:
            code = frame.f_code
            self.codes.setdefault(id(code), code)
            codes.append(id(code))
            frame = frame.f_back
        key = (tuple(codes), line)
        found = self.stacks.get(key)
        if found is None:
            found = self.stacks[key] = len(self.stack_list)
            self.stack_list.append(key)
        return found

    def run(self):
        me = threading.get_ident()
        due = previous = time.perf_counter()
        while not self.halt.is_set():
            due += self.interval
            delay = due - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            now = time.perf_counter()
            self.ticks.append((now, now - due))
            weight = now - previous
            previous = now
            if now - due > 20 * self.interval:
                due = now  # do not catch up with a burst of ticks after a long stall
            try:
                frames = sys._current_frames()
                threads = {t.ident: t for t in threading.enumerate()}
                warmup_alive = False
                for ident, frame in frames.items():
                    if ident == me:
                        continue
                    kind = _kind(threads.get(ident), self.main)
                    if kind == "warmup":
                        warmup_alive = True
                    self.samples.append((now, kind, self._stack_id(frame), weight))
                del frames
                if warmup_alive and self.warmup_seen is None:
                    self.warmup_seen = now
                if not warmup_alive and self.warmup_seen is not None and self.warmup_gone is None:
                    self.warmup_gone = now
            except Exception:  # noqa: BLE001 - diagnostics must not break the run
                pass


def _name(code):
    path = code.co_filename
    for marker in ("dist-packages/", "site-packages/"):
        i = path.rfind(marker)
        if i >= 0:
            path = path[i + len(marker):]
            break
    else:
        i = path.rfind("/solution/")
        if i >= 0:
            path = path[i + 1:]
    return f"{path}:{code.co_firstlineno}({getattr(code, 'co_qualname', code.co_name)})"


# ---------------------------------------------------------------- the import system's stat calls

_STATS = {"installed": False, "calls": {}, "original": None, "active": True}


def _stat_root(path):
    """The sys.path entry (or other top directory) a stat'ed path belongs to."""
    best = ""
    for entry in sys.path:
        if entry and path.startswith(entry) and len(entry) > len(best):
            best = entry
    return best or "/".join(path.split("/")[:3])


def _install_stat_counter():
    """Counts and times importlib's `_path_stat` (the os.stat behind every module lookup) per
    sys.path entry, from `apply begin` to the end of the first step."""
    try:
        from importlib import _bootstrap_external as external
    except Exception:  # noqa: BLE001
        return
    original = getattr(external, "_path_stat", None)
    if original is None or _STATS["installed"]:
        return
    calls = _STATS["calls"]
    clock = time.perf_counter

    def count(path, ok, elapsed):
        key = (_stat_root(str(path)), ok)
        found = calls.get(key)
        calls[key] = (1, elapsed) if found is None else (found[0] + 1, found[1] + elapsed)

    def _path_stat(path):
        if not _STATS["active"]:
            return original(path)
        start = clock()
        try:
            result = original(path)
        except OSError:
            count(path, False, clock() - start)
            raise
        count(path, True, clock() - start)
        return result

    external._path_stat = _path_stat
    _STATS.update(installed=True, original=original)


def _path_entry_timings():
    """os.stat of each sys.path entry and of a missing name in it, 50 times each (microseconds)."""
    out = []
    for entry in sys.path:
        if not entry:
            continue
        timings = []
        for target in (entry, os.path.join(entry, "__soup_missing__")):
            start = time.perf_counter()
            for _ in range(50):
                try:
                    os.stat(target)
                except OSError:
                    pass
            timings.append(1e6 * (time.perf_counter() - start) / 50)
        out.append(f"{entry} {timings[0]:.0f}/{timings[1]:.0f}")
    return out


def _report_stats():
    calls = _STATS["calls"]
    if not calls:
        return
    total_n = sum(n for n, _ in calls.values())
    total_s = sum(t for _, t in calls.values())
    _log(f"import system os.stat calls from apply begin to the first step: {total_n} calls, {total_s:.3f} s "
         f"(in the stat call itself, the GIL released)")
    for (root, ok), (n, seconds) in sorted(calls.items(), key=lambda kv: -kv[1][1])[:12]:
        _log(f"  stat {'found' if ok else 'missing'} under {root or '?'}: {n} calls, {seconds:.3f} s, "
             f"{1e6 * seconds / n:.0f} us each")


def on_mark(name):
    """Called by phases.mark for every mark: starts the sampler at `apply begin`, prints the CPU
    split since the previous mark, and the report at `train end` (or `exit`)."""
    if not enabled():
        return
    try:
        _on_mark(name)
    except Exception as exc:  # noqa: BLE001
        _log(f"failed at {name}: {exc!r}")


def _on_mark(name):
    now = time.perf_counter()
    main = threading.main_thread().ident
    if _STATE["t0"] is None:
        _STATE["t0"] = now
        sampler = _Sampler(INTERVAL)
        _STATE["sampler"] = sampler
        sampler.start()
        _log(f"sampling every {INTERVAL * 1000:.0f} ms from `{name}`; cpus {os.cpu_count()}, "
             f"affinity {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else '?'}, "
             f"switch interval {sys.getswitchinterval() * 1000:.1f} ms")
        try:
            _log("os.stat of each sys.path entry / of a missing name in it, us: "
                 + "; ".join(_path_entry_timings()))
            _install_stat_counter()
        except Exception as exc:  # noqa: BLE001
            _log(f"stat diagnostics failed: {exc!r}")
    record = {"name": name, "t": now, "main_cpu": _cpu_of(main), "proc_cpu": time.process_time(),
              "threads": {t.name: _cpu_of(t.ident) for t in threading.enumerate()
                          if t.ident != main and not t.name.startswith("solution-startup-profile")}}
    marks = _STATE["marks"]
    previous = marks[-1] if marks else None
    marks.append(record)
    t0 = _STATE["t0"]
    text = f"mark {name} +{now - t0:.3f} s"
    if previous is not None:
        wall = now - previous["t"]
        main_cpu = (record["main_cpu"] or 0) - (previous["main_cpu"] or 0)
        other = record["proc_cpu"] - previous["proc_cpu"] - main_cpu
        late = _late_between(previous["t"], now)
        text += (f" (+{wall:.3f} since {previous['name']}: main cpu {main_cpu:.3f} s "
                 f"{100 * main_cpu / wall if wall > 0 else 0:.0f}%, other threads cpu {other:.3f} s"
                 f"{late})")
        busy = []
        for thread, cpu in record["threads"].items():
            before = previous["threads"].get(thread, 0.0) or 0.0
            if cpu is not None and cpu - before >= 0.02:
                busy.append(f"{thread} {cpu - before:.2f}")
        if busy:
            text += "; threads: " + ", ".join(busy)
    _log(text)
    if name == "first step end":
        sampler = _STATE["sampler"]
        if sampler is not None:
            sampler.halt.set()
        _STATS["active"] = False
    if name in ("train end", "exit") and not _STATE["reported"]:
        _STATE["reported"] = True
        report()


def _late_between(start, end):
    sampler = _STATE["sampler"]
    if sampler is None:
        return ""
    late = sorted(x for t, x in list(sampler.ticks) if start <= t < end)
    if not late:
        return ""
    mean = sum(late) / len(late)
    return (f"; sampler ticks late mean {1000 * mean:.2f} ms, p90 {1000 * late[int(0.9 * len(late))]:.2f}"
            f" ms, max {1000 * late[-1]:.1f} ms")


def mark(name):
    """A mark of its own (only with the switch on), through phases.mark so it is in the log."""
    if enabled():
        from .phases import mark as phase_mark

        phase_mark(name)


def report():
    """The sampling profile per window (see the module docstring)."""
    sampler = _STATE["sampler"]
    if sampler is None:
        return
    sampler.halt.set()
    _STATS["active"] = False
    try:
        _report_stats()
    except Exception as exc:  # noqa: BLE001
        _log(f"stat report failed: {exc!r}")
    t0 = _STATE["t0"]
    times = {}
    for record in _STATE["marks"]:
        times.setdefault(record["name"], record["t"])
    if sampler.warmup_seen is not None:
        gone = (f"+{sampler.warmup_gone - t0:.2f} s" if sampler.warmup_gone is not None
                else "not before the sampler stopped")
        _log(f"warm-up threads seen from +{sampler.warmup_seen - t0:.2f} s, all ended {gone}")
    samples = list(sampler.samples)
    stacks = sampler.stack_list
    codes = sampler.codes
    for label, begin, end in WINDOWS:
        if begin not in times or end not in times:
            continue
        a, b = times[begin], times[end]
        chosen = [s for s in samples if a <= s[0] < b]
        _log(f"window {label}: +{a - t0:.2f} -> +{b - t0:.2f} s ({b - a:.2f} s wall){_late_between(a, b)}")
        kinds = {}
        for t, kind, sid, weight in chosen:
            kinds.setdefault(kind, []).append((sid, weight))
        main = kinds.pop("main", [])
        _print_top(f"{label} | main", main, stacks, codes, b - a, TOP_INCLUSIVE, TOP_SELF)
        warm = kinds.pop("warmup", [])
        if warm:
            _print_top(f"{label} | warm-up threads", warm, stacks, codes, b - a, TOP_WARMUP, 0)
        for kind, rows in sorted(kinds.items(), key=lambda kv: -sum(w for _, w in kv[1])):
            if sum(w for _, w in rows) >= 0.05:
                _print_top(f"{label} | {kind[6:]}", rows, stacks, codes, b - a, TOP_OTHER, 0)


_PLUMBING = ("<frozen importlib._bootstrap", "threading.py", "typer/", "click/", "harness/_train.py",
             "soup_cli/cli.py", "soup_cli/commands/train.py", "<string>", "runpy.py")


def _print_top(label, rows, stacks, codes, wall, top_inclusive, top_self):
    """Inclusive and self time per function; frames that every sample of the window shares and the
    import / CLI plumbing are left out of the inclusive list (they say nothing)."""
    total = sum(w for _, w in rows)
    if not rows:
        return
    inclusive, own = {}, {}
    for sid, weight in rows:
        ids, line = stacks[sid]
        for code_id in set(ids):
            inclusive[code_id] = inclusive.get(code_id, 0.0) + weight
        key = (ids[0], line)
        own[key] = own.get(key, 0.0) + weight
    _log(f"  {label}: {total:.2f} thread-s sampled")
    shown = 0
    for code_id, seconds in sorted(inclusive.items(), key=lambda kv: -kv[1]):
        name = _name(codes[code_id])
        if seconds >= 0.995 * total and total > 0.05 or any(p in name for p in _PLUMBING):
            continue
        _log(f"  {label} incl {seconds:6.3f} s {100 * seconds / wall if wall else 0:4.0f}%  {name}")
        shown += 1
        if shown >= top_inclusive:
            break
    for (code_id, line), seconds in sorted(own.items(), key=lambda kv: -kv[1])[:top_self]:
        _log(f"  {label} self {seconds:6.3f} s {100 * seconds / wall if wall else 0:4.0f}%  "
             f"{_name(codes[code_id])} line {line}")


# ---------------------------------------------------------------- marks inside the trainer build

def _wrap(owner, attribute, label, end_name=None):
    original = getattr(owner, attribute, None)
    if original is None or getattr(original, "_soup_startup_mark", False):
        return

    def wrapper(*args, **kwargs):
        mark(f"{label} begin")
        try:
            return original(*args, **kwargs)
        finally:
            mark(end_name or f"{label} end")

    wrapper._soup_startup_mark = True
    wrapper.__wrapped__ = original
    for name in ("__name__", "__qualname__", "__doc__"):
        try:
            setattr(wrapper, name, getattr(original, name))
        except (AttributeError, TypeError):
            pass
    setattr(owner, attribute, wrapper)


def install_marks():
    """The extra marks of the module docstring (only with the switch on)."""
    if not enabled() or _STATE["installed"]:
        return
    _STATE["installed"] = True
    try:
        import transformers
        import trl
        from soup_cli.trainer import sft as soup_sft
        from soup_cli.utils import peft_wiring

        _wrap(soup_sft, "_map_text_sft_rows", "tokenize rows")
        _wrap(soup_sft, "_ensure_assistant_masks", "assistant masks")
        _wrap(soup_sft.SFTTrainerWrapper, "_build_rewind_trainer", "rewind trainer")
        _wrap(soup_sft.SFTTrainerWrapper, "setup", "setup", end_name="setup end")
        _wrap(trl.SFTTrainer, "__init__", "SFTTrainer init")
        _wrap(trl.SFTTrainer, "_prepare_dataset", "TRL prepare dataset")
        _wrap(transformers.Trainer, "__init__", "Trainer init")
        _wrap(peft_wiring, "attach_plugin_callback", "plugin callback")
        _wrap(transformers.Trainer, "_inner_training_loop", "inner loop")

        original_train = transformers.Trainer.train

        class _Marks(transformers.TrainerCallback):
            def __init__(self):
                self.steps = 0

            def on_train_begin(self, args, state, control, **kwargs):
                mark("on_train_begin")

            def on_step_end(self, args, state, control, **kwargs):
                self.steps += 1
                if self.steps == 1:
                    import torch

                    torch.cuda.synchronize()
                    mark("first step end")

        def train(self, *args, **kwargs):
            mark("Trainer.train begin")
            try:
                self.add_callback(_Marks())
            except Exception:  # noqa: BLE001
                pass
            return original_train(self, *args, **kwargs)

        train._soup_startup_mark = True
        transformers.Trainer.train = train
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        _log(f"marks not installed: {exc!r}")
