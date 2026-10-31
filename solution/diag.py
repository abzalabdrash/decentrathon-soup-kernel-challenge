"""Diagnostics of a measured run, off unless SOUP_SOLUTION_DIAG=1. Nothing here changes what
training computes; it only reads memory counters, records one step's allocations and traces two
steps, so that the numbers of a Kaggle run can be read offline:

- every phase mark (phases.py) also prints torch's allocated / reserved / peak-allocated MiB, and
  so do the entry and exit of peft's `prepare_model_for_kbit_training`;
- at the end of apply(): a PCIe probe, pinned host <-> GPU, 64 MB each way;
- a trainer callback: the peak allocated of every step, and
  - step SNAP_STEP (1-based, SOUP_SOLUTION_DIAG_SNAP_STEP, default 6): the allocator's history is
    recorded with stacks and written as `diag_memory_step<k>.pickle` (torch.cuda.memory._dump_snapshot;
    read it with tools/mem_snapshot_report.py or https://pytorch.org/memory_viz);
  - steps TRACE_STEPS (default 8 and 9): torch.profiler with CPU and CUDA activities, written as
    `diag_trace_steps8-9.json.gz` (a chrome trace: GPU idle gaps and what the CPU did meanwhile);
- at train begin and after the first step: nvidia-smi's memory.used next to torch's reserved, the
  difference being the CUDA context and the libraries.

Files go to SOUP_SOLUTION_DIAG_DIR, else the run directory (the parent of Soup's --config), else
the working directory.
"""

import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

SNAP_STEP = int(os.environ.get("SOUP_SOLUTION_DIAG_SNAP_STEP", "6"))
# "8,9" or a range "10-13" (the variants runner splits its environment on commas)
TRACE_STEPS = tuple(int(s) for s in os.environ.get("SOUP_SOLUTION_DIAG_TRACE_STEPS", "8,9")
                    .replace("-", ",").split(","))
# SOUP_SOLUTION_DIAG_TRACE_CPU=0: the trace records the CUDA activity only (kernels, copies and the CUDA
# runtime calls with their correlation ids); with the CPU activity the traced steps run CPU-bound under
# the profiler and the GPU's waits for copies hide behind it
TRACE_CPU = os.environ.get("SOUP_SOLUTION_DIAG_TRACE_CPU", "1") not in ("0", "false", "no", "")
# SOUP_SOLUTION_DIAG_STACK=1: the trace with Python stacks, and diag_stacks.log: the call stacks of the
# torch ops behind the small elementwise launches (copies, casts, binary ops), counted (task H2)
STACK = os.environ.get("SOUP_SOLUTION_DIAG_STACK", "0") not in ("0", "false", "no", "")
STACK_OPS = ("aten::copy_", "aten::_to_copy", "aten::contiguous", "aten::clone", "aten::add", "aten::add_",
             "aten::mul", "aten::mul_", "aten::div", "aten::div_", "aten::sub", "aten::where", "aten::fill_",
             "aten::zero_", "aten::cat", "aten::index_copy_", "aten::index_select", "aten::abs", "aten::amax",
             "aten::clamp", "aten::clamp_", "aten::floor", "aten::maximum", "aten::reciprocal", "aten::neg",
             "aten::sum", "aten::masked_fill_", "aten::addcmul_", "aten::lerp_", "aten::sqrt")
MAX_ENTRIES = 1_000_000
MIB = 2 ** 20


def enabled():
    return os.environ.get("SOUP_SOLUTION_DIAG", "0") not in ("0", "false", "no", "")


def out_dir():
    explicit = os.environ.get("SOUP_SOLUTION_DIAG_DIR")
    if explicit:
        path = Path(explicit)
    else:
        path = Path.cwd()
        argv = sys.argv
        if "--config" in argv and argv.index("--config") + 1 < len(argv):
            path = Path(argv[argv.index("--config") + 1]).resolve().parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def memory_line():
    """`alloc A MiB reserved R MiB peak P MiB` of the current device, or '' without CUDA."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return ""
    return (f"alloc {torch.cuda.memory_allocated() / MIB:.0f} MiB reserved "
            f"{torch.cuda.memory_reserved() / MIB:.0f} MiB peak {torch.cuda.max_memory_allocated() / MIB:.0f} MiB")


def log(text):
    print(f"[solution] diag {text}", flush=True)


def nvidia_smi_used():
    """memory.used of GPU 0 as nvidia-smi reports it (MiB), or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False).stdout
        return int(out.split()[0])
    except Exception:  # noqa: BLE001 - diagnostics only
        return None


def pcie_probe(megabytes=64, repeats=3):
    """Pinned host <-> GPU bandwidth of plain copies, GB/s each way (the lower bound streaming and
    offloading can count on); prints and returns (h2d, d2h)."""
    if not torch.cuda.is_available():
        return None
    n = megabytes * MIB
    host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    results = {}
    for name, src, dst in (("h2d", host, dev), ("d2h", dev, host)):
        dst.copy_(src, non_blocking=True)  # warm
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(repeats):
            dst.copy_(src, non_blocking=True)
        end.record()
        torch.cuda.synchronize()
        results[name] = repeats * n / (begin.elapsed_time(end) / 1e3) / 1e9
    del host, dev
    log(f"pcie pinned {megabytes} MB x{repeats}: host->GPU {results['h2d']:.2f} GB/s, "
        f"GPU->host {results['d2h']:.2f} GB/s")
    return results["h2d"], results["d2h"]


class _Trace:
    """torch.profiler around TRACE_STEPS, exported as a gzipped chrome trace."""

    def __init__(self, directory):
        self.directory, self.prof = directory, None

    def start(self):
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if TRACE_CPU else [ProfilerActivity.CUDA]
        self.prof = profile(activities=activities, record_shapes=STACK, with_stack=STACK and TRACE_CPU,
                            profile_memory=False)
        self.prof.start()
        log(f"trace started at step {TRACE_STEPS[0]}")

    def stop(self):
        if self.prof is None:
            return
        self.prof.stop()
        if STACK:
            try:
                self._stacks()
            except Exception as exc:  # noqa: BLE001 - a diagnostic must not break the run
                log(f"stack table failed: {exc!r}")
        name = f"diag_trace_steps{TRACE_STEPS[0]}-{TRACE_STEPS[-1]}.json"
        raw = self.directory / name
        self.prof.export_chrome_trace(str(raw))
        with open(raw, "rb") as src, gzip.open(str(raw) + ".gz", "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)
        raw.unlink()
        self.prof = None
        log(f"trace written: {raw}.gz")


    def _stacks(self):
        steps = max(len(TRACE_STEPS), 1)
        rows = []
        for evt in self.prof.key_averages(group_by_stack_n=12, group_by_input_shape=True):
            if evt.key not in STACK_OPS:
                continue
            stack = [f for f in (evt.stack or []) if "solution/" in f or "transformers/" in f or "peft/" in f
                     or "torch/nn/" in f or "torch/autograd/" in f][:8]
            gpu_us = getattr(evt, "device_time_total", 0) or getattr(evt, "cuda_time_total", 0)
            rows.append((evt.count / steps, gpu_us / steps, evt.key, str(evt.input_shapes)[:160], stack))
        rows.sort(key=lambda r: -r[0])
        path = self.directory / "diag_stacks.log"
        with open(path, "w") as out:
            out.write(f"torch ops per step (steps {TRACE_STEPS}), by call stack and input shapes\n")
            for count, gpu, key, shapes, stack in rows[:150]:
                out.write(f"\n{count:7.1f} per step, GPU {gpu / 1000:7.3f} ms  {key}  {shapes}\n")
                for frame in stack:
                    out.write(f"      {frame}\n")
        log(f"stack table written: {path}")


def make_callback():
    """The TrainerCallback (built lazily: transformers may not be imported where diag is set up)."""
    import transformers

    class DiagCallback(transformers.TrainerCallback):
        def __init__(self):
            self.directory = out_dir()
            self.step_peaks = []
            self.trace = _Trace(self.directory)
            self.recording = False
            self.started = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.perf_counter()
            used = nvidia_smi_used()
            log(f"train begin: {memory_line()}; nvidia-smi used {used} MiB")

        def on_step_begin(self, args, state, control, **kwargs):
            step = len(self.step_peaks) + 1
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if step == SNAP_STEP:
                try:
                    torch.cuda.memory._record_memory_history(max_entries=MAX_ENTRIES)
                    self.recording = True
                    log(f"memory history on for step {step}")
                except Exception as exc:  # noqa: BLE001
                    log(f"memory history unavailable: {exc!r}")
            if step == TRACE_STEPS[0]:
                try:
                    self.trace.start()
                except Exception as exc:  # noqa: BLE001
                    log(f"trace unavailable: {exc!r}")

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step = len(self.step_peaks) + 1
            peak = torch.cuda.max_memory_allocated() / MIB
            self.step_peaks.append(round(peak))
            if step == SNAP_STEP and self.recording:
                path = self.directory / f"diag_memory_step{step}.pickle"
                try:
                    torch.cuda.memory._dump_snapshot(str(path))
                    log(f"memory snapshot written: {path} ({path.stat().st_size / MIB:.1f} MiB); "
                        f"step peak {peak:.0f} MiB")
                except Exception as exc:  # noqa: BLE001
                    log(f"memory snapshot failed: {exc!r}")
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
                    self.recording = False
            if step == TRACE_STEPS[-1]:
                try:
                    self.trace.stop()
                except Exception as exc:  # noqa: BLE001
                    log(f"trace export failed: {exc!r}")
            if step == 1:
                used = nvidia_smi_used()
                log(f"after step 1: {memory_line()}; nvidia-smi used {used} MiB; step peak {peak:.0f} MiB")

        def on_train_end(self, args, state, control, **kwargs):
            log(f"train end: {memory_line()}; max reserved {torch.cuda.max_memory_reserved() / MIB:.0f} MiB")
            log(f"per-step peak allocated MiB: {self.step_peaks}")

    return DiagCallback()


def patch_soup_diag():
    """Installs the hooks above on a real CUDA trainer; Soup's tests with fakes are untouched."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_diag", False):
        return
    original_train = cls.train

    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        try:
            from transformers import Trainer

            real = isinstance(trainer, Trainer) and torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            real = False
        if real:
            try:
                trainer.add_callback(make_callback())
            except Exception as exc:  # noqa: BLE001
                log(f"callback not installed: {exc!r}")
        return original_train(self, *args, **kwargs)

    cls.train = train
    cls._soup_diag = True

    try:
        import peft

        original_prepare = peft.prepare_model_for_kbit_training

        def prepare_model_for_kbit_training(model, *args, **kwargs):
            log(f"before prepare_model_for_kbit_training: {memory_line()}")
            result = original_prepare(model, *args, **kwargs)
            log(f"after prepare_model_for_kbit_training: {memory_line()}")
            return result

        peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training
    except Exception as exc:  # noqa: BLE001
        log(f"peft hook not installed: {exc!r}")
    if torch.cuda.is_available():
        try:
            pcie_probe()
        except Exception as exc:  # noqa: BLE001
            log(f"pcie probe failed: {exc!r}")
