"""The run's pinned host memory handed back while Soup saves, not at the process exit
(SOUP_SOLUTION_EXIT_RELEASE=1, or =all to also hand back the caching allocator's free GPU blocks;
SOUP_SOLUTION_EXIT_NOGC=1 also stops the garbage collector there).

The harness times the whole process, its exit included. Measured on the T4 (solution/bench_exit.py,
soup-e1-exit1, two runs each), from Trainer.train's end to the process's end: base 2.39 s, ours 1.75 s,
of which the kernel's teardown of the exited process 0.81 s against 0.32 s for the base, the atexit
callbacks 0.09 s (weakref.finalize unregistering the registered buffers: 69 ms) and the
finalization's full garbage collection 46 ms. Our process holds ~3.6 GB of page-locked host memory
at the end: the offload's slabs (2,560 MiB from cudaHostAlloc, prepinned), the layers' 4-bit weights
(645 MiB) and the fp16 head (445 MiB) registered with cudaHostRegister (head_pack.py's 334 MiB more).
The driver gives it back page by page when the process ends, on the clock and with nothing else to
overlap.

None of it is needed once the last optimizer step is done. So at the last step's on_step_end (the
harness's callback, added before ours, has synchronized the device; at the end of train() instead
when the trainer evaluates, which runs the model after the last step) a thread, not a daemon:
1. frees the offload's slabs, spare and used (offload.release_host_memory: cudaFreeHost through
   torch's host allocator; an offload that runs again pins new slabs on demand);
2. unregisters every registered buffer (weight_stream.unpin_all): the values stay where they are as
   pageable memory, so anything that runs the model afterwards (eval, generation) still reads the
   right values, only through slower copies; their finalizers are spent, nothing unregisters at exit.
The main thread meanwhile saves the last checkpoint, Soup saves the adapter and prints its summary,
the harness writes its metrics: the release runs under that work instead of after it. The thread is
joined by the interpreter at exit (threading._shutdown), so nothing CUDA runs while it finalizes.
With SOUP_SOLUTION_EXIT_NOGC=1 the garbage collector is disabled at the same point: the training is
over, and Py_FinalizeEx's own full collection is skipped (PyGC_Collect returns at once when the
collector is disabled); the module teardown and its own collection run as always, every file is
already flushed and closed by then. No output is skipped and nothing is cut short: the adapter, the
checkpoint, the rewind log, the experiment database and the logs are written exactly as before.

T4 (soup-e1-exit2/3, bench_exit.py, 2 runs each): train end -> process end 1.84 s -> 1.27 s with the
release (kernel teardown 0.88 -> 0.36 s, the base's is 0.32), 1.22 s with NOGC, 1.08 s with =all and
NOGC (another session); the thread takes 0.45-0.53 s and is done before main() returns.
"""

import atexit
import gc
import os
import threading
import time

_STATE = {"thread": None, "seconds": None, "released": 0, "unpinned": 0, "device": 0, "error": None}


def _on(name):
    return os.environ.get(name, "0").strip().lower() not in ("0", "false", "no", "")


def _release(device):
    import torch

    from .offload import release_host_memory
    from .weight_stream import unpin_all

    started = time.perf_counter()
    try:
        torch.cuda.synchronize(device)  # no copy in flight reads or writes what goes now
        _STATE["released"] = release_host_memory()
        _STATE["unpinned"] = unpin_all()
        if os.environ.get("SOUP_SOLUTION_EXIT_RELEASE", "").strip().lower() == "all":
            # the caching allocator's free blocks too: the CUDA context teardown at exit has less to unmap
            reserved = torch.cuda.memory_reserved(device)
            torch.cuda.empty_cache()
            _STATE["device"] = reserved - torch.cuda.memory_reserved(device)
    except Exception as exc:  # noqa: BLE001 - an optimisation must not break the end of a run
        _STATE["error"] = repr(exc)
    _STATE["seconds"] = time.perf_counter() - started


def start_release(device=None):
    """Starts the release thread once (later calls do nothing). Returns the thread."""
    if _STATE["thread"] is None:
        thread = threading.Thread(target=_release, args=(device,), name="soup-exit-release", daemon=False)
        _STATE["thread"] = thread
        thread.start()
        atexit.register(lambda: print(f"[solution] exit release: {report()}", flush=True))
        if _on("SOUP_SOLUTION_EXIT_NOGC"):
            gc.disable()
    return _STATE["thread"]


def report():
    if _STATE["thread"] is None:
        return "not started"
    if _STATE["seconds"] is None:
        return "still running"
    device = f", {_STATE['device'] / 2**20:.0f} MiB of cached GPU memory released" if _STATE["device"] else ""
    return (f"{_STATE['released'] / 2**20:.0f} MiB of offload slabs freed, {_STATE['unpinned'] / 2**20:.0f} MiB "
            f"unregistered{device} in {_STATE['seconds']:.2f} s" + (f" ({_STATE['error']})" if _STATE["error"] else ""))


def patch_soup_exit_release():
    """SFTTrainerWrapper.train gets a callback that starts the release at the last step's end (or at
    the train end, if training stops before max_steps); fake trainers (Soup's tests) and CPU runs
    are left alone."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_exit_release", False):
        return
    original = cls.train

    def train(self, *args, **kwargs):
        from .offload import _real_cuda_trainer

        trainer = getattr(self, "trainer", None)
        if _real_cuda_trainer(trainer):
            from transformers import TrainerCallback

            device = next(trainer.model.parameters()).device
            # an evaluation after the last step would run the model (copies from the host buffers):
            # with one configured the release waits for the end of train()
            strategy = getattr(getattr(trainer, "args", None), "eval_strategy", "no")
            evaluates = getattr(trainer, "eval_dataset", None) is not None or                 str(getattr(strategy, "value", strategy)).lower() != "no"

            class _ExitRelease(TrainerCallback):
                def on_step_end(self, args, state, control, **kw):
                    if not evaluates and state.max_steps and state.global_step >= state.max_steps:
                        start_release(device)

                def on_train_end(self, args, state, control, **kw):
                    start_release(device)

            trainer.add_callback(_ExitRelease())
        return original(self, *args, **kwargs)

    cls.train = train
    cls._soup_exit_release = True
