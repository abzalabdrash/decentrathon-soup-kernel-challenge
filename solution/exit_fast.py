"""The process ends with os._exit once everything it had to do is done (SOUP_SOLUTION_EXIT_FAST=1,
off by default).

The harness times the whole process, its exit included. After the last atexit callback the
interpreter still finalizes its modules with a full garbage collection, runs weakref finalizers and
the C exit (static destructors, the CUDA runtime's teardown) before the kernel reaps the process; on
the T4 that was 0.12 s of finalization and 0.25 s of C exit with the pinned release on
(docs/notes/e1-exit-head.md:63-70). None of it writes anything the run produces.

Organizers' answer (31 Oct ~20:54, docs/briefs/organizers-qa-2026-10-31.md): os._exit is allowed
when every computation, copy and write is complete, Soup's outputs are saved, errors are not hidden,
no unfinished work leaves the measured span and the behaviour is the same in every run, without
reading metrics.json or detecting the harness. How this module keeps to it:
- the exit code is the one Soup's own entry point ends with: `soup_cli.cli.run` is wrapped and the
  code of its SystemExit normalized as soup_cli/cli.py:1035-1041 does (the console script `soup`
  and the harness, harness/_train.py:174-179, both exit with it). Only a clean 0 exits early; a
  non-zero code, an exception out of run(), an uncaught exception after it (sys.last_exc) or a
  process that never called run() (pytest, Soup's tests) exits the normal way;
- the callback is registered at the start of apply(), before any of ours, so atexit (LIFO) runs it
  after all of them: the compile threads' joins, the tracker writer's flush, the release report. The
  interpreter joins every non-daemon thread (exit_release's soup-exit-release among them) in
  threading._shutdown before atexit; the release thread is joined here once more, and any other
  non-daemon thread still alive sends the process the normal way;
- then the device is synchronized (no kernel or copy in flight), logging.shutdown() flushes and
  closes every handler (its own atexit callback was registered before ours and would come later),
  Python's and C's stdio buffers are flushed, and a log line with the wall clock is printed right
  before os._exit (check 9: the next run's GPU memory at start is read after the harness's wait()).
Skipped: the weakref finalizers left at exit (in-memory clean-up), DataLoader's exit flag, the
modules' finalization and the C exit. Soup's files are all written and closed by then: the adapter
and checkpoint-64 (save_model returns after the files are closed), rewind.jsonl (opened per record),
the experiment database (each row committed with an fsync of the WAL; the connection is left open,
so its -wal file stays next to it with every committed row).
"""

import atexit
import os
import sys
import threading
import time

_STATE = {"code": None, "registered": False}


def enabled():
    return os.environ.get("SOUP_SOLUTION_EXIT_FAST", "0").strip().lower() not in ("0", "false", "no", "")


def _normalize(code):
    """soup_cli/cli.py:1035-1041."""
    if code is None:
        return 0
    if isinstance(code, int) and not isinstance(code, bool):
        return code
    return 1


def register_exit():
    """Registers the exit callback; called first in apply(), so it runs after every atexit callback
    our modules register later. Once per process."""
    if not _STATE["registered"]:
        _STATE["registered"] = True
        atexit.register(_final)


def wrap_cli_run(module=None):
    """Wraps `run` of soup_cli.cli (or of `module`) to remember the code Soup's entry point ends with.
    Called at the end of apply(): the harness imports run after apply() (harness/_train.py:175)."""
    if module is None:
        import soup_cli.cli as module
    original = module.run
    if getattr(original, "_soup_exit_fast", False):
        return

    def run(*args, **kwargs):
        _STATE["code"] = None
        try:
            result = original(*args, **kwargs)
        except SystemExit as exc:
            _STATE["code"] = _normalize(exc.code)
            raise
        except BaseException:
            _STATE["code"] = "error"
            raise
        _STATE["code"] = 0 if result is None else "returned"
        return result

    run._soup_exit_fast = True
    run.__wrapped__ = original
    module.run = run


def _uncaught():
    return getattr(sys, "last_exc", None) is not None or getattr(sys, "last_value", None) is not None


def _flush_stdio():
    for stream in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__):
        try:
            if stream is not None:
                stream.flush()
        except Exception:  # noqa: BLE001 - a closed stream has nothing to flush
            pass
    if os.name == "posix":
        try:
            import ctypes

            ctypes.CDLL(None).fflush(None)  # C stdio of extensions: _exit() does not flush it
        except Exception:  # noqa: BLE001
            pass


def _final():
    if _STATE["code"] != 0 or _uncaught():
        return  # run() not called, failed, or something failed after it: the normal exit
    release = sys.modules.get("solution.exit_release")
    thread = getattr(release, "_STATE", {}).get("thread") if release is not None else None
    if thread is not None and thread is not threading.current_thread():
        thread.join()
    main = threading.main_thread()
    alive = [t for t in threading.enumerate() if t is not main and t.is_alive()]
    if any(not t.daemon for t in alive):
        print(f"[solution] exit fast: normal exit, threads still running: "
              f"{', '.join(t.name for t in alive if not t.daemon)}", flush=True)
        return
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 - a device error is not hidden: the normal exit reports it
        print(f"[solution] exit fast: normal exit, synchronize failed: {exc!r}", flush=True)
        return
    import logging

    logging.shutdown()
    daemons = f", daemon threads left: {', '.join(t.name for t in alive)}" if alive else ""
    print(f"[solution] exit fast: os._exit(0) at {time.time():.3f} "
          f"(perf_counter {time.perf_counter():.3f}){daemons}", flush=True)
    _flush_stdio()
    os._exit(0)
