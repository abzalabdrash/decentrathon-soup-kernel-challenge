"""Soup's per-log SQLite writes in a thread of the training process (SOUP_SOLUTION_TRACKER_THREAD=1,
off by default until measured; the organizers allow it, answer 16 of the second wave).

Soup's `ExperimentTracker.log_metrics` (soup_cli/experiment/tracker.py:379-410) runs on the main thread
at every training log (logging_steps 1: 64 times a run): an INSERT and a `commit()` on a WAL database
with SQLite's default `synchronous=FULL`, so an fsync of the WAL file per row; on Kaggle 3.3-3.6 ms
each at the step boundary, while the GPU idles (B1, docs/notes/b1-boundary.md).

Here, while the switch is on:
- `log_metrics` takes its timestamp where Soup takes it (`datetime.now().isoformat()` when the call
  is made, on the calling thread) and queues the row; a writer thread of this process, with its own
  connection to the same database file (the journal mode is the file's: WAL), runs Soup's INSERT with
  the same columns and values and commits each row by itself, in the order of the calls: the same rows,
  the same fsync per row; nothing about the database's settings changes;
- every other use of the tracker goes through `_get_conn` (the tracker's connection), which first waits
  until the writer has committed every queued row of that database, so reads, `finish_run`, `fail_run`
  and `close` see exactly what the synchronous code would have committed by then; the interpreter's
  exit waits as well;
- a writer error is raised by the next tracker call (or at exit, logged), as Soup's own call would have
  raised it; a row with arguments the wrapper does not know takes Soup's own method.
"""

import atexit
import functools
import inspect
import os
import queue
import threading
from datetime import datetime

_INSERT = """INSERT INTO metrics
               (run_id, step, epoch, loss, val_loss, val_reward, lr, grad_norm, speed,
                gpu_mem, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""  # soup_cli/experiment/tracker.py:402-406
_FIELDS = ("run_id", "step", "epoch", "loss", "val_loss", "val_reward", "lr", "grad_norm", "speed", "gpu_mem")
LAST = {"queued": 0, "written": 0, "error": None}


def enabled():
    return os.environ.get("SOUP_SOLUTION_TRACKER_THREAD", "1").strip().lower() not in ("0", "false", "no", "")


class _Writer:
    """One database file's queue and writer thread."""

    def __init__(self, db_path):
        self.db_path = db_path
        self.jobs = queue.Queue()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="soup-tracker-writer", daemon=True)
        self.thread.start()

    def _run(self):
        import sqlite3

        conn = None
        while True:
            row = self.jobs.get()
            try:
                if row is None:
                    return
                if self.error is None:
                    if conn is None:
                        conn = sqlite3.connect(self.db_path)
                    conn.execute(_INSERT, row)
                    conn.commit()
                    LAST["written"] += 1
            except BaseException as exc:  # noqa: BLE001 - raised by the next tracker call
                self.error = exc
            finally:
                self.jobs.task_done()
                if row is None and conn is not None:
                    conn.close()

    def put(self, row):
        self.jobs.put(row)
        LAST["queued"] += 1

    def flush(self):
        """Waits until every queued row is committed; raises the writer's error, once."""
        self.jobs.join()
        if self.error is not None:
            error, self.error = self.error, None
            LAST["error"] = repr(error)
            raise error

    def close(self):
        self.jobs.put(None)
        self.thread.join()


_WRITERS = {}
_LOCK = threading.Lock()


def _writer(db_path):
    key = str(db_path)
    with _LOCK:
        found = _WRITERS.get(key)
        if found is None:
            found = _WRITERS[key] = _Writer(key)
        return found


def _flush_all():
    for writer in list(_WRITERS.values()):
        try:
            writer.flush()
        except BaseException as exc:  # noqa: BLE001 - at exit: logged, nothing left to raise into
            print(f"[solution] tracker writer failed: {exc!r}", flush=True)
        writer.close()


def _async_log_metrics(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def log_metrics(self, *args, **kwargs):
        try:
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            if set(values) - {"self"} != set(_FIELDS) or getattr(self, "db_path", None) is None:
                return original(self, *args, **kwargs)
        except TypeError:
            return original(self, *args, **kwargs)
        now = datetime.now().isoformat()  # where Soup takes it: when the call is made
        _writer(self.db_path).put(tuple(values[name] for name in _FIELDS) + (now,))

    log_metrics._soup_original = original
    return log_metrics


def _flushing_conn(original):
    @functools.wraps(original)
    def _get_conn(self):
        writer = _WRITERS.get(str(getattr(self, "db_path", "")))
        if writer is not None:
            writer.flush()
        return original(self)

    _get_conn._soup_original = original
    return _get_conn


def patch_tracker():
    """Installs the wrappers on Soup's ExperimentTracker (once). Returns whether they are in place."""
    try:
        from soup_cli.experiment import tracker as soup_tracker
    except Exception:  # noqa: BLE001 - another Soup: its own tracker
        return False
    cls = getattr(soup_tracker, "ExperimentTracker", None)
    if cls is None or not callable(getattr(cls, "log_metrics", None)) or not callable(getattr(cls, "_get_conn", None)):
        return False
    if getattr(cls.log_metrics, "_soup_original", None) is not None:
        return True
    params = list(inspect.signature(cls.log_metrics).parameters)
    if set(params[1:]) != set(_FIELDS):
        return False  # another log_metrics: its own code
    cls.log_metrics = _async_log_metrics(cls.log_metrics)
    cls._get_conn = _flushing_conn(cls._get_conn)
    atexit.register(_flush_all)
    return True


def unpatch_tracker():
    """Puts Soup's methods back after the queued rows are committed (tests)."""
    from soup_cli.experiment import tracker as soup_tracker

    cls = soup_tracker.ExperimentTracker
    for name in ("log_metrics", "_get_conn"):
        original = getattr(getattr(cls, name), "_soup_original", None)
        if original is not None:
            setattr(cls, name, original)
    with _LOCK:
        writers = list(_WRITERS.values())
        _WRITERS.clear()
    for writer in writers:
        try:
            writer.flush()
        finally:
            writer.close()
