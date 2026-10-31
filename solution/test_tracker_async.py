"""SOUP_SOLUTION_TRACKER_THREAD (tracker_async.py): Soup's metric rows committed by a writer thread are
the rows Soup's own log_metrics commits, in order, and visible to every later tracker call."""

import pytest

tracker_mod = pytest.importorskip("soup_cli.experiment.tracker")

from solution import tracker_async  # noqa: E402

COLUMNS = ("run_id", "step", "epoch", "loss", "val_loss", "val_reward", "lr", "grad_norm", "speed", "gpu_mem")


def _log(tracker, run_id, n):
    for step in range(1, n + 1):
        tracker.log_metrics(run_id, step, epoch=step / n, loss=2.0 - step * 0.01, lr=2e-4 * step,
                            grad_norm=None if step % 3 else 0.5, speed=1.5, gpu_mem="371MiB")


def _rows(tracker, run_id):
    return [tuple(row[c] for c in COLUMNS) for row in tracker.get_metrics(run_id)]


def _start(tracker):
    return tracker.start_run({"base": "Qwen/Qwen2.5-1.5B-Instruct", "task": "sft"}, "cuda", "Tesla T4",
                             {"memory_total": "15 GB"})


def test_same_rows_in_order(tmp_path):
    plain = tracker_mod.ExperimentTracker(tmp_path / "plain.db")
    run = _start(plain)
    _log(plain, run, 20)
    expected = _rows(plain, run)
    plain.close()
    assert tracker_async.patch_tracker()
    try:
        ours = tracker_mod.ExperimentTracker(tmp_path / "ours.db")
        run2 = _start(ours)
        _log(ours, run2, 20)  # queued; get_metrics waits for them through _get_conn
        got = _rows(ours, run2)
        stamps = [row["timestamp"] for row in ours.get_metrics(run2)]
        ours.close()
    finally:
        tracker_async.unpatch_tracker()
    assert [r[1:] for r in got] == [r[1:] for r in expected] and len(got) == 20
    assert stamps == sorted(stamps)
    assert getattr(tracker_mod.ExperimentTracker.log_metrics, "_soup_original", None) is None


def test_writer_error_raised_by_the_next_call(tmp_path, monkeypatch):
    assert tracker_async.patch_tracker()
    try:
        ours = tracker_mod.ExperimentTracker(tmp_path / "ours.db")
        run = _start(ours)
        monkeypatch.setattr(tracker_async, "_INSERT", "INSERT INTO no_such_table VALUES (?)")
        ours.log_metrics(run, 1, loss=1.0)
        with pytest.raises(Exception):
            ours.get_metrics(run)
        ours.close()
    finally:
        monkeypatch.undo()
        tracker_async.unpatch_tracker()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SOUP_SOLUTION_TRACKER_THREAD", raising=False)
    assert not tracker_async.enabled()
