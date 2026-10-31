"""Exit and RNG safety of the warm-up threads (A1 jury audit; tests from codex/review-1, 49c639b).

Check 10 runs Soup's 79 tests in a process that applies the solution and exits ~0.6 s later, while
the warm-up still compiles: the exit must wait for every compiler thread (an abort is exit code 134).
CPU only: the threads are stand-ins."""

import threading

import pytest

from solution import warmup


def test_cublas_warmup_draws_no_global_random_numbers(monkeypatch):
    torch = pytest.importorskip("torch")
    # Exercise the actual warm-up on CPU too: both its old random allocation and its
    # deterministic replacement get redirected, so the old code fails the RNG assertion.
    for name in ("ones", "randn"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    monkeypatch.setattr(warmup, "training_half_dtype", lambda: torch.float32)
    state = torch.get_rng_state()
    warmup.warm_cublas()
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("kind", ["thread", "key_thread"])
def test_shutdown_drains_threads_past_one_join_interval(monkeypatch, kind):
    state = warmup._new_state()
    monkeypatch.setattr(warmup, "_STATE", state)
    monkeypatch.setattr(warmup, "_STOP", threading.Event())
    monkeypatch.setattr(warmup, "EXIT_TIMEOUT", 0.001)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def work():
        entered.set()
        release.wait()
        finished.set()

    state[kind] = thread = threading.Thread(target=work)
    thread.start()
    assert entered.wait(5)
    timer = threading.Timer(0.05, release.set)
    timer.start()
    try:
        warmup._shutdown()
        assert finished.is_set() and not thread.is_alive()
    finally:
        release.set()
        thread.join(5)
        timer.join(5)


def test_early_hash_thread_is_tracked_once_and_joined(monkeypatch):
    state = warmup._new_state()
    monkeypatch.setattr(warmup, "_STATE", state)
    monkeypatch.setattr(warmup, "_STOP", threading.Event())
    monkeypatch.setenv("SOUP_SOLUTION_WARMUP_START", "apply_end")
    monkeypatch.setattr(warmup, "install_launcher_build", lambda: False)
    monkeypatch.setattr(warmup, "install_light_triton_key", lambda: False, raising=False)
    monkeypatch.setattr(warmup, "start_launchers_ahead", lambda: None, raising=False)
    registered = []
    monkeypatch.setattr(warmup, "_register_shutdown", lambda: registered.append(True))
    entered, release = threading.Event(), threading.Event()

    def key():
        entered.set()
        release.wait()

    monkeypatch.setattr(warmup, "_triton_key", key)
    warmup.start_early(log=False)
    assert entered.wait(5)
    thread = state["key_thread"]
    try:
        warmup.start_early(log=False)
        assert state["key_thread"] is thread and registered == [True]
    finally:
        release.set()
        warmup._shutdown()
    assert not thread.is_alive()


@pytest.mark.parametrize("failure", ["worker_setup", "helper_start"])
def test_coordinator_failure_drains_live_helpers_before_done(monkeypatch, failure):
    torch = pytest.importorskip("torch")
    state, done, stop = warmup._new_state(), threading.Event(), threading.Event()
    monkeypatch.setattr(warmup, "_STATE", state)
    monkeypatch.setattr(warmup, "_DONE", done)
    monkeypatch.setattr(warmup, "_STOP", stop)
    monkeypatch.setattr(warmup, "_prepare_triton", lambda: None)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)
    entered, release, finished, failed = (threading.Event() for _ in range(4))
    synced = []

    class Stream:
        def synchronize(self):
            synced.append(finished.is_set())

    monkeypatch.setattr(torch.cuda, "Stream", lambda device: Stream())

    def worker(device, stream, log):
        if threading.current_thread().name == "solution-warmup-1":
            entered.set()
            release.wait()
            finished.set()
        else:
            assert entered.wait(5)
            failed.set()
            raise RuntimeError("worker device setup failed")

    monkeypatch.setattr(warmup, "_worker", worker)
    original_thread = threading.Thread
    helpers = []

    class CannotStart:
        def start(self):
            assert entered.wait(5)
            failed.set()
            raise RuntimeError("cannot start another worker")

    def make_thread(*args, **kwargs):
        if kwargs.get("name") == "solution-warmup-2":
            return CannotStart()
        thread = original_thread(*args, **kwargs)
        helpers.append(thread)
        return thread

    monkeypatch.setattr(warmup.threading, "Thread", make_thread)
    coordinator = original_thread(target=warmup._run,
                                  args=(0, False, 3 if failure == "helper_start" else 2))
    coordinator.start()
    try:
        assert entered.wait(5) and failed.wait(5)
        assert not done.wait(0.05), "DONE was published while a native worker was still active"
        assert coordinator.is_alive()
    finally:
        release.set()
        coordinator.join(5)
        for helper in helpers:
            helper.join(5)
    assert not coordinator.is_alive() and done.is_set() and stop.is_set()
    assert "(thread)" in state["errors"] and synced == [True]


def _check10_last_line(stdout):
    """harness/check_rules.py run_tests(): the last line that names passed / failed / error."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    summary = [line for line in lines if " passed" in line or " failed" in line or "error" in line]
    return (summary or lines or ["no output"])[-1]


def test_registration_failing_during_exit_prints_nothing(monkeypatch, capsys):
    """A registration that raises after the exit began (pytest has printed "79 passed" by then)
    must not print "[solution] warmup <name> failed: ...": check_rules would read that line as the
    summary and fail check 10. The error is still recorded."""
    torch = pytest.importorskip("torch")
    state, stop = warmup._new_state(), threading.Event()
    monkeypatch.setattr(warmup, "_STATE", state)
    monkeypatch.setattr(warmup, "_STOP", stop)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda stream: None)

    def compile_during_exit():
        stop.set()  # _shutdown runs while this registration compiles
        raise RuntimeError("compiler gone")

    state["queue"].put(("late", compile_during_exit))
    state["closed"] = True
    print("79 passed in 0.62s")
    warmup._worker(0, None, log=True)
    out = capsys.readouterr().out
    assert "late" in state["errors"] and "failed" not in out
    assert _check10_last_line(out) == "79 passed in 0.62s"
