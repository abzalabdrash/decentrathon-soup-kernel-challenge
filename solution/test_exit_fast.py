"""The fast exit (exit_fast.py) in child processes: os._exit only after a clean 0 from the wrapped
run(), after every atexit callback registered later and every non-daemon thread, with stdout
(a pipe, unbuffered mode off), C stdio and logging flushed; any other ending exits the normal way
with its own code. A pytest process (run() never called) keeps its summary line and exit code."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent('''
    import atexit, logging, logging.handlers, sys, threading, time, types

    atexit.register(lambda: print("LATE-ATEXIT", flush=True))  # registered before ours: runs after it
    from solution import exit_fast

    exit_fast.register_exit()
    MODE, LOG = sys.argv[1], sys.argv[2]
    cli = types.ModuleType("fake_soup_cli")

    def run():
        print("in run")
        if MODE in ("zero", "after", "cuda"):
            sys.exit(0)
        if MODE == "one":
            sys.exit(1)
        if MODE == "none":
            raise SystemExit(None)
        if MODE == "error":
            raise RuntimeError("boom")

    cli.run = run
    exit_fast.wrap_cli_run(cli)
    exit_fast.wrap_cli_run(cli)  # a second call does not wrap twice
    atexit.register(lambda: print("OUR-ATEXIT"))  # registered after ours: runs before it

    def slow():
        time.sleep(0.3)
        print("THREAD-DONE")

    threading.Thread(target=slow, name="slow-writer", daemon=False).start()
    log = logging.getLogger("child")
    target = logging.FileHandler(LOG)
    log.addHandler(logging.handlers.MemoryHandler(1000, logging.CRITICAL + 1, target))  # flushed on close only
    log.warning("logged-line")
    if MODE == "cuda":
        import torch

        x = torch.randn(2048, 2048, device="cuda")
        for _ in range(20):
            x = x @ x
            x = x / x.norm()
        pinned = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        pinned.copy_(x, non_blocking=True)  # in flight until the exit's synchronize
    for i in range(20000):
        print(f"line {i}")
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL(None).printf(b"C-LEVEL\\n")
    code = 3
    if MODE != "skip":
        try:
            cli.run()
            code = 0
        except SystemExit as exc:  # harness/_train.py:174-179
            code = 0 if exc.code is None else (exc.code if isinstance(exc.code, int) else 1)
    if MODE == "after":
        raise ValueError("after run")
    sys.exit(code)
''')


def _child(tmp_path, mode):
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    log = tmp_path / "child.log"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(script), mode, str(log)], cwd=tmp_path, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
    return proc, (log.read_text(encoding="utf-8") if log.exists() else "")


def _flushed(out, err, logged, ran=True):
    assert "line 19999" in out and "THREAD-DONE" in out and ("in run" in out) == ran
    assert "logged-line" in logged
    if sys.platform.startswith("linux"):
        assert "C-LEVEL" in out


@pytest.mark.parametrize("mode", ["zero", "none"])
def test_clean_exit_is_fast(tmp_path, mode):
    proc, logged = _child(tmp_path, mode)
    out = proc.stdout
    assert proc.returncode == 0, proc.stderr
    assert "[solution] exit fast: os._exit(0) at " in out
    assert "LATE-ATEXIT" not in out  # the interpreter's own exit never ran
    assert out.index("OUR-ATEXIT") < out.index("[solution] exit fast")
    assert out.index("THREAD-DONE") < out.index("[solution] exit fast")
    _flushed(out, proc.stderr, logged)


@pytest.mark.parametrize("mode,code", [("one", 1), ("error", 1), ("skip", 3), ("after", 1)])
def test_other_endings_exit_normally(tmp_path, mode, code):
    proc, logged = _child(tmp_path, mode)
    out = proc.stdout
    assert proc.returncode == code, proc.stderr
    assert "exit fast" not in out
    assert "LATE-ATEXIT" in out and "OUR-ATEXIT" in out
    _flushed(out, proc.stderr, logged, ran=mode != "skip")
    if mode == "error":
        assert "RuntimeError: boom" in proc.stderr
    if mode == "after":
        assert "ValueError: after run" in proc.stderr


def test_cuda_work_is_synchronized(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    proc, logged = _child(tmp_path, "cuda")
    assert proc.returncode == 0, proc.stderr
    assert "[solution] exit fast: os._exit(0) at " in proc.stdout and "LATE-ATEXIT" not in proc.stdout
    _flushed(proc.stdout, proc.stderr, logged)


CONFTEST = textwrap.dedent('''
    import atexit, types
    from solution import exit_fast

    atexit.register(lambda: print("LATE-ATEXIT", flush=True))
    exit_fast.register_exit()
    cli = types.ModuleType("fake_soup_cli")
    cli.run = lambda: None
    exit_fast.wrap_cli_run(cli)  # as apply() in a pytest process: run() is never called
''')


@pytest.mark.parametrize("fail", [False, True])
def test_pytest_process_keeps_its_summary(tmp_path, fail):
    (tmp_path / "conftest.py").write_text(CONFTEST, encoding="utf-8")
    body = "def test_a():\n    assert True\n\ndef test_b():\n    assert %s\n" % ("False" if fail else "True")
    (tmp_path / "test_inner.py").write_text(body, encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUNBUFFERED", "PYTEST_ADDOPTS")}
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                          cwd=tmp_path, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=300)
    out = proc.stdout
    assert proc.returncode == (1 if fail else 0), out
    assert ("1 failed, 1 passed" if fail else "2 passed") in out.splitlines()[-2] + out.splitlines()[-1]
    assert "exit fast" not in out and "LATE-ATEXIT" in out


def test_switch_and_order(monkeypatch):
    from solution import exit_fast

    monkeypatch.delenv("SOUP_SOLUTION_EXIT_FAST", raising=False)
    assert not exit_fast.enabled()
    monkeypatch.setenv("SOUP_SOLUTION_EXIT_FAST", "1")
    assert exit_fast.enabled()
    assert [exit_fast._normalize(c) for c in (None, 0, 2, True, "x")] == [0, 0, 2, 1, 1]
    source = (REPO / "solution" / "__init__.py").read_text(encoding="utf-8")
    body = source[source.index("def apply():"):]
    # the exit callback is registered before every other atexit user of apply(), the wrapper last
    assert body.index("register_exit()") < body.index("patch_soup_phases()")
    assert body.index("register_exit()") < body.index("patch_weight_stream(")
    assert body.index("wrap_cli_run()") > body.index("start_background_warmup()")
