"""warmup.py SOUP_SOLUTION_LAUNCHERS_EARLY: the launchers built ahead during apply() are the ones the
warm-up then asks for (same source text, each used once), and the kernels launched through them
compute what they compute without it. Each case runs in a fresh process with an empty triton cache,
as a measured run does. Needs CUDA (the probe sets up triton's CUDA driver)."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
REPO = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent(r"""
    import hashlib, json, os, sys, threading
    sys.path.insert(0, sys.argv[1])
    import torch
    torch.empty(1, device="cuda")
    from triton import knobs
    from solution import warmup
    asked = []  # (source sha256, thread name) of every launcher build triton asks for

    warmup.install_launcher_build()
    ours = knobs.build.impl

    def recording(name, src, *args, **kwargs):
        with open(src, encoding="utf-8") as f:
            asked.append((hashlib.sha256(f.read().encode("utf-8")).hexdigest(), threading.current_thread().name))
        return ours(name, src, *args, **kwargs)

    recording._soup_launcher_build = ours._soup_launcher_build
    knobs.build.impl = recording
    thread = warmup.start_launchers_ahead()
    if thread is not None:
        thread.join(300)
    ahead = sorted(warmup._AHEAD["entries"])
    asked_ahead = list(asked)
    del asked[:]
    # every registration this file and the late modules make, compiled for real (no apply(): the
    # registrations of the other modules are called directly, as the probe calls them)
    for name, fn in warmup._probe_registrations():
        fn()
    from solution import rmsnorm  # one kernel launched for real through its launcher
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(37, 1536, device="cuda", generator=gen)
    weight = torch.randn(1536, device="cuda", generator=gen)
    with torch.no_grad():
        got = rmsnorm.rmsnorm(x, weight, 1e-6)
        want = rmsnorm.reference_rmsnorm(x, weight, 1e-6)
    err = float((got.float() - want.float()).abs().max())
    print(json.dumps({"ahead": ahead, "asked_while_ahead": asked_ahead, "asked": asked,
                      "built": warmup._AHEAD["built"], "failed": warmup._AHEAD["failed"],
                      "hits": warmup._AHEAD["hits"], "error": warmup._AHEAD["error"], "err": err}))
""")


def _child(early):
    env = dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="soup-test-triton-"),
               SOUP_SOLUTION_LAUNCHERS_EARLY="1" if early else "0")
    out = subprocess.run([sys.executable, "-c", CHILD, str(REPO)], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@cuda
def test_launchers_built_ahead_are_the_ones_asked_for():
    plain = _child(early=False)
    early = _child(early=True)
    assert plain["ahead"] == [] and plain["built"] == 0
    assert early["error"] is None and early["failed"] == 0 and early["built"] == len(early["ahead"]) > 0
    asked = [sha for sha, _ in plain["asked"]]  # the launchers (and cuda_utils) the warm-up builds
    # with the launchers ahead, the thread builds cuda_utils and every launcher source the warm-up
    # will ask for; the warm-up's own builds are all answered from them (same source text)
    built_by_thread = {sha for sha, thread in early["asked_while_ahead"]}
    assert set(early["ahead"]) <= built_by_thread
    launchers_asked = [sha for sha, _ in early["asked"]]
    assert set(launchers_asked) <= set(early["ahead"])
    assert early["hits"] == len(launchers_asked)
    assert set(launchers_asked) == set(asked) - (built_by_thread - set(early["ahead"]))  # minus cuda_utils
    assert early["err"] == plain["err"] and early["err"] < 1e-4  # the same launch, the same result


@cuda
def test_probe_skips_only_the_probing_thread(monkeypatch):
    """The probe's jit_cache_hook answers "skip" for the probing thread only; another thread's call
    gets the previous hook's answer."""
    import threading

    from solution import warmup

    seen = []

    def previous(**kwargs):
        seen.append(threading.get_ident())
        return "previous"

    hook = warmup._probe_hook(previous)
    probe = {"ident": threading.get_ident(), "signatures": []}
    monkeypatch.setitem(warmup._AHEAD, "probe", probe)
    assert hook(compile={"signature": {"x": "*fp32", "n": "i32"}}) is True
    assert probe["signatures"] == [{"x": "*fp32", "n": "i32"}] and seen == []
    answers = []
    other = threading.Thread(target=lambda: answers.append(hook(compile={"signature": {}})))
    other.start()
    other.join()
    assert answers == ["previous"] and len(seen) == 1 and probe["signatures"] == [{"x": "*fp32", "n": "i32"}]
    monkeypatch.setitem(warmup._AHEAD, "probe", None)
    assert hook(compile={}) == "previous"


def test_off_by_default(monkeypatch):
    from solution import warmup

    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHERS_EARLY", raising=False)
    assert warmup.start_launchers_ahead() is None
