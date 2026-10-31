"""SOUP_SOLUTION_INT8_WARMUP and the COSTS table of SOUP_SOLUTION_WARMUP_ORDER=longest (warmup.py):
the int8 kernels compiled by the warm-up leave nothing for int8_bwd.py's and offload_int8.py's own
compile threads, and the registrations go where the order says."""

import threading

import pytest
import torch

from solution.test_warmup import fresh  # noqa: F401 - the fixture

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_costs_name_every_registration():
    from solution import warmup

    names = {name for name, _ in warmup.WARMUPS} | {name for name, _ in warmup._probe_registrations()}
    names |= set(warmup.INT8_WARMUPS) | {"fused_proj_nf4_dequant4"}
    missing = sorted(name for name in names if name not in warmup.COSTS and name not in warmup.FIRST)
    assert not missing


def test_register_int8_follows_the_switch(fresh, monkeypatch):
    monkeypatch.setenv("SOUP_SOLUTION_INT8_WARMUP", "0")
    assert not fresh._register_int8() and fresh.WARMUPS == []
    monkeypatch.setenv("SOUP_SOLUTION_INT8_WARMUP", "1")
    assert fresh._register_int8()
    assert [name for name, _ in fresh.WARMUPS] == list(fresh.INT8_WARMUPS)


def test_probe_runs_them():
    from solution import launcher_one, warmup

    assert all(launcher_one.probe_safe(fn) for fn in (warmup._warm_int8_bwd, warmup._warm_offload_int8,
                                                      warmup._warm_int8_in))


@pytest.mark.parametrize("order", ["", "longest"])
def test_int8_registrations_order(fresh, monkeypatch, order):
    monkeypatch.setenv("SOUP_SOLUTION_WARMUP_WORKERS", "1")  # completion order = queue order
    monkeypatch.setenv("SOUP_SOLUTION_WARMUP_ORDER", order)
    monkeypatch.delenv("SOUP_SOLUTION_WARMUP_START", raising=False)
    monkeypatch.delenv("SOUP_SOLUTION_WARMUP_SKIP", raising=False)
    early = ["cublas", "step_ops", "rms_norm_rows_fwd", "example_swiglu_fwd_kernel"]
    late = ["head_stream_kernels", "fused_proj_nf4_dequant4"]
    for name in early:
        fresh.register(lambda: None, name=name)
    for name in fresh.INT8_WARMUPS:
        fresh.register(lambda: None, name=name)
    fresh.start_background_warmup(log=False, final=False)  # the early start keeps int8 back
    for name in late:
        fresh.register(lambda: None, name=name)
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(30)
    done = list(fresh._STATE["results"])
    if order == "longest":  # each call's batch by cost, FIRST in front; int8 among the final call's
        assert done == ["cublas", "step_ops", "rms_norm_rows_fwd", "example_swiglu_fwd_kernel",
                        "fused_proj_nf4_dequant4", "head_stream_kernels", "offload_int8_kernels",
                        "int8_bwd_kernels", "int8_in_kernels"]
    else:
        assert done == early + late + list(fresh.INT8_WARMUPS)


def test_int8_warmup_leaves_nothing_for_the_own_threads(fresh, monkeypatch):
    from solution import int8_bwd, offload_int8

    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    monkeypatch.setitem(offload_int8._STATE, "on", True)
    monkeypatch.setitem(offload_int8.RAW, "on", True)
    monkeypatch.setenv("SOUP_SOLUTION_INT8_WARMUP", "1")
    monkeypatch.delenv("SOUP_SOLUTION_WARMUP_START", raising=False)
    monkeypatch.delenv("SOUP_SOLUTION_WARMUP_SKIP", raising=False)
    fresh.watch_compiles(log=False)
    assert fresh._register_int8()
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(600)
    report = fresh.warmup_report()
    assert report["errors"] == {}
    assert all(report["results"][name] != fresh.SKIPPED for name in fresh.INT8_WARMUPS)
    kernels = {c["kernel"] for c in fresh.COMPILES if c["thread"] in fresh._STATE["idents"]}
    before = len(fresh.COMPILES)

    # what the own compile threads run (int8_bwd._start_compile, offload_int8._start_compile)
    errors = []

    def own():
        try:
            assert int8_bwd._warm() is None
            assert offload_int8._warm() is None
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=own)
    thread.start()
    thread.join(120)
    assert errors == []
    assert fresh.COMPILES[before:] == []  # every variant was in triton's in-memory cache
    # in a fresh process the warm-up compiled them all (here earlier tests may have compiled some)
    assert kernels <= {"nf4_int8_t_kernel", "int8_swiglu_bwd_kernel", "quant_t_int8_kernel",
                       "gu_quant_kernel", "gu_dequant_kernel", "quant_t_int8_il_kernel"}


def test_int8_warmup_skips_when_switched_off(fresh, monkeypatch):
    from solution import int8_bwd, offload_int8

    monkeypatch.setitem(int8_bwd._STATE, "on", False)
    monkeypatch.setitem(offload_int8._STATE, "on", False)
    assert fresh._warm_int8_bwd() == fresh.SKIPPED
    assert fresh._warm_offload_int8() == fresh.SKIPPED
    assert fresh._warm_int8_in() == fresh.SKIPPED
    monkeypatch.setitem(offload_int8._STATE, "on", True)  # int8_bwd off: no INT8_IN variants
    assert fresh._warm_int8_in() == fresh.SKIPPED
