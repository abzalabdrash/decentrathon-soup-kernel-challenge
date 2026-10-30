"""The background warm-up: the registry and the thread, and that after it the training launches of
our kernels compile nothing (seen through triton's JIT hooks)."""

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fresh(monkeypatch):
    """The warm-up module with an empty registry, a fresh thread state and its own JIT hooks (all
    restored afterwards)."""
    from triton import knobs

    from solution import warmup

    monkeypatch.setattr(warmup, "BUILTIN", list(warmup.WARMUPS), raising=False)
    monkeypatch.setattr(warmup, "WARMUPS", [])
    monkeypatch.setattr(warmup, "COMPILES", [])
    monkeypatch.setattr(warmup, "_STATE", {"thread": None, "ident": None, "begin": None,
                                           "seconds": None, "results": {}, "errors": {}})
    monkeypatch.setattr(warmup, "_DONE", threading.Event())
    monkeypatch.setattr(warmup, "_HOOKS", {"installed": False})
    monkeypatch.setattr(knobs.runtime, "jit_cache_hook", None)
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", None)
    return warmup


def test_register_forms_and_replace(fresh):
    @fresh.register
    def first():
        pass

    @fresh.register(name="named")
    def second():
        pass

    def third():
        pass

    fresh.register(third, name="named")  # same name: replaces, keeps the place
    assert [name for name, _ in fresh.WARMUPS] == [f"{__name__}.{first.__qualname__}", "named"]
    assert fresh.WARMUPS[1][1] is third


def test_thread_runs_all_and_survives_errors(fresh):
    seen = []

    interval = sys.getswitchinterval()

    @fresh.register(name="a")
    def a():
        seen.append(("a", threading.current_thread().name,
                     torch.cuda.current_stream() != torch.cuda.default_stream()))
        seen.append(("switch", sys.getswitchinterval(), None))

    @fresh.register(name="boom")
    def boom():
        raise RuntimeError("broken warm-up")

    @fresh.register(name="c")
    def c():
        x = torch.ones(1024, device="cuda")  # a real launch, on the thread's own stream
        seen.append(("c", float(x.sum()), None))

    assert not fresh.wait_for_warmup(0)  # never started
    thread = fresh.start_background_warmup(log=False)
    assert fresh.start_background_warmup(log=False) is thread  # once per process
    assert fresh.wait_for_warmup(30)
    report = fresh.warmup_report()
    assert seen == [("a", "solution-warmup", True),
                    ("switch", pytest.approx(min(interval, fresh.SWITCH_INTERVAL)), None),
                    ("c", 1024.0, None)]
    assert sys.getswitchinterval() == interval  # restored
    assert set(report["results"]) == {"a", "c"}
    assert "broken warm-up" in report["errors"]["boom"]
    assert report["done"] and report["seconds"] is not None


def _training_calls():
    """Our kernels launched the way training launches them: `fused_lm_head_loss` with the run's
    512-row chunks on 826 tokens (n_grad_rows 512, then 88) and with 32-row chunks that give
    n_grad_rows 32, 3, 0 and 1; the example SwiGLU forward and backward on 826 and 77 tokens."""
    from solution.bench_warmup import fused_head_call, swiglu_call

    for n, n_labelled, num_items, chunk in ((826, 600, 900.0, 512), (80, 35, None, 32),
                                            (80, 33, 33.0, 32)):
        fused_head_call(n, n_labelled, num_items, chunk)
    for tokens in (826, 77):
        swiglu_call(tokens)


OUR_KERNELS = {"fused_ce_rows_kernel", "example_swiglu_fwd_kernel", "example_swiglu_bwd_kernel",
               "rms_norm_rows_fwd", "rms_norm_rows_bwd", "nf4_dequant_kernel", "rope_qk_kernel"}


def _forget_compiled():
    """Empties triton's in-memory kernel cache of every kernel in solution/ (the disk cache stays)."""
    from triton.runtime.jit import JITFunction

    from solution import fused_ce, nf4, rmsnorm, rope, swiglu

    for module in (fused_ce, swiglu, rmsnorm, nf4, rope):
        for kernel in vars(module).values():
            if isinstance(kernel, JITFunction):
                kernel.device_caches.clear()


def _keys(compiles):
    return {(c["kernel"], c["spec"], c["num_warps"], c["num_stages"]) for c in compiles}


def _builtin_warmups(fresh, monkeypatch):
    """The warm-ups warmup.py registers, for the T4's fp16 (this card would pick bf16)."""
    monkeypatch.setattr(fresh, "training_half_dtype", lambda: torch.float16)
    for name, fn in fresh.BUILTIN:
        fresh.register(fn, name=name)


def test_training_launches_compile_nothing_after_warmup(fresh, monkeypatch):
    _builtin_warmups(fresh, monkeypatch)
    compiles = fresh.watch_compiles(log=False)

    # without a warm-up the training launches compile, and the hook sees it
    _forget_compiled()
    _training_calls()
    needed = _keys(compiles)
    assert {k[0] for k in needed} == {"fused_ce_rows_kernel", "example_swiglu_fwd_kernel",
                                      "example_swiglu_bwd_kernel"}
    n_grad_classes = {k[1].split("n_grad_rows")[1].split(",")[0] for k in needed
                      if k[0] == "fused_ce_rows_kernel"}
    assert n_grad_classes == {":i32 D", ":i32", "=1"}

    # the warm-up compiles every one of them in its thread ...
    _forget_compiled()
    compiles.clear()
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(300)
    report = fresh.warmup_report()
    assert not report["errors"], report["errors"]
    assert report["compiles_in_warmup"] == len(compiles) > 0
    assert needed <= _keys(compiles)

    # ... and the same training launches then compile nothing
    compiles.clear()
    _training_calls()
    assert compiles == []


def test_model_step_compiles_nothing_after_warmup(fresh, monkeypatch):
    """apply() and the training step of one decoder layer with the run's shapes, 4-bit weights and
    LoRA (bench_warmup.one_layer_model): every kernel the step compiles, the warm-up has compiled."""
    import solution
    from solution.bench_warmup import STEP_LENGTHS, model_step, one_layer_model

    _builtin_warmups(fresh, monkeypatch)
    for switch in ("SOUP_SOLUTION_RMSNORM", "SOUP_SOLUTION_NF4", "SOUP_SOLUTION_ROPE",
                   "SOUP_SOLUTION_UNPAD", "SOUP_SOLUTION_MLP16"):
        monkeypatch.delenv(switch, raising=False)
    solution.apply()
    model = one_layer_model()
    compiles = fresh.watch_compiles(log=False)

    _forget_compiled()
    for lengths in STEP_LENGTHS:
        model_step(model, lengths)
    needed = _keys(compiles)
    assert {k[0] for k in needed} == OUR_KERNELS

    _forget_compiled()
    compiles.clear()
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(300)
    report = fresh.warmup_report()
    assert not report["errors"], report["errors"]
    missing = needed - _keys(compiles)
    assert not missing, missing

    compiles.clear()
    for lengths in STEP_LENGTHS:
        model_step(model, lengths)
    assert compiles == []


def test_switched_off_patches_warm_nothing(fresh, monkeypatch):
    for switch in ("SOUP_SOLUTION_RMSNORM", "SOUP_SOLUTION_NF4", "SOUP_SOLUTION_ROPE"):
        monkeypatch.setenv(switch, "0")
    for name, fn in fresh.BUILTIN:
        if name.startswith(("rms_norm", "nf4", "rope")):
            fresh.register(fn, name=name)
    compiles = fresh.watch_compiles(log=False)
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(60)
    results = fresh.warmup_report()["results"]
    assert len(results) == 5 and set(results.values()) == {fresh.SKIPPED}
    assert compiles == []


def test_watch_compiles_keeps_previous_hooks(fresh, monkeypatch):
    from triton import knobs

    from solution.swiglu import example_swiglu_fwd_kernel

    calls = []
    monkeypatch.setattr(knobs.runtime, "jit_cache_hook", lambda **kw: calls.append("before"))
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", lambda **kw: calls.append("after"))
    compiles = fresh.watch_compiles(log=False)
    assert fresh.watch_compiles(log=False) is compiles  # installed once
    example_swiglu_fwd_kernel.device_caches.clear()
    fresh.compile_kernel(example_swiglu_fwd_kernel, torch.float32, torch.float32, torch.float32,
                         387, BLOCK=1024)
    assert calls == ["before", "after"]
    assert len(compiles) == 1
    record = compiles[0]
    assert record["kernel"] == "example_swiglu_fwd_kernel" and record["is_warmup"]
    assert record["spec"] == ("gate_ptr:*fp32 D, up_ptr:*fp32 D, out_ptr:*fp32 D, "
                              "n_elements:i32, BLOCK=1024")


def test_preload_in_background():
    from solution.warmup import preload_in_background

    ready = preload_in_background(["json", "solution_no_such_module_xyz"], log=False)
    assert ready.wait(30)
    assert set(ready.errors) == {"solution_no_such_module_xyz"}
    assert set(ready.seconds) == {"json", "solution_no_such_module_xyz"}


def test_soup_hooks_start_and_wait(fresh, monkeypatch):
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    order = []
    monkeypatch.setattr(cls, "_soup_warmup", False, raising=False)
    monkeypatch.setattr(cls, "_setup_transformers", lambda self: order.append("setup"))
    monkeypatch.setattr(cls, "train", lambda self: order.append("train"))

    @fresh.register(name="slow")
    def slow():
        time.sleep(0.5)
        order.append("warmed")

    fresh.patch_soup_warmup(start_in_setup=True)
    wrapper = object.__new__(cls)
    wrapper._setup_transformers()
    assert fresh.warmup_report()["started"]
    wrapper.train()
    assert order == ["setup", "warmed", "train"]


def test_first_call_cold_and_warm(tmp_path):
    """The first training-shape launch in a fresh process with an empty triton cache, without and
    with the warm-up (bench_warmup.py first-call; numbers go to the report)."""
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path / "cache"))
    out = subprocess.run([sys.executable, "-m", "solution.bench_warmup", "first-call"], cwd=REPO,
                         env=env, capture_output=True, text=True, timeout=600)
    print(out.stdout)
    assert out.returncode == 0, out.stderr[-3000:]
    times = dict(re.findall(r"(\w+_s)=([0-9.]+)", out.stdout))
    assert float(times["warm_first_s"]) < float(times["cold_first_s"]) / 2
