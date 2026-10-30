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
    monkeypatch.setattr(warmup, "NEEDS_GPU", set(warmup.NEEDS_GPU))
    monkeypatch.setattr(warmup, "COMPILES", [])
    monkeypatch.setattr(warmup, "_STATE", warmup._new_state())
    monkeypatch.setattr(warmup, "_DONE", threading.Event())
    monkeypatch.setattr(warmup, "_STOP", threading.Event())
    monkeypatch.setattr(warmup, "_HOOKS", {"installed": False})
    monkeypatch.setattr(knobs.runtime, "jit_cache_hook", None)
    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", None)
    monkeypatch.setattr(knobs.compilation, "listener", None)
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
        seen.append(("a", threading.current_thread().name.startswith("solution-warmup"),
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
    assert sorted(seen, key=str) == sorted([("a", True, True), ("c", 1024.0, None),
                                            ("switch", pytest.approx(min(interval,
                                                                         fresh.SWITCH_INTERVAL)),
                                             None)], key=str)
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
    LoRA (bench_warmup.one_layer_model): every kernel the step compiles, the warm-up has compiled.
    apply() runs with its own warm-up off (it would start the thread once for the process, before
    the kernels are forgotten below); the test starts it."""
    import solution
    from solution.bench_warmup import STEP_LENGTHS, model_step, one_layer_model

    _builtin_warmups(fresh, monkeypatch)
    for switch in ("SOUP_SOLUTION_RMSNORM", "SOUP_SOLUTION_NF4", "SOUP_SOLUTION_ROPE",
                   "SOUP_SOLUTION_UNPAD", "SOUP_SOLUTION_MLP16"):
        monkeypatch.delenv(switch, raising=False)
    monkeypatch.setenv("SOUP_SOLUTION_WARMUP", "0")
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


def _fused_rope_calls(n):
    """rope.py's kernel launched as fused_proj.py launches it: q and k views of the half qkv buffer
    (N, 2048) into half (N, 1536) and (N, 256) outputs, fp32 cos / sin; then the backward from
    contiguous gradients into views of the (N, 2304) gradient buffer."""
    from solution import rope
    from solution.fused_proj import _heads

    half = torch.float16
    qkv = torch.randn(n, 2048, device="cuda", dtype=half)
    cos = torch.randn(1, n, 128, device="cuda")
    sin = torch.randn(1, n, 128, device="cuda")
    q4 = _heads(torch.empty(n, 1536, dtype=half, device="cuda"), n, 12, 128)
    k4 = _heads(torch.empty(n, 256, dtype=half, device="cuda"), n, 2, 128)
    rope._launch(_heads(qkv[:, :1536], n, 12, 128), _heads(qkv[:, 1536:1792], n, 2, 128), q4, k4,
                 cos, sin, backward=False)
    g = torch.empty(n, 2304, dtype=half, device="cuda")
    rope._launch(torch.randn(1, 12, n, 128, device="cuda", dtype=half),
                 torch.randn(1, 2, n, 128, device="cuda", dtype=half),
                 _heads(g[:, :1536], n, 12, 128), _heads(g[:, 1536:1792], n, 2, 128), cos, sin,
                 backward=True)


def test_fused_proj_rope_launches_compile_nothing_after_warmup(fresh, monkeypatch):
    """The half RoPE of fused_proj.py's attention (the one-layer model of the test above takes
    rope.py's fp32 path on this card; on the T4 the real step takes this one)."""
    from solution import rope

    monkeypatch.delenv("SOUP_SOLUTION_ROPE", raising=False)
    monkeypatch.delenv("SOUP_SOLUTION_FUSED_PROJ", raising=False)
    _builtin_warmups(fresh, monkeypatch)
    compiles = fresh.watch_compiles(log=False)
    rope.rope_qk_kernel.device_caches.clear()
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(300)
    assert not fresh.warmup_report()["errors"]
    compiles.clear()
    for n in (832, 640, 1424):  # packed batches, padded to a multiple of 16
        _fused_rope_calls(n)
    assert compiles == []


def test_helper_process_compiles_for_this_one(tmp_path):
    """bench_warmup.py helper in a fresh process with an empty triton cache: the helper (no GPU,
    no CUDA context) compiles, and every kernel the in-process warm-up then meets is read from the
    on-disk cache, launcher and cuda_utils modules included (no C build in the training process)."""
    from solution.bench_warmup import _child

    res = _child(["child-helper"], str(tmp_path))
    assert res["helper_started"] == "True"
    assert res["helper_errors"] == "-" and res["errors"] == "-", res
    assert res["helper_cuda"] == "False"  # CUDA_VISIBLE_DEVICES is empty there
    assert int(res["helper_results"]) >= 7
    assert int(res["misses"]) >= 7 and res["cache_hits"] == res["misses"], res
    assert res["built"] == "-", res


@pytest.mark.parametrize("mode", ["helper", "thread"])
def test_exit_while_compiling_is_clean(tmp_path, mode):
    """A process that ends right after starting the warm-up (Soup's tests end 0.6 s after apply())
    exits with 0: the helper is killed and the exit waits for the thread's current registration,
    instead of tearing the compiler down under it (exit code 134 before warmup._shutdown)."""
    from solution.bench_warmup import _child_env

    for _ in range(3):
        out = subprocess.run([sys.executable, "-m", "solution.bench_warmup", "child-exit", mode],
                             cwd=REPO, env=_child_env(str(tmp_path)), capture_output=True,
                             text=True, timeout=300)
        assert out.returncode == 0, out.stderr[-3000:]
        assert "done_at_exit=False" in out.stdout  # it did exit in the middle of the warm-up


def test_late_registrations_go_to_the_helper(tmp_path):
    """Warm-ups registered after the helper started (from patch functions) are sent to it by name
    and compiled there: the training process still only loads from the cache."""
    from solution.bench_warmup import _child

    res = _child(["child-helper", "1"], str(tmp_path))
    assert res["helper_errors"] == "-" and res["errors"] == "-", res
    assert {"fused_proj_x16_rebuild", "head_stream_kernels"} <= set(res["helper_names"].split(","))
    assert int(res["misses"]) >= 10 and res["cache_hits"] == res["misses"], res
    assert res["built"] == "-", res


def test_reference_only_for_importable_functions():
    import functools

    from solution import rmsnorm, warmup

    assert warmup._reference("a", rmsnorm.warm_rebuild)["qualname"] == "warm_rebuild"
    spec = warmup._reference("b", functools.partial(warmup.warm_rope))
    assert spec["module"] == "solution.warmup" and spec["args"] == []

    def closure():
        pass

    assert warmup._reference("c", closure) is None
    assert warmup._reference("d", lambda: None) is None
    assert warmup._reference("e", functools.partial(rmsnorm.warm_rebuild, object())) is None


def test_cublas_never_goes_to_the_helper(fresh):
    @fresh.register(name="kernel")
    def kernel():
        pass

    @fresh.register(name="gpu-work", gpu=True)
    def gpu_work():
        pass

    for name, fn in fresh.BUILTIN:
        fresh.register(fn, name=name)
    assert "gpu-work" in fresh.NEEDS_GPU and "cublas" in fresh.NEEDS_GPU
    assert "kernel" not in fresh.NEEDS_GPU and "nf4_dequant_kernel" not in fresh.NEEDS_GPU


def test_helper_failure_falls_back_to_compiling_here(fresh, monkeypatch):
    """A helper that dies leaves the warm-up to this process: every registration still runs."""
    seen = []

    @fresh.register(name="a")
    def a():
        seen.append("a")

    monkeypatch.setattr(fresh, "__name__", "solution_no_such_module_xyz")
    assert fresh.start_compile_helper(log=False) is not None
    fresh.start_background_warmup(log=False)
    assert fresh.wait_for_warmup(120)
    report = fresh.warmup_report()
    assert seen == ["a"] and "a" in report["results"]
    assert "(helper)" in report["errors"] and report["helper"] is None


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
    assert len(results) == 4 and set(results.values()) == {fresh.SKIPPED}
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
