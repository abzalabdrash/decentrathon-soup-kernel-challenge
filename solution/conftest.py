"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward on int8 tensor cores and offload_int8.py sends the MLP's (gate|up)
to the host as int8, both by default in every process; int8_fwd.py (the gate|up forward on int8
tensor cores) is behind SOUP_SOLUTION_INT8_FWD, off by default. The bitwise tests of the fp16 path
(test_fused_proj.py and the tests of the modules built on it) compare against plain PyTorch, so every
test starts with all three off; the int8 paths have their own tests (test_int8_bwd.py,
test_int8_fwd.py, test_offload_int8.py, fixture `int8_on`), which switch them on explicitly, and
test_default_config.py runs whole training steps in the configuration the training runs (fixture
`default_config`: INT8_DEFAULTS) against the all-fp16 one.

Memory tests measure with the `cuda_bytes` fixture: the allocator's requested bytes, which do not
depend on what the earlier tests of the process left in its pool."""

import pytest
import torch

# the int8 switches of the configuration the training runs (the modules' defaults, no SOUP_SOLUTION_*
# variable set): int8 MLP backward and int8 (gate|up) offload on, int8 gate|up forward off
INT8_DEFAULTS = {"int8_bwd": True, "offload_int8": True, "int8_fwd": False}


@pytest.fixture(autouse=True)
def _fp16_mlp_backward():
    """Yields the switches as this test found them (the modules' import-time values: every test
    restores them), {module name: on}."""
    try:
        from solution import int8_bwd, int8_fwd
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield None
        return
    from solution import offload_int8

    saved = int8_bwd._STATE["on"], int8_fwd._STATE["on"], offload_int8._STATE["on"]
    int8_bwd._STATE["on"] = int8_fwd._STATE["on"] = offload_int8._STATE["on"] = False
    yield dict(zip(("int8_bwd", "int8_fwd", "offload_int8"), saved))
    int8_bwd._STATE["on"], int8_fwd._STATE["on"], offload_int8._STATE["on"] = saved


@pytest.fixture
def default_config(_fp16_mlp_backward):
    """The int8 switches set as INT8_DEFAULTS instead of the fp16 pins above, the kernels compiled.
    Yields use(default): True sets INT8_DEFAULTS, False all three off (the fp16 reference); the pins'
    teardown restores the switches. Where the int8 paths take over is each module's START_CALLS."""
    if _fp16_mlp_backward is None or not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU and triton")
    from solution import int8_bwd, int8_fwd, offload_int8

    modules = {"int8_bwd": int8_bwd, "offload_int8": offload_int8, "int8_fwd": int8_fwd}

    def use(default):
        for name, module in modules.items():
            module._STATE["on"] = INT8_DEFAULTS[name] and default

    int8_bwd.compile_now()
    offload_int8.compile_now()
    use(True)
    yield use


class CudaBytes:
    """What the live tensors asked the caching allocator for (`requested_bytes`), now and at the peak.

    torch.cuda.memory_allocated() counts blocks, not requests: a request served from a cached free
    block keeps the whole block when less than 1 MiB would be left over, so the same tensors count up
    to ~1 MiB more each, depending on which blocks earlier code (other tests, other modes of the same
    test) freed in which order. A half (826, 1536) branch placed into a freed (826, 2048) QKV block
    reads 826 KiB too many. The requested bytes are the tensors' own sizes, whatever the pool holds."""

    @staticmethod
    def _stat(key):
        return torch.cuda.memory_stats().get(f"requested_bytes.all.{key}", 0)

    def now(self):
        return self._stat("current")

    def peak(self):
        return self._stat("peak")

    def reset_peak(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


@pytest.fixture
def cuda_bytes():
    """CudaBytes, after anything of earlier tests that could still allocate or free has settled: the
    warm-up thread of an apply() call (it compiles and runs kernels on the GPU) and tensors held in
    garbage cycles (an autograd graph freed by a gc pass in the middle of a measurement)."""
    import gc
    import sys

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    warmup = sys.modules.get("solution.warmup")
    if warmup is not None:
        warmup.wait_for_warmup(120)
    gc.collect()
    torch.cuda.synchronize()
    return CudaBytes()
