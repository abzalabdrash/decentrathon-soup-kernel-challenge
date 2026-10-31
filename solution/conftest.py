"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward on int8 tensor cores by default, in every process. The bitwise
tests of the fp16 path (test_fused_proj.py and the tests of the modules built on it) compare against
plain PyTorch, so every test starts with the int8 backward off; the int8 path has its own tests
(test_int8_bwd.py, fixture `int8_on`), which switch it on explicitly.

Memory tests measure with the `cuda_bytes` fixture: the allocator's requested bytes, which do not
depend on what the earlier tests of the process left in its pool."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _fp16_mlp_backward():
    try:
        from solution import int8_bwd
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield
        return
    saved = int8_bwd._STATE["on"]
    int8_bwd._STATE["on"] = False
    yield
    int8_bwd._STATE["on"] = saved


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
