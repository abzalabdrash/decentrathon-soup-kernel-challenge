"""cuda_limits.py: the per-thread stack limit goes down (never up) and kernels still run."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_lower_stack_limit_frees_device_memory_and_kernels_run():
    from solution import cuda_limits

    torch.ones(1, device="cuda")
    before = cuda_limits.stack_limit()
    assert before is not None and before > 0
    a = torch.randn(1440, 1536, device="cuda").half()
    w = torch.randn(2048, 1536, device="cuda").half()
    ref = a @ w.t()
    try:
        assert cuda_limits.lower_stack_limit(before * 2) is None  # never raised
        changed = cuda_limits.lower_stack_limit(256)
        if before > 256:
            assert changed is not None and changed[1] == 256
        q = torch.randn(1, 12, 300, 128, device="cuda", dtype=torch.float16, requires_grad=True)
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        out.sum().backward()
        assert torch.equal(a @ w.t(), ref)
        torch.cuda.synchronize()
    finally:
        cuda_limits._cudart().cudaDeviceSetLimit(cuda_limits.LIMIT_STACK_SIZE,
                                                 __import__("ctypes").c_size_t(before))
