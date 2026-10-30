"""The GEMM formulations of fused_proj.py behind SOUP_SOLUTION_QKV_SPLITK / O_LT / DOWN_LT: the q|k|v
backward in two K-halves against the single GEMM and an fp64 reference, the threshold, and the
cuBLASLt call wrapper (result, the backend set back). The layer-level comparison against the original
path is test_fused_proj.py run with the switches on."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _operands(m, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    g = torch.randn(m, 2048, generator=gen, device="cuda").half()
    w = (torch.randn(2048, 1536, generator=gen, device="cuda") * 0.02).half()
    c = torch.randn(m, 1536, generator=gen, device="cuda") * 0.01
    return g, w, c


@pytest.mark.parametrize("beta", [1.0, 1.25])
def test_qkv_split_k_within_fp32_rounding(monkeypatch, beta):
    from solution import fused_proj

    g, w, c = _operands(832)
    exact = beta * c.double() + g.double() @ w.double()
    monkeypatch.setitem(fused_proj._GEMM, "qkv_splitk", False)
    single = fused_proj._qkv_grad_x(c.clone(), g, w, beta)
    monkeypatch.setitem(fused_proj._GEMM, "qkv_splitk", True)
    split = fused_proj._qkv_grad_x(c.clone(), g, w, beta)
    assert split.dtype == torch.float32 and split.shape == (832, 1536)
    top = exact.abs().max()
    err_single = (single.double() - exact).abs().max() / top
    err_split = (split.double() - exact).abs().max() / top
    # one more fp32 rounding of the partial sums: both at fp32 accumulation level
    assert err_single < 1e-5 and err_split < 1e-5, (float(err_single), float(err_split))


def test_split_k_only_below_the_threshold(monkeypatch):
    from solution import fused_proj

    monkeypatch.setitem(fused_proj._GEMM, "qkv_splitk", True)
    g, w, c = _operands(fused_proj.SPLITK_MAX_M)
    big = fused_proj._qkv_grad_x(c.clone(), g, w, 1.0)
    monkeypatch.setitem(fused_proj._GEMM, "qkv_splitk", False)
    assert torch.equal(big, fused_proj._qkv_grad_x(c.clone(), g, w, 1.0))


def test_lt_wrapper_restores_the_backend():
    from solution.fused_proj import _lt

    before = torch._C._get_blas_preferred_backend()
    a = torch.randn(832, 1536, device="cuda").half()
    b = (torch.randn(1536, 1536, device="cuda") * 0.02).half()
    ref = a.double() @ b.double()
    out = _lt(torch.mm, a, b.t().contiguous().t())
    assert out.dtype == torch.float16
    assert ((out.double() - ref).abs().max() / ref.abs().max()) < 2e-3
    assert torch._C._get_blas_preferred_backend() == before
    with pytest.raises(RuntimeError):
        _lt(torch.mm, a, torch.randn(3, 3, device="cuda").half())
    assert torch._C._get_blas_preferred_backend() == before
