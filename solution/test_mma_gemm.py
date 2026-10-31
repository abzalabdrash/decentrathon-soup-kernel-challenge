"""mma_gemm.py (B3): the stage hook's TTGIR rewrite, the tensor-core GEMMs against torch, and the fused
int8 SwiGLU backward against int8_bwd.py's two-kernel path, bit for bit (the int32 product is exact,
the epilogue repeats the same fp32 operations). On a T4 the kernels run Triton's Turing MMA through
the hook; on newer GPUs tl.dot takes Triton's own MMA path, the same code."""

import pytest
import torch

from solution import mma_gemm

cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (7, 5),
                          reason="needs a CUDA GPU with tensor cores (sm_75+)")


def test_rewrite_marks_turing_and_cuda75():
    text = ('#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], '
            'instrShape = [16, 8]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:80"} {')
    out = mma_gemm._rewrite_ttgir(text)
    assert "versionMajor = 2, versionMinor = 1" in out and '"cuda:75"' in out and "cuda:80" not in out


@cuda
def test_fp16_matmul_matches_torch():
    gen = torch.Generator("cuda").manual_seed(0)
    for fwd in (True, False):
        a = (torch.randn(336, 1536, device="cuda", generator=gen) * 0.5).half()
        w = (torch.randn((2048, 1536) if fwd else (1536, 2048), device="cuda", generator=gen) * 0.05).half()
        b = w.t() if fwd else w
        compiled, out = mma_gemm.matmul(a, b, (64, 64, 32, 4, 8))
        ref = torch.mm(a.float(), b.float())
        assert (out.float() - ref).abs().max() <= 2e-3 * ref.abs().max()
        if torch.cuda.get_device_capability() == (7, 5):
            assert "mma.sync.aligned.m16n8k8" in compiled.asm["ptx"] and "cp.async" not in compiled.asm["ptx"]


@cuda
def test_int8_matmul_exact():
    gen = torch.Generator("cuda").manual_seed(1)
    a = torch.randint(-127, 128, (336, 1536), device="cuda", dtype=torch.int8, generator=gen)
    w = torch.randint(-127, 128, (2048, 1536), device="cuda", dtype=torch.int8, generator=gen)
    _, out = mma_gemm.matmul_i8(a, w, (128, 128, 64, 4, 8))
    assert torch.equal(out, torch._int_mm(a, w.t()))


@cuda
@pytest.mark.parametrize("tokens", [64, 400, 832])
def test_int8_swiglu_bwd_bitwise(tokens):
    import triton

    from solution import int8_bwd

    gen = torch.Generator("cuda").manual_seed(tokens)
    inter, d = 8960, 1536
    w = torch.randint(-127, 128, (inter, d), device="cuda", dtype=torch.int8, generator=gen)
    g = torch.randint(-127, 128, (tokens, d), device="cuda", dtype=torch.int8, generator=gen)
    w_scale = torch.rand(inter, device="cuda", generator=gen) * 1e-3
    g_scale = torch.rand(tokens, device="cuda", generator=gen) * 1e-3
    gate_up = (torch.randn(2 * inter, tokens, device="cuda", generator=gen) * 2).half()
    ref = gate_up.clone()
    groups = int8_bwd.GROUPS
    pref = torch.empty((groups, tokens), dtype=torch.float32, device="cuda")
    int8_bwd.int8_swiglu_bwd_kernel[(triton.cdiv(tokens, int8_bwd.TILE_M), groups)](
        torch._int_mm(w, g.t()), w_scale, g_scale, ref, pref, inter, tokens, inter // groups,
        TI=int8_bwd.TILE_I, TM=int8_bwd.TILE_M)
    pmax = mma_gemm.int8_swiglu_bwd(w, g, w_scale, g_scale, gate_up)
    assert torch.equal(gate_up.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(pmax.amax(0), pref.amax(0))


@cuda
def test_mlp_backward_same_with_and_without_mma(monkeypatch):
    """int8_bwd.mlp_backward with SOUP_SOLUTION_MMA_INT8_BWD on and off: the same dL/dx bit for bit
    (on a T4 the tensor-core kernel runs; elsewhere both runs take the two-kernel path)."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import int8_bwd
    from solution.test_fused_proj import _config, _quantize_mlp

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    int8_bwd.compile_now()
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=22).cuda().train()
    gen = torch.Generator("cuda").manual_seed(8)
    x = torch.randn(1, 832, 1536, device="cuda", generator=gen)
    grad = (torch.randn(1, 832, 1536, device="cuda", generator=gen) * 1e-2).half()
    grads = []
    for mma in (False, True):
        monkeypatch.setitem(int8_bwd._STATE, "mma", mma)
        xg = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(xg)
        out.backward(grad.to(out.dtype))
        grads.append(xg.grad)
    assert torch.equal(grads[0], grads[1])
