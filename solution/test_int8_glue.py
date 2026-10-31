"""int8_glue.py (task H2): its kernels give the torch glue's values bit for bit, non-finite rows stay
non-finite, the gate|up scales are kept per layer, and int8_bwd's MLP backward with the glue equals the
one without it."""

import pytest
import torch

from solution.test_fused_proj import _config, _quantize_mlp, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def int8_on():
    from solution import int8_bwd

    saved = int8_bwd._STATE["on"]
    int8_bwd._STATE["on"] = True
    int8_bwd.compile_now()
    yield int8_bwd
    int8_bwd._STATE["on"] = saved


def _rows(n, d, dtype, seed):
    gen = torch.Generator("cuda").manual_seed(seed)
    scale = torch.rand(n, 1, device="cuda", generator=gen) * 4 + 1e-3
    x = (torch.randn(n, d, device="cuda", generator=gen) * scale).to(dtype)
    x[: min(3, n)] = 0  # padding rows
    return x


@pytest.mark.parametrize("n", [1, 17, 832, 1440])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quant_rows_equals_torch(int8_on, n, dtype):
    from solution import int8_glue

    x = _rows(n, 1536, dtype, seed=n)
    q_ref, s_ref = int8_on._quant_rows(x)
    q, s = int8_glue.quant_rows(x)
    assert torch.equal(q, q_ref) and torch.equal(s, s_ref)


def test_quant_rows_nonfinite(int8_on):
    from solution import int8_glue

    x = _rows(40, 1536, torch.float16, seed=3)
    x[10, 5] = float("nan")
    x[11, 7] = float("inf")
    x[12, 9] = float("-inf")
    q, s = int8_glue.quant_rows(x)
    _, s_ref = int8_on._quant_rows(x)
    assert torch.isnan(s[10]) and torch.isinf(s[11]) and torch.isinf(s[12])
    finite = torch.ones(40, dtype=torch.bool, device="cuda")
    finite[10:13] = False
    assert torch.equal(s[finite], s_ref[finite])


@pytest.mark.parametrize("n", [17, 832])
def test_dequant_equals_torch(n):
    from solution import int8_glue

    gen = torch.Generator("cuda").manual_seed(n)
    acc = torch.randint(-(2**24), 2**24, (n, 1536), dtype=torch.int32, device="cuda", generator=gen)
    amax = torch.rand(n, device="cuda", generator=gen) * 3
    col = torch.rand(1536, device="cuda", generator=gen) * 1e-3
    ref = acc.float().mul_((amax / 127.0)[:, None]).mul_(col[None, :])
    assert torch.equal(int8_glue.dequant_rows_cols(acc, amax, col), ref)


def test_gate_up_scales_kept(int8_on):
    from solution import int8_glue

    gate = torch.nn.Module()
    sg, su = torch.rand(1536, device="cuda"), torch.rand(1536, device="cuda")
    a = int8_glue.gate_up_scales(gate, sg, su, int8_on._inverse)
    assert torch.equal(a[0], torch.maximum(sg, su)) and torch.equal(a[1], int8_on._inverse(torch.maximum(sg, su) * 127.0))
    b = int8_glue.gate_up_scales(gate, sg, su, int8_on._inverse)
    assert b[0] is a[0] and b[1] is a[1]
    c = int8_glue.gate_up_scales(gate, sg.clone(), su, int8_on._inverse)  # another scale tensor: recomputed
    assert c[0] is not a[0] and torch.equal(c[0], a[0])


def test_mlp_backward_glue_equals_torch_glue(int8_on, monkeypatch):
    """The whole int8 MLP backward with the glue kernels gives the torch glue's dL/dx bit for bit."""
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import int8_glue

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()
    gen = torch.Generator("cuda").manual_seed(7)
    x = torch.randn(1, 832, 1536, device="cuda", generator=gen)
    grad = (torch.randn(1, 832, 1536, device="cuda", generator=gen) * 1e-2).half()
    results = []
    for on in (False, True):
        monkeypatch.setattr(int8_glue, "ON", on)
        xg = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(xg)
        out.backward(grad.to(out.dtype))
        results.append(xg.grad)
    assert torch.equal(results[0], results[1])
