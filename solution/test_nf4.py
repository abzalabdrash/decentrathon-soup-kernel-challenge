"""The Triton NF4 dequantization against bitsandbytes, and the patched MatMul4Bit against the original."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# (in_features, out_features) of the task's 4-bit layers: q/o, k/v, gate/up, down
SHAPES = [(1536, 1536), (1536, 256), (1536, 8960), (8960, 1536)]


def _layer(k, n, seed, quant_type="nf4", compress_statistics=True):
    """A bnb Linear4bit holding a random fp16 weight, quantized by the move to CUDA."""
    import bitsandbytes as bnb

    torch.manual_seed(seed)
    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16, quant_type=quant_type,
                              compress_statistics=compress_statistics)
    weight = (torch.randn(n, k) * 0.02).half()
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type=quant_type,
                                     compress_statistics=compress_statistics, module=layer)
    return layer.cuda()


@pytest.fixture
def launches(monkeypatch):
    """Counts the Triton kernel launches of the patched MatMul4Bit."""
    from solution import nf4

    calls = []
    launch = nf4._launch

    def counted(*args):
        calls.append(args[2])
        return launch(*args)

    monkeypatch.setattr(nf4, "_launch", counted)
    return calls


def _original_matmul4bit(monkeypatch):
    """Makes bnb.matmul_4bit apply the unpatched MatMul4Bit (for the reference run)."""
    from bitsandbytes.autograd import _functions as bnb_autograd

    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    cls = bnb_autograd.MatMul4Bit

    class Original(torch.autograd.Function):
        forward = staticmethod(cls._soup_original_forward)
        backward = staticmethod(cls._soup_original_backward)

    monkeypatch.setattr(bnb_autograd, "MatMul4Bit", Original)
    return lambda: monkeypatch.setattr(bnb_autograd, "MatMul4Bit", cls)


# (96, 40): 60 blocks, a partial program and a partial 256-block scale; (64, 13): 13 blocks
@pytest.mark.parametrize("k,n", SHAPES + [(96, 40), (64, 13)])
def test_dequant_bit_exact(k, n):
    import bitsandbytes.functional as bnb_functional

    from solution.nf4 import _fast_path, dequantize_nf4

    layer = _layer(k, n, seed=k + n)
    qs = layer.weight.quant_state
    assert _fast_path(layer.weight, qs, torch.float16)
    ours = dequantize_nf4(layer.weight, qs)
    ref = bnb_functional.dequantize_4bit(layer.weight, qs)
    assert ours.shape == ref.shape == (n, k) and ours.dtype == ref.dtype == torch.float16
    assert torch.equal(ours, ref)


@pytest.mark.parametrize("quant_type,compress_statistics,k,n", [
    ("nf4", False, 256, 64), ("fp4", True, 256, 64),
    ("nf4", True, 100, 7), ("nf4", True, 101, 7),  # N * K not a multiple of 64 (odd, too)
])
def test_dequant_other_formats_fall_back(quant_type, compress_statistics, k, n):
    import bitsandbytes.functional as bnb_functional

    from solution.nf4 import _fast_path, dequantize_nf4

    layer = _layer(k, n, seed=1, quant_type=quant_type, compress_statistics=compress_statistics)
    qs = layer.weight.quant_state
    assert not _fast_path(layer.weight, qs, torch.float16)
    assert torch.equal(dequantize_nf4(layer.weight, qs), bnb_functional.dequantize_4bit(layer.weight, qs))


@pytest.mark.parametrize("rows", [(1536, 256, 256), (8960, 8960)])  # q+k+v, gate+up
def test_dequant_into_row_slices_of_one_buffer(rows, launches):
    import bitsandbytes.functional as bnb_functional

    from solution.nf4 import dequantize_nf4

    layers = [_layer(1536, n, seed=i) for i, n in enumerate(rows)]
    buf = torch.full((sum(rows) + 3, 1536), float("nan"), device="cuda", dtype=torch.float16)
    start = 0
    for layer, n in zip(layers, rows):
        view = buf[start:start + n]
        assert dequantize_nf4(layer.weight, layer.weight.quant_state, out=view) is view
        start += n
    assert len(launches) == len(rows)  # straight into the buffer, no copies
    ref = torch.cat([bnb_functional.dequantize_4bit(layer.weight, layer.weight.quant_state) for layer in layers])
    assert torch.equal(buf[:start], ref)
    assert torch.isnan(buf[start:]).all()


def test_dequant_out_strided_fallback_and_errors():
    import bitsandbytes.functional as bnb_functional

    from solution.nf4 import dequantize_nf4

    layer = _layer(256, 64, seed=2)
    qs = layer.weight.quant_state
    ref = bnb_functional.dequantize_4bit(layer.weight, qs)
    wide = torch.zeros(64, 300, device="cuda", dtype=torch.float16)
    dequantize_nf4(layer.weight, qs, out=wide[:, 10:266])  # not contiguous: written through a copy
    assert torch.equal(wide[:, 10:266], ref) and not wide[:, :10].any() and not wide[:, 266:].any()

    other = _layer(256, 64, seed=2, compress_statistics=False)  # bitsandbytes' path, then a copy
    out = torch.empty(64, 256, device="cuda", dtype=torch.float16)
    dequantize_nf4(other.weight, other.weight.quant_state, out=out)
    assert torch.equal(out, bnb_functional.dequantize_4bit(other.weight, other.weight.quant_state))

    with pytest.raises(ValueError):
        dequantize_nf4(layer.weight, qs, out=torch.empty(65, 256, device="cuda", dtype=torch.float16))
    with pytest.raises(ValueError):
        dequantize_nf4(layer.weight, qs, dtype=torch.float16,
                       out=torch.empty(64, 256, device="cuda", dtype=torch.float32))


def _forward_backward(layer, x, autocast):
    x = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16, enabled=autocast):
        out = layer(x)
    grad = torch.randn(out.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(7))
    out.float().backward(grad.to(out.dtype))
    return out.detach(), x.grad


@pytest.mark.parametrize("autocast", [True, False])
@pytest.mark.parametrize("k,n", SHAPES)
def test_patched_layer_matches_original(k, n, autocast, monkeypatch, launches):
    layer = _layer(k, n, seed=k * 3 + n)
    x = torch.randn(2048, k, device="cuda", dtype=torch.float32)

    restore = _original_matmul4bit(monkeypatch)
    ref_out, ref_grad = _forward_backward(layer, x, autocast)
    restore()
    assert launches == []

    out, grad = _forward_backward(layer, x, autocast)
    assert launches == [torch.float16, torch.float16]  # forward and backward both on the kernel
    assert out.dtype == ref_out.dtype and grad.dtype == ref_grad.dtype == torch.float32
    torch.testing.assert_close(out, ref_out, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(grad, ref_grad, rtol=1e-3, atol=1e-3)
    # same dequantized weight, same cuBLAS calls: the results are identical
    assert torch.equal(out, ref_out) and torch.equal(grad, ref_grad)


def test_patched_layer_small_batch_and_3d_input(monkeypatch):
    """Few tokens: where bitsandbytes runs its own fused GEMM the original forward stays; the
    result is the original's either way."""
    layer = _layer(1536, 1536, seed=5)
    x = torch.randn(2, 8, 1536, device="cuda")
    restore = _original_matmul4bit(monkeypatch)
    ref_out, ref_grad = _forward_backward(layer, x, True)
    restore()
    out, grad = _forward_backward(layer, x, True)
    assert torch.equal(out, ref_out) and torch.equal(grad, ref_grad)


def test_unexpected_quant_state_takes_original_path(monkeypatch, launches):
    """An object that is not a bitsandbytes QuantState (but quacks like one, so the original code
    works with it) goes to the original forward and backward without touching the kernel."""
    import bitsandbytes as bnb

    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    layer = _layer(1536, 256, seed=11)
    qs = layer.weight.quant_state
    fake = types.SimpleNamespace(**{name: getattr(qs, name) for name in (
        "absmax", "shape", "code", "dtype", "blocksize", "quant_type", "offset", "state2", "nested")})
    x = torch.randn(2048, 1536, device="cuda", dtype=torch.float16, requires_grad=True)

    out = bnb.matmul_4bit(x, layer.weight, quant_state=fake)
    out.float().sum().backward()
    assert launches == []

    x_ref = x.detach().clone().requires_grad_(True)
    ref = bnb.matmul_4bit(x_ref, layer.weight, quant_state=qs)
    ref.float().sum().backward()
    assert launches == [torch.float16, torch.float16]
    assert torch.equal(out, ref) and torch.equal(x.grad, x_ref.grad)


def test_patch_is_idempotent_and_keeps_originals():
    from bitsandbytes.autograd import _functions as bnb_autograd

    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    forward = bnb_autograd.MatMul4Bit.forward
    original = bnb_autograd.MatMul4Bit._soup_original_forward
    patch_bnb_matmul4bit()
    assert bnb_autograd.MatMul4Bit.forward is forward
    assert bnb_autograd.MatMul4Bit._soup_original_forward is original is not forward
