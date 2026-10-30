"""Night task N1b: the decoder layers' fp32 inputs offloaded to host memory with the MLP's activations
(offload.py, SOUP_SOLUTION_OFFLOAD_RESIDUAL): the same outputs and gradients, the inputs actually
leave the GPU, and the post-attention norm reads the residual from the input norm (no second saved
reference)."""

import pytest
import torch

from solution.test_fused_proj import _lora_params, _rel, patched  # noqa: F401 - autouse patches
from solution.test_recompute import _layer_forward, _set, decoder_1p5b, modes  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def residual_switch():
    from solution import offload

    saved = dict(offload._RESIDUAL)
    yield offload._RESIDUAL
    offload._RESIDUAL.clear()
    offload._RESIDUAL.update(saved)


def _step(layer, rotary, lengths, offload_on, residual_switch):
    from solution.offload import offload_saved_tensors

    residual_switch["on"] = offload_on
    for p in layer.parameters():
        p.grad = None
    with offload_saved_tensors(weights=list(layer.parameters())) as offloader:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.manual_seed(7)
        (mlp_out, residual), (hidden, branch_in) = _layer_forward(layer, rotary, lengths)
        torch.cuda.synchronize()
        held = torch.cuda.memory_allocated() - before
        g = torch.Generator("cuda").manual_seed(8)
        grads = [torch.randn(mlp_out.shape, device="cuda", generator=g).to(mlp_out.dtype),
                 torch.randn(residual.shape, device="cuda", generator=g)]
        torch.autograd.backward([mlp_out, residual], grads)
    stats = dict(offloader.stats)
    return {"held": held, "out": (mlp_out.detach(), residual.detach()), "stats": stats,
            "grads": [hidden.grad, branch_in.grad] + [p.grad.clone() for p in _lora_params(layer)]}


@pytest.mark.parametrize("mode", [None, "all"])
def test_residual_offload_same_results(modes, decoder_1p5b, residual_switch, mode):
    layer, rotary = decoder_1p5b
    lengths = [212, 187, 310, 117]
    _set(modes, recompute=mode, split=mode is not None, check=mode is not None)
    _step(layer, rotary, lengths, False, residual_switch)  # buffers, caches, the pool
    ref = _step(layer, rotary, lengths, False, residual_switch)
    got = _step(layer, rotary, lengths, True, residual_switch)
    assert modes[0]["mismatches"] == []
    # the layer's input; without the recompute also the post-attention sum it keeps itself
    assert ref["stats"].get("residuals", 0) == 0
    assert got["stats"].get("residuals", 0) == (1 if mode == "all" else 2)
    assert torch.equal(got["out"][0], ref["out"][0]) and torch.equal(got["out"][1], ref["out"][1])
    for a, b in zip(got["grads"], ref["grads"]):
        assert _rel(a, b) < 1e-3
    print(f"\nheld after the forward: {ref['held'] / 2**20:.1f} MiB kept, "
          f"{got['held'] / 2**20:.1f} MiB with the input offloaded")


def test_post_attention_norm_reads_the_residual_from_the_input_norm(modes, decoder_1p5b,
                                                                       residual_switch):
    from solution.rmsnorm import AddRMSNormFunction

    layer, rotary = decoder_1p5b
    _set(modes, recompute="all", split=True)
    residual_switch["on"] = False
    (mlp_out, residual), _ = _layer_forward(layer, rotary, [300, 200])
    # walk back from the post-attention norm's residual output to its node
    node = residual.grad_fn
    assert isinstance(node, AddRMSNormFunction._backward_cls)
    assert node.residual_node is not None  # the input norm's node, which keeps the sum
    assert len(node.saved_tensors) == 2  # weight and rstd only: no copy of the residual
    del mlp_out, residual
