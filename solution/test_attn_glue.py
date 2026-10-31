"""SOUP_SOLUTION_ATTN_GLUE (fused_proj.py): the attention recompute expands k and v once and hands the
expanded pair to the attention backward. A decoder layer at the task's shapes gives the same outputs
and gradients bit for bit as without it, rebuilds the forward's tensors bit for bit, and the
backward's allocated peak does not grow."""

import pytest
import torch

from solution.test_fused_proj import _lora_params, patched  # noqa: F401 - autouse patch fixture
from solution.test_recompute import _layer_forward, _set, decoder_1p5b, modes  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def glue():
    from solution import fused_proj

    saved = fused_proj._ATTN["glue"]
    yield fused_proj._ATTN
    fused_proj._ATTN["glue"] = saved


# 1 440 and 830 tokens (the task's long and typical steps), and lengths off the kernel's alignment
@pytest.mark.parametrize("lengths", [[512, 512, 416], [212, 187, 310, 121], [97, 33, 501, 5]])
def test_layer_bitwise_and_peak(modes, glue, decoder_1p5b, lengths):
    from solution.rmsnorm import add_rmsnorm

    layer, rotary = decoder_1p5b
    next_norm = torch.ones(1536, device="cuda")

    def run(on, check=False):
        glue["glue"] = on
        _set(modes, recompute="all", split=True, check=check)
        for p in layer.parameters():
            p.grad = None
        torch.cuda.synchronize()
        torch.manual_seed(7)
        (mlp_out, residual), (hidden, branch_in) = _layer_forward(layer, rotary, lengths)
        s_next, y_next = add_rmsnorm(residual, mlp_out, next_norm, 1e-6)
        out = (mlp_out.detach().clone(), residual.detach().clone())
        del mlp_out, residual
        g = torch.Generator("cuda").manual_seed(8)
        grads = [torch.randn(s_next.shape, device="cuda", generator=g),
                 torch.randn(y_next.shape, device="cuda", generator=g)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        torch.autograd.backward([s_next, y_next], grads)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        del s_next, y_next, grads
        return {"peak": peak, "out": [t.cpu() for t in out],
                "grads": [t.cpu() for t in [hidden.grad, branch_in.grad]
                          + [p.grad for p in _lora_params(layer)]]}

    run(False)  # buffers, caches, the allocator's pool
    run(True, check=True)
    assert modes[0]["mismatches"] == []  # x16, q, expanded k and v, output, lse, branch
    off, off2, on = run(False), run(False), run(True)
    for a, b in zip(on["out"], off["out"]):
        assert torch.equal(a, b)
    # bit for bit wherever the kept path is itself repeatable (dQ is added with atomics)
    for a, b, c in zip(on["grads"], off["grads"], off2["grads"]):
        if torch.equal(b, c):
            assert torch.equal(a, b)
        else:
            assert (a.float() - b.float()).abs().max() <= (c.float() - b.float()).abs().max() * 4
    print(f"\n{sum(lengths)} tokens: backward peak {off['peak'] / 2 ** 20:.2f} -> "
          f"{on['peak'] / 2 ** 20:.2f} MiB")
    assert on["peak"] <= off["peak"]
