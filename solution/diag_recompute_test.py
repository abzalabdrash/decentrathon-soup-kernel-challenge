"""Diagnostic (measurement only, run explicitly: `pytest solution/diag_recompute_test.py -q -s`): one
1.5B decoder layer as in test_recompute's memory test; gradients of the kept path, the split sum and
the attention recompute with and without the check mode, with the layer's outputs held on the GPU or
not, each against the kept path, plus recompute check vs recompute bitwise."""

import pytest
import torch

from solution.test_fused_proj import _lora_params, _rel, patched  # noqa: F401 - autouse patches
from solution.test_recompute import _layer_forward, _set, decoder_1p5b, modes  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("lengths", [[212, 187, 310, 117], [500, 480, 447]])
def test_diag_recompute(modes, decoder_1p5b, lengths):
    from solution import fused_proj
    from solution.rmsnorm import add_rmsnorm

    layer, rotary = decoder_1p5b
    next_norm = torch.ones(1536, device="cuda")

    def run(mode, split, check=False, hold=False):
        _set(modes, recompute=mode, split=split, check=check)
        for p in layer.parameters():
            p.grad = None
        torch.manual_seed(7)
        (mlp_out, residual), (hidden, branch_in) = _layer_forward(layer, rotary, lengths)
        s_next, y_next = add_rmsnorm(residual, mlp_out, next_norm, 1e-6)
        keep = (mlp_out, residual) if hold else None
        del mlp_out, residual
        g = torch.Generator("cuda").manual_seed(8)
        grads = [torch.randn(s_next.shape, device="cuda", generator=g),
                 torch.randn(y_next.shape, device="cuda", generator=g)]
        torch.autograd.backward([s_next, y_next], grads)
        del keep
        return [t.detach().cpu() for t in [hidden.grad, branch_in.grad]
                + [p.grad for p in _lora_params(layer)]]

    run(None, False)
    kept = run(None, False)
    cases = {}
    for name, args in (("kept again", (None, False)), ("split", (None, True)),
                       ("recompute check, held", ("all", True, True, True)),
                       ("recompute check", ("all", True, True)),
                       ("recompute, held", ("all", True, False, True)),
                       ("recompute", ("all", True)), ("recompute again", ("all", True)),
                       ("recompute no split", ("all", False)),
                       ("split after recompute", (None, True)), ("kept last", (None, False))):
        cases[name] = run(*args)
        if args[0] is not None and len(args) > 2 and args[2]:
            print(f"\n  check-mode mismatches: {fused_proj._RECOMPUTE['mismatches']}")
    names = ["hidden", "branch_in"] + [f"lora{i}" for i in range(len(kept) - 2)]
    print(f"\n{sum(lengths)} tokens")
    for case, grads in cases.items():
        cells = []
        for nm, a, b in zip(names, grads, kept):
            cells.append(f"{nm} {_rel(a, b):.1e}{'=' if torch.equal(a, b) else ''}")
        print(f"  {case:24s} vs kept: " + ", ".join(cells), flush=True)
    a, b = cases["recompute check"], cases["recompute"]
    print("  recompute check vs recompute, bitwise:", [torch.equal(x, y) for x, y in zip(a, b)])
