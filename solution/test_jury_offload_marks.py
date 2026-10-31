"""A residual mark left over from before an offload context (A1 jury audit).

mark_offload / mark_residual mark a tensor's address right before its Function saves it; the pack
hook of the active offloader takes the mark. A mark nobody took (a forward without a graph, another
test) stayed in the global dict, and an unrelated fp32 tensor that later got the same address went
to host memory as a "layer input": the values come back right, but the offload counts change
(test_offload.py::test_mode_matches_plain_and_lowers_peak saw 13 tensors instead of 12 on the T4,
soup-a1-extremes2, after the new tests ran before it). A new context starts without stale marks.
"""

import pytest
import torch

from solution import offload

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_mark_from_before_the_context_is_not_taken():
    a = torch.randn(512, 512, device="cuda", requires_grad=True)
    w = torch.randn(512, 512, device="cuda", requires_grad=True)
    x = a * 2.0  # fp32, 1 MiB, not a leaf: kept on the GPU unless marked
    offload.mark_offload(x)  # stale: made outside any offload context
    with offload.offload_saved_tensors() as offloader:
        y = (x @ w).sum()  # mm saves x
    y.backward()
    assert offloader.stats["marked"] == 0 and offloader.stats["offloaded"] == 0
    assert not offload._RESIDUAL["marked"]


def test_mark_inside_the_context_is_still_taken():
    a = torch.randn(512, 512, device="cuda", requires_grad=True)
    w = torch.randn(512, 512, device="cuda", requires_grad=True)
    with offload.offload_saved_tensors() as offloader:
        x = a * 2.0
        offload.mark_offload(x)
        y = (x @ w).sum()
    y.backward()
    torch.cuda.synchronize()
    assert offloader.stats["marked"] == 1
    assert torch.allclose(a.grad, 2.0 * w.sum(1).expand(512, 512), rtol=1e-4, atol=1e-4)
