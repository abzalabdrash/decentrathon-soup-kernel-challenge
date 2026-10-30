"""adam_host.py: AdamW's moments on the host between steps give bit-identical parameters and
moments, and are off the GPU between steps."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _train(on, monkeypatch, steps=4):
    from solution import adam_host

    monkeypatch.setattr(adam_host, "ON", on)
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(16, 1536, device="cuda")),
              torch.nn.Parameter(torch.randn(1536, 16, device="cuda"))]
    opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.01)
    g = torch.Generator("cuda").manual_seed(1)
    between = []
    with adam_host.moments_on_host():
        for _ in range(steps):
            for p in params:
                p.grad = torch.randn(p.shape, device="cuda", generator=g)
            opt.step()
            between.append(all(not opt.state[p]["exp_avg"].is_cuda for p in params))
    moments = [opt.state[p][name].clone() for p in params for name in ("exp_avg", "exp_avg_sq")]
    return [p.detach().clone() for p in params], moments, between


def test_same_bits_and_off_the_gpu_between_steps(monkeypatch):
    ref_p, ref_m, ref_between = _train(False, monkeypatch)
    got_p, got_m, got_between = _train(True, monkeypatch)
    assert not any(ref_between) and all(got_between)
    assert all(torch.equal(a, b) for a, b in zip(got_p, ref_p))
    assert all(m.is_cuda for m in got_m)  # put back on the GPU when the training returns
    assert all(torch.equal(a, b) for a, b in zip(got_m, ref_m))


def test_other_optimizers_untouched(monkeypatch):
    from solution import adam_host

    monkeypatch.setattr(adam_host, "ON", True)
    p = torch.nn.Parameter(torch.randn(8, 8, device="cuda"))
    opt = torch.optim.Adam([p], lr=1e-3)  # not AdamW
    with adam_host.moments_on_host():
        p.grad = torch.ones_like(p)
        opt.step()
        assert opt.state[p]["exp_avg"].is_cuda
