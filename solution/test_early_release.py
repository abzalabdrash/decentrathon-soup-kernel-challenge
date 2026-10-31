"""SOUP_SOLUTION_EARLY_RELEASE: the int8 MLP backward releases each 4-bit matrix's ring slot right after
its last read is queued (down before the first int8 GEMM, gate and up before the second), once each,
with bit-for-bit the same gradients."""

import pytest
import torch

from solution.test_fused_proj import _config, _quantize_mlp, patched  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture
def int8_on(monkeypatch):
    from solution import int8_bwd

    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    int8_bwd.compile_now()
    return int8_bwd


def _log_int_mm(monkeypatch, log):
    real = torch._int_mm

    def logged(*args, **kwargs):
        log.append("mm")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "_int_mm", logged)


def test_mlp_backward_release_order_and_values(int8_on, monkeypatch):
    from types import SimpleNamespace

    from transformers.models.qwen2 import modeling_qwen2 as mq

    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=31).cuda()
    plan = SimpleNamespace(gate=mlp.gate_proj, up=mlp.up_proj, down=mlp.down_proj, half=torch.float16)
    gen = torch.Generator("cuda").manual_seed(5)
    inter, d, tokens = 8960, 1536, 832
    gate_up = (torch.randn(2 * inter, tokens, device="cuda", generator=gen) * 2).half()
    grad16 = (torch.randn(tokens, d, device="cuda", generator=gen) * 1e-2).half()
    buffer = torch.empty(inter * d, dtype=torch.float16, device="cuda")
    ref = int8_on.mlp_backward(plan, gate_up.clone(), grad16, buffer)
    log = []
    _log_int_mm(monkeypatch, log)
    names = {id(plan.down): "down", id(plan.gate): "gate", id(plan.up): "up"}
    got = int8_on.mlp_backward(plan, gate_up.clone(), grad16, buffer, release=lambda b: log.append(names[id(b)]))
    assert torch.equal(got, ref)
    assert log == ["down", "mm", "gate", "up", "mm"]


class _Ring:
    """Stands in for weight_stream.py's type ring: records acquire / release in call order."""

    def __init__(self, log, names):
        self.log, self.names = log, names

    def acquire(self, base, step):
        self.log.append(("acquire", self.names.get(id(base), "?"), step))

    def release(self, base, step):
        self.log.append(("release", self.names.get(id(base), "?"), step))


@pytest.mark.parametrize("early", [False, True])
def test_fused_mlp_backward_releases_each_slot_once(int8_on, monkeypatch, early):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj, weight_stream

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=32).cuda().train()
    names = {id(mlp.down_proj): "down", id(mlp.gate_proj): "gate", id(mlp.up_proj): "up"}
    log = []
    ring = _Ring(log, names)
    monkeypatch.setattr(weight_stream, "mlp_ring", lambda *args: ring)
    monkeypatch.setitem(fused_proj._EARLY_RELEASE, "on", early)
    gen = torch.Generator("cuda").manual_seed(9)
    x = torch.randn(1, 832, 1536, device="cuda", generator=gen, requires_grad=True)
    grad = (torch.randn(1, 832, 1536, device="cuda", generator=gen) * 1e-2).half()
    with torch.autocast("cuda", dtype=torch.float16):
        out = mlp(x)
    del log[:]
    _log_int_mm(monkeypatch, log)
    out.backward(grad.to(out.dtype))
    bwd = [e for e in log if e == "mm" or e[2] == -1]
    if early:
        assert bwd == [("acquire", "down", -1), ("acquire", "gate", -1), ("acquire", "up", -1),
                       ("release", "down", -1), "mm", ("release", "gate", -1), ("release", "up", -1), "mm"]
    else:
        assert bwd == [("acquire", "down", -1), ("acquire", "gate", -1), ("acquire", "up", -1), "mm", "mm",
                       ("release", "down", -1), ("release", "gate", -1), ("release", "up", -1)]


def test_fused_mlp_same_gradients_with_and_without(int8_on, monkeypatch):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj, weight_stream

    monkeypatch.setenv("SOUP_SOLUTION_FUSED_GU_SPLIT", "1")
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=33).cuda().train()
    monkeypatch.setattr(weight_stream, "mlp_ring", lambda *args: _Ring([], {}))
    gen = torch.Generator("cuda").manual_seed(10)
    x = torch.randn(1, 640, 1536, device="cuda", generator=gen)
    grad = (torch.randn(1, 640, 1536, device="cuda", generator=gen) * 1e-2).half()
    grads = []
    for early in (False, True):
        monkeypatch.setitem(fused_proj._EARLY_RELEASE, "on", early)
        xg = x.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = mlp(xg)
        out.backward(grad.to(out.dtype))
        grads.append(xg.grad)
    assert torch.equal(grads[0], grads[1])
