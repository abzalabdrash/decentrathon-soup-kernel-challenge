"""offload_int8.py (task P1): the two kernels against plain torch, the offload's int8 handles through
all three ways back (still on the GPU, prefetched, fetched on demand) under GPU delays, the switch
point, and fused_proj's MLP with its (gate|up) offloaded as int8 against the half offload."""

import pytest
import torch

from solution.test_offload import restore_classes  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SLEEP = 100_000_000  # GPU cycles for torch.cuda._sleep: tens of milliseconds


@pytest.fixture
def int8_on(monkeypatch):
    from solution import offload_int8

    monkeypatch.setitem(offload_int8._STATE, "on", True)
    offload_int8.compile_now()
    yield offload_int8


def _reference(x, group):
    """(q, s, rebuilt) of x (rows, tokens) by plain torch: the kernels' arithmetic."""
    rows, tokens = x.shape
    xf = x.float().view(rows // group, group, tokens)
    amax = xf.abs().amax(1)
    inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
    q = torch.floor(xf * inv[:, None, :] + 0.5).clamp_(-127, 127)
    s = amax / 127.0
    return q.view(rows, tokens).to(torch.int8), s, (q * s[:, None, :]).view(rows, tokens)


@pytest.mark.parametrize("tokens", [832, 72, 1000])
def test_kernels_match_torch(int8_on, tokens):
    g = torch.Generator("cuda").manual_seed(tokens)
    x = (torch.randn(2 * 8960, tokens, device="cuda", generator=g) * 3).half()
    x[:, -8:] = 0  # padding tokens: scale 0, zeros
    x[5, 3] = 60.0  # an outlier only moves its own group's scale
    wire, codec = int8_on.encode(x)
    assert wire.dtype == torch.uint8 and wire.numel() == int8_on.wire_bytes(*x.shape)
    assert wire.numel() < 0.53 * x.numel() * 2
    q, s = int8_on.views(wire, *x.shape)
    rq, rs, rebuilt = _reference(x, int8_on.GROUP)
    assert torch.equal(s, rs)
    # the kernel's x * (127 / amax) may round differently from torch's at an exact .5 boundary
    diff = (q.int() - rq.int()).abs()
    assert diff.max().item() <= 1 and (diff > 0).float().mean().item() < 1e-4
    out = codec.decode(wire)
    assert out.dtype == x.dtype and out.shape == x.shape
    assert torch.equal(out, (q.float().view(-1, int8_on.GROUP, tokens) * s[:, None, :]).view_as(x).half())
    # at most half a step of the token's group scale off (plus the half rounding of the result)
    step = s.repeat_interleave(int8_on.GROUP, dim=0)
    err = (out.float() - x.float()).abs()
    assert torch.all(err <= 0.5 * step + x.float().abs() * 2 ** -10 + 1e-7)
    assert torch.all(out[:, -8:] == 0)
    rel = ((out.float() - x.float()).norm() / x.float().norm()).item()
    print(f"\nint8 (gate|up) at {tokens} tokens, group {int8_on.GROUP}: relative error {rel:.4f}")
    assert rel < 0.01


def _roundtrip(module, v):
    wire, codec = module.encode(v)
    return codec.decode(wire)


class Keep(torch.autograd.Function):
    """Returns x and saves `saved`; the gradient of x is grad * saved."""

    @staticmethod
    def forward(ctx, x, saved):
        ctx.save_for_backward(saved)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        (saved,) = ctx.saved_tensors
        return grad * saved, None


def _spare(t, kind):
    """What the fused MLP passes as offload_early's spare: its hidden activation ((rows / 2, tokens)
    half, exactly the int8 size), or a tensor too small to hold it (an int8 buffer is allocated)."""
    rows, tokens = t.shape
    return torch.empty((rows // 2, tokens) if kind == "hidden" else (8, tokens), dtype=t.dtype,
                       device=t.device)


def _chain(values, early, delay, spare="hidden"):
    """x through one Keep per tensor of `values` (each made fresh on the GPU, offload_early's two
    calls first, as the fused MLP makes them), with GPU sleeps between, inside offload_saved_tensors;
    returns x.grad and the stats."""
    from solution.offload import offload_early, offload_saved_tensors

    x = torch.ones(values[0].shape, device="cuda", dtype=torch.float16, requires_grad=True)
    with offload_saved_tensors(**early) as off:
        h = x
        for v in values:
            t = v.clone()  # owns its storage, made on the compute stream
            offload_early(t)
            if delay:
                torch.cuda._sleep(SLEEP // 4)
            offload_early(t, spare=_spare(t, spare))
            h = Keep.apply(h, t)
            del t
        if delay:
            torch.cuda._sleep(SLEEP)
        h.backward(torch.ones_like(h))
    torch.cuda.synchronize()
    return x.grad, off.stats


@pytest.mark.parametrize("spare", ["hidden", "small"])
@pytest.mark.parametrize("knobs", [{}, {"prefetch": 0}, {"max_inflight": 4}])
def test_int8_handles_through_every_way_back(int8_on, knobs, spare):
    """Six (gate|up)-shaped tensors offloaded early as int8: the backward gets each one rebuilt from
    its int8 form (the product of all of them, exactly), whichever way it came back."""
    tokens = 256
    g = torch.Generator("cuda").manual_seed(3)
    values = [(1 + 0.05 * torch.randn(2 * 8960, tokens, device="cuda", generator=g)).half()
              for _ in range(6)]
    rebuilt = [_roundtrip(int8_on, v) for v in values]
    expected = torch.ones_like(values[0])
    for r in reversed(rebuilt):  # the backward multiplies from the last Keep to the first
        expected = expected * r
    knobs = {"min_numel": 1, **knobs}
    grad, stats = _chain(values, knobs, delay=True, spare=spare)
    assert stats["int8"] == 6 and stats["early"] == 6
    back = stats["still_on_gpu"] + stats["prefetch_hits"] + stats["fetched"]
    assert back == 6
    if knobs.get("prefetch") == 0:
        assert stats["fetched"] >= 1 and stats["prefetch_hits"] == 0
    else:
        assert stats["prefetch_hits"] >= 1
    assert torch.equal(grad, expected)


def test_written_after_quantization_goes_half(int8_on):
    """A tensor written in place between offload_early and its save is not taken from the int8 handle
    (its version moved): the save offloads it again in half, the backward gets the written values."""
    from solution.offload import offload_early, offload_saved_tensors

    x = torch.ones(2 * 8960, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    t = (torch.randn(2 * 8960, 64, device="cuda") * 0.3).half()
    with offload_saved_tensors(min_numel=1) as off:
        offload_early(t)
        offload_early(t, spare=_spare(t, "hidden"))
        t.add_(1)
        h = Keep.apply(x, t)
        h.backward(torch.ones_like(h))
    torch.cuda.synchronize()
    assert off.stats["int8"] == 1 and off.stats["offloaded"] == 2
    assert torch.equal(x.grad, t)


def test_takeover_after_start_calls(int8_on, monkeypatch):
    monkeypatch.setattr(int8_on, "START_CALLS", 3)
    monkeypatch.setitem(int8_on._STATE, "calls", 0)
    monkeypatch.setitem(int8_on._STATE, "thread", None)
    ready = int8_on._READY.is_set()
    int8_on._READY.clear()
    try:
        t = torch.zeros(2 * 8960, 32, device="cuda", dtype=torch.float16)
        assert not int8_on.applies(t[:, :16])  # not contiguous: never counted
        assert not int8_on.applies(t.float())
        assert [int8_on.applies(t) for _ in range(5)] == [False] * 3 + [True] * 2
        assert int8_on._STATE["calls"] == 5 and not int8_on._STATE["thread"].is_alive()
    finally:
        if ready:
            int8_on._READY.set()


def test_off_switch_keeps_the_half_offload(monkeypatch):
    from solution import offload_int8

    monkeypatch.setitem(offload_int8._STATE, "on", False)
    values = [(torch.randn(2 * 8960, 64, device="cuda") * 0.5 + 1).half() for _ in range(3)]
    grad, stats = _chain(values, {"min_numel": 1}, delay=False)
    assert stats["int8"] == 0 and stats["early"] == 3
    assert torch.equal(grad, values[2] * values[1] * values[0])


def test_fused_mlp_offloaded_as_int8(int8_on, monkeypatch, restore_classes):
    """fused_proj.py's MLP of the task's shapes inside offload_saved_tensors (the training path,
    EARLY): every layer's (gate|up) goes as int8; the forward is bit for bit the half offload's, the
    input gradient close to it (fp16 and int8 MLP backward)."""
    bnb = pytest.importorskip("bitsandbytes")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import int8_bwd, offload
    from solution.fused_proj import patch_qwen2_mlp_fused, set_mlp_recompute

    monkeypatch.setattr(offload, "EARLY", True)
    patch_qwen2_mlp_fused()
    set_mlp_recompute(False)
    torch.manual_seed(0)
    hidden, inter, rows, layers = 1536, 8960, 826, 3
    config = Qwen2Config(hidden_size=hidden, intermediate_size=inter)

    def nf4(n_in, n_out):
        layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                  compress_statistics=True, quant_type="nf4")
        layer.weight.requires_grad_(False)
        return layer

    mlps = []
    for _ in range(layers):
        with torch.device("meta"):
            mlp = Qwen2MLP(config)
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = nf4(hidden, inter), nf4(hidden, inter), \
            nf4(inter, hidden)
        mlps.append(mlp.cuda())
    x0 = torch.randn(1, rows, hidden, device="cuda")
    grad = torch.randn_like(x0) * 1e-2

    def go(on, bwd_int8):
        monkeypatch.setitem(int8_on._STATE, "on", on)
        monkeypatch.setitem(int8_bwd._STATE, "on", bwd_int8)
        x = x0.clone().requires_grad_(True)
        with offload.offload_saved_tensors(weights=[p for m in mlps for p in m.parameters()]) as off, \
                torch.autocast("cuda", dtype=torch.float16):
            h = x
            for mlp in mlps:
                h = h + mlp(h)
            h.backward(grad)
        torch.cuda.synchronize()
        return h.detach(), x.grad, off.stats

    int8_bwd.compile_now()
    for bwd_int8 in (False, True):
        ref_out, ref_gx, ref_stats = go(False, bwd_int8)
        out, gx, stats = go(True, bwd_int8)
        assert ref_stats["int8"] == 0 and stats["int8"] == layers and stats["early"] == layers
        assert torch.equal(out, ref_out)
        err = ((gx - ref_gx).norm() / ref_gx.norm()).item()
        print(f"\nfused MLP x{layers}, int8 backward {bwd_int8}: dL/dx relative error of the int8 "
              f"offload against the half one {err:.4f}")
        assert err < 0.02


def test_no_record_stream_and_no_cpu_wait_on_the_int8_path(int8_on, monkeypatch):
    """The int8 path keeps the offload's stream rules on the training path (one stream): no
    record_stream, no CPU wait on an event; the values come back as the kernels rebuild them."""
    import threading

    from solution import offload

    calls = []
    me = threading.get_ident()
    original_sync, original_record = torch.cuda.Event.synchronize, torch.Tensor.record_stream

    def record_stream(self, stream):
        if threading.get_ident() == me:
            calls.append("record_stream")
        return original_record(self, stream)

    def synchronize(self):
        if threading.get_ident() == me:
            calls.append("synchronize")
        return original_sync(self)

    k = 5
    values = [(torch.randn(2 * 8960, 128, device="cuda") * 0.5 + 1).half() for _ in range(k)]
    xs = [torch.zeros_like(v, requires_grad=True) for v in values]
    off = offload.Offloader(min_numel=1000, max_inflight=1, prefetch=1)
    torch.cuda.synchronize()
    monkeypatch.setitem(offload._ACTIVE, "offloader", off)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    monkeypatch.setattr(torch.cuda.Event, "synchronize", synchronize)
    with torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
        ys = []
        for x, v in zip(xs, values):
            t = v.clone()
            offload.offload_early(t)
            offload.offload_early(t, spare=_spare(t, "hidden"))
            ys.append(Keep.apply(x, t))
            del t
        torch.cuda._sleep(SLEEP)  # the copies are still queued when the backward starts
        sum(y.float().sum() for y in ys).backward()
    assert calls == []
    monkeypatch.undo()
    off.close()
    torch.cuda.synchronize()
    s = off.stats
    assert s["int8"] == k and s["prefetch_hits"] == k - 1, off.summary()
    for x, v in zip(xs, values):
        assert torch.equal(x.grad, _roundtrip(int8_on, v))


def test_raw_unpack_feeds_the_int8_backward_directly(int8_on, monkeypatch, restore_classes):
    """SOUP_SOLUTION_OFFLOAD_INT8_FUSED: with the int8 MLP backward the unpack hands over the int8
    form and int8_swiglu_bwd_kernel reads it; the result is bit for bit that of the rebuild pass."""
    bnb = pytest.importorskip("bitsandbytes")
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution import int8_bwd, offload
    from solution.fused_proj import patch_qwen2_mlp_fused, set_mlp_recompute

    monkeypatch.setattr(offload, "EARLY", True)
    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    int8_bwd.compile_now()
    patch_qwen2_mlp_fused()
    set_mlp_recompute(False)
    torch.manual_seed(1)
    config = Qwen2Config(hidden_size=1536, intermediate_size=8960)
    layers = []
    for _ in range(2):
        with torch.device("meta"):
            mlp = Qwen2MLP(config)
        for name, (n_in, n_out) in (("gate_proj", (1536, 8960)), ("up_proj", (1536, 8960)),
                                    ("down_proj", (8960, 1536))):
            layer = bnb.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
            layer.weight.requires_grad_(False)
            setattr(mlp, name, layer)
        layers.append(mlp.cuda())
    x0 = torch.randn(1, 640, 1536, device="cuda")
    grad = torch.randn_like(x0) * 1e-2
    decoded = []
    original = int8_on.Codec.decode
    monkeypatch.setattr(int8_on.Codec, "decode", lambda self, w: decoded.append(1) or original(self, w))

    def go(fused):
        monkeypatch.setitem(int8_on.RAW, "on", fused)
        decoded.clear()
        x = x0.clone().requires_grad_(True)
        with offload.offload_saved_tensors(weights=[p for m in layers for p in m.parameters()]) as off, \
                torch.autocast("cuda", dtype=torch.float16):
            h = x
            for mlp in layers:
                h = h + mlp(h)
            h.backward(grad)
        torch.cuda.synchronize()
        assert off.stats["int8"] == len(layers)
        return x.grad, len(decoded), off.stats["still_on_gpu"]

    ref, rebuilt, _ = go(False)
    got, rebuilt_fused, on_gpu = go(True)
    # a layer whose int8 pair is still on the GPU (q in its hidden's storage) is rebuilt: autograd
    # takes a tensor from the unpack
    assert rebuilt == len(layers) and rebuilt_fused == on_gpu < len(layers)
    assert torch.equal(got, ref)
