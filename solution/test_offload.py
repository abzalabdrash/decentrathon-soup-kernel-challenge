"""The MLP memory modes of offload.py against the plain half path (mlp16.py): same outputs and
gradients, a lower peak, no leaks over steps; and the offload's stream ordering under delays."""

import contextlib
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

SLEEP = 100_000_000  # GPU cycles for torch.cuda._sleep: tens of milliseconds
TOKENS = 826


class Keep(torch.autograd.Function):
    """Returns x and saves `saved`; the gradient of x is grad * saved, so it shows what the backward
    got back for the saved tensor."""

    @staticmethod
    def forward(ctx, x, saved):
        ctx.save_for_backward(saved)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        (saved,) = ctx.saved_tensors
        return grad * saved, None


def activation(values):
    """A half tensor holding `values` that has a grad_fn and owns its storage, like gate/up."""
    return values.float().requires_grad_(True).half()


def leaf(n):
    return torch.zeros(n, device="cuda", dtype=torch.float16, requires_grad=True)


def close(actual, expected, rtol=1e-3):
    """Max abs difference within rtol of the largest reference value; returns the difference."""
    scale = expected.float().abs().max().item()
    diff = (actual.float() - expected.float()).abs().max().item()
    assert diff <= rtol * scale, f"max diff {diff:.3g} against scale {scale:.3g}"
    return diff


@pytest.fixture(scope="module")
def net():
    from solution.bench_offload import Net, step

    torch.manual_seed(0)
    model = Net(4).cuda()
    x = torch.randn(1, TOKENS, 1536, device="cuda", requires_grad=True)
    step(model, x, torch.randn_like(x))  # one-time allocations (cuBLAS workspace) out of the peaks
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def inputs():
    torch.manual_seed(1)
    x = torch.randn(1, TOKENS, 1536, device="cuda", requires_grad=True)
    return x, torch.randn_like(x)


def zero_grads(net, x):
    """Gradients allocated and zeroed, so that the peak over the baseline is the activations and
    the backward's temporaries (the synthetic weights are trainable, the task's are not)."""
    for t in (x, *net.parameters()):
        if t.grad is None:
            t.grad = torch.zeros_like(t)
        else:
            t.grad.zero_()


def run(net, x, grad, mode):
    """One forward + backward in `mode`: output, [x.grad, *param grads], peak MiB, offloader."""
    from solution.bench_offload import mode_context, step

    zero_grads(net, x)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with mode_context(net, mode) as offloader:
        out = step(net, x, grad)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    grads = [x.grad.clone()] + [p.grad.clone() for p in net.parameters()]
    return out.detach().clone(), grads, peak, offloader


@pytest.mark.parametrize("mode", ["offload", "recompute"])
def test_mode_matches_plain_and_lowers_peak(net, inputs, mode):
    x, grad = inputs
    ref_out, ref_grads, ref_peak, _ = run(net, x, grad, "none")
    out, grads, peak, offloader = run(net, x, grad, mode)
    diffs = [close(out, ref_out)] + [close(g, r) for g, r in zip(grads, ref_grads)]
    print(f"\n{mode}: peak {peak:.1f} MiB vs {ref_peak:.1f} MiB without, max diff {max(diffs):.3g}")
    if offloader is not None:
        # 4 blocks x (gate, up, and the down projection's input saved for its weight gradient)
        assert offloader.stats["offloaded"] == 12, offloader.summary()
        print(offloader.summary())
    # one saved (826, 8960) fp16 tensor is 14.1 MiB; 4 blocks save 8 of them as gate/up
    assert ref_peak - peak > 4 * 14.1


@pytest.mark.parametrize("mode", ["offload", "recompute"])
def test_five_steps_no_leak(net, inputs, mode):
    from solution.bench_offload import mode_context, step

    x, grad = inputs
    peaks, after, pinned = [], [], []
    with mode_context(net, mode) as offloader:
        for _ in range(5):
            zero_grads(net, x)
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            step(net, x, grad)
            torch.cuda.synchronize()
            peaks.append(torch.cuda.max_memory_allocated() - base)
            after.append(torch.cuda.memory_allocated() - base)
            if offloader is not None:
                assert offloader.host.live() == 0 and not offloader.inflight
                pinned.append((offloader.host.nbytes, len(offloader.host.slabs)))
    print(f"\n{mode}: peaks MiB {[round(p / 2**20, 1) for p in peaks]}, pinned {pinned}")
    assert max(peaks[1:]) <= peaks[0] + (1 << 20)
    assert max(after) <= 1 << 20
    if pinned:
        assert len(set(pinned)) == 1


@pytest.mark.parametrize("order", ["second_first", "first_first", "together"])
def test_offload_two_independent_branches(order):
    from solution.bench_offload import Net
    from solution.offload import offload_saved_tensors

    torch.manual_seed(2)
    nets = Net(2).cuda(), Net(2).cuda()
    xs = [torch.randn(1, TOKENS, 1536, device="cuda", requires_grad=True) for _ in range(2)]
    grads = [torch.randn_like(x) for x in xs]

    def go(offload):
        for x in xs:
            x.grad = None
        for p in [*nets[0].parameters(), *nets[1].parameters()]:
            p.grad = None
        with offload_saved_tensors() if offload else contextlib.nullcontext() as offloader:
            with torch.autocast("cuda", dtype=torch.float16):
                outs = [n(x) for n, x in zip(nets, xs)]
            if order == "second_first":
                outs[1].backward(grads[1])
                outs[0].backward(grads[0])
            elif order == "first_first":
                outs[0].backward(grads[0])
                outs[1].backward(grads[1])
            else:
                torch.autograd.backward(outs, grads)
        torch.cuda.synchronize()
        result = [x.grad.clone() for x in xs]
        result += [p.grad.clone() for p in [*nets[0].parameters(), *nets[1].parameters()]]
        return result, offloader

    ref, _ = go(False)
    got, offloader = go(True)
    for g, r in zip(got, ref):
        close(g, r)
    assert offloader.stats["offloaded"] == 12 and offloader.host.live() == 0
    print(f"\n{order}: {offloader.summary()}")
    del nets
    torch.cuda.empty_cache()


def test_slot_reuse_waits_for_a_read_on_another_stream():
    """A slot read back by an H2D copy on stream A (held up by a sleep) is reused for a tensor
    packed on stream B: its D2H must wait for that read, or the backward on A reads the new data.
    On a GPU whose copies in both directions share one queue in submission order (the WSL2 laptop
    card this was written on) the race cannot happen, so the bookkeeping is checked as well."""
    from solution.offload import Offloader

    n = 1 << 20

    def scenario(sleep):
        v1, v2 = torch.randn(n, device="cuda").half(), torch.randn(n, device="cuda").half()
        x1, x2 = leaf(n), leaf(n)
        off = Offloader(min_numel=1000, max_inflight=0, prefetch=0)
        stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
            with torch.cuda.stream(stream_a):
                y1 = Keep.apply(x1, activation(v1))
                ones = torch.ones_like(y1)
                torch.cuda._sleep(sleep)
                y1.backward(ones)
            assert off.host.live() == 0
            assert len(off.reads) == 1  # the event after the H2D read on stream A
            with torch.cuda.stream(stream_b):
                Keep.apply(x2, activation(v2))
            assert off.reads == []  # the offload stream waits for it before the next D2H
        torch.cuda.synchronize()
        assert off.stats["fetched"] == 1
        assert len(off.host.slots) == 1 and off.host.slots[0].start == 0  # the same slot again
        assert torch.equal(x1.grad, v1)

    scenario(1000)  # loads every kernel first: a lazy module load synchronizes the device
    scenario(SLEEP)


def test_gpu_tensor_kept_until_its_copy_is_done():
    """The D2H is held up by a sleep on the offload stream while the forward frees and reallocates
    memory: the saved tensor's block must not be handed out before the copy has read it."""
    from solution.offload import Offloader

    n = 1 << 22

    def scenario(sleep):
        v = torch.randn(n, device="cuda").half()
        x = leaf(n)
        off = Offloader(min_numel=1000, max_inflight=4, prefetch=0)
        torch.cuda.synchronize()
        with torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
            with torch.cuda.stream(off.stream):
                torch.cuda._sleep(sleep)
            y = Keep.apply(x, activation(v))  # the offloader holds the activation's only reference
            junk = [torch.full((n,), float("nan"), device="cuda", dtype=torch.float16)
                    for _ in range(4)]
            del junk
            off.close()  # waits for the copy, drops the GPU tensor: the backward reads the host copy
            y.backward(torch.ones_like(y))
        torch.cuda.synchronize()
        assert off.stats["fetched"] == 1
        assert torch.equal(x.grad, v)

    scenario(1000)
    scenario(SLEEP)


def test_prefetched_tensor_waits_for_its_copy():
    """Each prefetch H2D is queued behind a sleep on the offload stream: the backward must wait
    for it instead of reading the uninitialized (NaN-filled) block."""
    from solution.offload import Offloader

    n, k = 1 << 20, 6

    def scenario(sleep):
        values = [torch.randn(n, device="cuda").half() for _ in range(k)]
        xs = [leaf(n) for _ in range(k)]
        off = Offloader(min_numel=1000, max_inflight=0, prefetch=2)
        torch.cuda.synchronize()

        def delay(grad):
            with torch.cuda.stream(off.stream):
                torch.cuda._sleep(sleep)
            return grad

        with torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
            ys = [Keep.apply(x, activation(v)) for x, v in zip(xs, values)]
            for y in ys:
                y.register_hook(delay)
            junk = [torch.full((n,), float("nan"), device="cuda", dtype=torch.float16)
                    for _ in range(2 * k)]
            del junk
            torch.autograd.backward(ys, [torch.ones_like(y) for y in ys])
        torch.cuda.synchronize()
        assert off.stats["prefetch_hits"] == k - 1 and off.stats["fetched"] == 1, off.summary()
        for x, v in zip(xs, values):
            assert torch.equal(x.grad, v)

    scenario(1000)
    scenario(SLEEP)


def test_prepinned_slab_is_used():
    from solution import offload

    offload.prepin(offload.SLAB_BYTES >> 20).join()
    spare = len(offload._SPARE)
    assert spare >= 1
    off = offload.Offloader(min_numel=1000)
    n = 1 << 20
    v, x = torch.randn(n, device="cuda").half(), leaf(n)
    with torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack):
        y = Keep.apply(x, activation(v))
        y.backward(torch.ones_like(y))
    torch.cuda.synchronize()
    assert off.host.prepinned == 1 and len(offload._SPARE) == spare - 1
    assert torch.equal(x.grad, v)


def test_long_gpu_work_between_pack_and_unpack():
    from solution.offload import offload_saved_tensors

    n, k = 1 << 21, 8
    values = [torch.randn(n, device="cuda").half() for _ in range(k)]
    xs = [leaf(n) for _ in range(k)]
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16) / 45
    with offload_saved_tensors(min_numel=1000) as off:
        total = 0
        for x, v in zip(xs, values):
            total = total + Keep.apply(x, activation(v)).float().sum()
            for _ in range(20):  # GPU work after every pack
                a = (a @ a).clamp_(-1, 1)
        for _ in range(50):  # and between the forward and the backward
            a = (a @ a).clamp_(-1, 1)
        total.backward()
    torch.cuda.synchronize()
    assert off.stats["offloaded"] == k
    for x, v in zip(xs, values):
        assert torch.equal(x.grad, v)


# ----------------------------------------------------------------------------------------------
# recompute on real Qwen2MLP and on 4-bit layers


def plain_mlp16(mlp, x):
    """mlp16.py's forward for an fp32 input under fp16 autocast."""
    from solution.swiglu import swiglu

    x16 = x.to(torch.float16)
    return mlp.down_proj(swiglu(mlp.gate_proj(x16), mlp.up_proj(x16))).to(x.dtype)


@pytest.fixture
def restore_classes():
    """Undo class patches made by a test (other test files patch the same classes)."""
    from soup_cli.trainer import sft
    from transformers.models.qwen2 import modeling_qwen2

    targets = ((sft.SFTTrainerWrapper, ("train", "_soup_mlp_offload", "_soup_offload_previous_train")),
               (modeling_qwen2.Qwen2MLP, ("forward", "_soup_mlp_recompute", "_soup_original_forward",
                                          "_soup_recompute_previous")))
    saved = [(cls, name, name in cls.__dict__, cls.__dict__.get(name))
             for cls, names in targets for name in names]
    yield
    for cls, name, present, value in saved:
        if present:
            setattr(cls, name, value)
        elif name in cls.__dict__:
            delattr(cls, name)


def saved_numels(fn):
    """Run fn() under hooks that record the numel of every saved tensor."""
    numels = []

    def pack(t):
        numels.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = fn()
    return result, numels


def test_qwen2_mlp_recompute_matches_original(restore_classes):
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution.offload import patch_qwen2_mlp_recompute

    patch_qwen2_mlp_recompute()
    hf_forward = Qwen2MLP._soup_original_forward
    torch.manual_seed(0)
    hidden, inter, rows = 256, 1024, 2 * 37
    mlp = Qwen2MLP(Qwen2Config(hidden_size=hidden, intermediate_size=inter)).cuda().half()
    x0 = torch.randn(2, 37, hidden, device="cuda")
    grad = torch.randn(2, 37, hidden, device="cuda")

    def go(forward):
        mlp.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)

        def fwd():
            with torch.autocast("cuda", dtype=torch.float16):
                return forward(mlp, x)

        out, numels = saved_numels(fwd)
        out.backward(grad.to(out.dtype))
        return out.detach().float(), [x.grad] + [p.grad for p in mlp.parameters()], numels

    out, grads, numels = go(Qwen2MLP.forward)
    ref_out, ref_grads, ref_numels = go(plain_mlp16)
    hf_out, hf_grads, _ = go(hf_forward)
    assert out.dtype == torch.float32
    close(out, ref_out)
    for g, r in zip(grads, ref_grads):
        close(g, r)
    close(out, hf_out, rtol=2e-2)
    close(grads[0], hf_grads[0], rtol=5e-2)
    # gate, up and the SwiGLU output (saved by the trainable fp16 down_proj) vs that output only
    assert ref_numels.count(rows * inter) == 3 and numels.count(rows * inter) == 1
    assert numels.count(rows * hidden) >= 1  # the half input, saved by the recompute

    # fake modules and CPU inputs take the previous forward
    cpu_mlp = Qwen2MLP(Qwen2Config(hidden_size=32, intermediate_size=64))
    out_cpu = cpu_mlp(torch.randn(3, 32))
    assert out_cpu.shape == (3, 32)


def test_recompute_and_offload_on_frozen_nf4_layers():
    pytest.importorskip("bitsandbytes")
    from solution.bench_offload import Block
    from solution.offload import offload_saved_tensors

    torch.manual_seed(0)
    hidden, inter, rows = 256, 1024, 600
    block = Block(hidden=hidden, intermediate=inter, bnb=True).cuda()
    x0 = torch.randn(1, rows, hidden, device="cuda")
    grad = torch.randn(1, rows, hidden, device="cuda")

    def go(mode):
        x = x0.clone().requires_grad_(True)
        block.recompute = mode == "recompute"
        ctx = (offload_saved_tensors(min_numel=rows * inter) if mode == "offload"
               else contextlib.nullcontext())

        def fwd():
            with torch.autocast("cuda", dtype=torch.float16):
                return block(x)

        with ctx as offloader:
            out, numels = saved_numels(fwd) if mode != "offload" else (fwd(), [])
            out.backward(grad)
        block.recompute = False
        return out.detach(), x.grad, numels, offloader

    ref_out, ref_gx, ref_numels, _ = go("none")
    out, gx, numels, _ = go("recompute")
    off_out, off_gx, _, offloader = go("offload")
    for o, g in ((out, gx), (off_out, off_gx)):
        close(o, ref_out)
        close(g, ref_gx)
    print(f"\nNF4 recompute exact: out {torch.equal(out, ref_out)}, grad {torch.equal(gx, ref_gx)}")
    assert ref_numels.count(rows * inter) == 2  # gate and up; the 4-bit layers save no input
    assert numels.count(rows * inter) == 0 and numels.count(rows * hidden) == 1
    assert offloader.stats["offloaded"] == 2


# ----------------------------------------------------------------------------------------------
# the patches


def test_train_wrapper_offloads_only_for_a_real_cuda_trainer(restore_classes, monkeypatch):
    from soup_cli.trainer import sft
    from transformers import Trainer

    from solution import offload

    created = []

    class Recording(offload.Offloader):
        def __init__(self, **knobs):
            super().__init__(**knobs)
            created.append(self)

    def fake_train(self):
        n = offload.MIN_NUMEL
        x = leaf(n)
        Keep.apply(x, activation(torch.randn(n, device="cuda").half())).float().sum().backward()
        return "trained"

    monkeypatch.setattr(offload, "Offloader", Recording)
    monkeypatch.setattr(sft.SFTTrainerWrapper, "train", fake_train)
    offload.patch_soup_train_offload()
    assert sft.SFTTrainerWrapper.train is not fake_train

    assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=object())) == "trained"
    assert created == []

    class RealTrainer(Trainer):
        pass

    trainer = RealTrainer.__new__(RealTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=trainer)) == "trained"
    assert created == []  # model on the CPU

    trainer.model = trainer.model.cuda()
    assert sft.SFTTrainerWrapper.train(types.SimpleNamespace(trainer=trainer)) == "trained"
    assert len(created) == 1 and created[0].stats["offloaded"] == 1


def test_patch_mlp_memory_switch(restore_classes, monkeypatch):
    from soup_cli.trainer import sft
    from transformers.models.qwen2 import modeling_qwen2

    from solution.offload import patch_mlp_memory

    train, forward = sft.SFTTrainerWrapper.train, modeling_qwen2.Qwen2MLP.forward
    monkeypatch.setenv("SOUP_SOLUTION_MLP_MEMORY", "none")
    assert patch_mlp_memory() == "none"
    assert sft.SFTTrainerWrapper.train is train and modeling_qwen2.Qwen2MLP.forward is forward
    monkeypatch.setenv("SOUP_SOLUTION_MLP_MEMORY", "swap")
    with pytest.raises(ValueError):
        patch_mlp_memory()
    monkeypatch.setenv("SOUP_SOLUTION_MLP_MEMORY", "Recompute")
    assert patch_mlp_memory() == "recompute"
    assert modeling_qwen2.Qwen2MLP.forward is not forward and sft.SFTTrainerWrapper.train is train
    assert patch_mlp_memory("offload") == "offload"
    assert sft.SFTTrainerWrapper.train is not train
