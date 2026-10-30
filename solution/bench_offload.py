"""Time and peak memory of forward + backward with the MLP's gate/up saved (none), offloaded to
pinned host memory (offload) or recomputed (recompute), on a stack of MLP blocks shaped like the
task's layers; plus what the offload depends on: pinning speed, PCIe bandwidth, the cost of the
saved-tensor hooks themselves.

    python -m solution.bench_offload [--blocks 4] [--tokens 826] [--bnb] [--iters 30]

--bnb makes the projections frozen 4-bit NF4 bitsandbytes layers, as in the task (the down
projection then saves nothing); without it they are trainable fp16 nn.Linear (the down projection
saves its input for the weight gradient, so the offload also moves that tensor).
"""

import argparse
import contextlib
import statistics
import time

import torch

from .offload import Offloader, mlp_forward_recompute, offload_saved_tensors
from .swiglu import swiglu

HIDDEN, INTERMEDIATE, TOKENS = 1536, 8960, 826


class Block(torch.nn.Module):
    """gate_proj, up_proj, SwiGLU, down_proj in half, as mlp16.py runs Qwen2MLP; `recompute`
    switches to offload.mlp_forward_recompute."""

    def __init__(self, hidden=HIDDEN, intermediate=INTERMEDIATE, bnb=False):
        super().__init__()
        if bnb:
            import bitsandbytes as bnb_lib

            def make(n_in, n_out):
                layer = bnb_lib.nn.Linear4bit(n_in, n_out, bias=False, compute_dtype=torch.float16,
                                              compress_statistics=True, quant_type="nf4")
                layer.weight.requires_grad_(False)
                return layer
        else:
            def make(n_in, n_out):
                return torch.nn.Linear(n_in, n_out, bias=False, dtype=torch.float16)
        self.gate_proj = make(hidden, intermediate)
        self.up_proj = make(hidden, intermediate)
        self.down_proj = make(intermediate, hidden)
        self.recompute = False

    def forward(self, x):
        if self.recompute:
            return mlp_forward_recompute(self, x)
        x16 = x.to(torch.float16)
        return self.down_proj(swiglu(self.gate_proj(x16), self.up_proj(x16))).to(x.dtype)


class Net(torch.nn.Module):
    """Blocks on an fp32 residual stream, like the decoder layers."""

    def __init__(self, blocks=4, **kwargs):
        super().__init__()
        self.blocks = torch.nn.ModuleList(Block(**kwargs) for _ in range(blocks))

    def set_recompute(self, on):
        for block in self.blocks:
            block.recompute = on

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


def step(net, x, grad):
    with torch.autocast("cuda", dtype=torch.float16):
        out = net(x)
    out.backward(grad)
    return out


@contextlib.contextmanager
def mode_context(net, mode):
    net.set_recompute(mode == "recompute")
    try:
        if mode == "offload":
            with offload_saved_tensors() as offloader:
                yield offloader
        elif mode == "hooks_idle":  # the hooks on every saved tensor, nothing large enough to move
            with offload_saved_tensors(min_numel=1 << 62) as offloader:
                yield offloader
        else:
            yield None
    finally:
        net.set_recompute(False)


def compare(net, modes, tokens=TOKENS, iters=30, rounds=10, warmup=3):
    """{mode: (min ms, median ms, peak MiB over the baseline)} of forward + backward. The modes
    take turns in `rounds` rounds of iters / rounds iterations, so that other work on the GPU (a
    shared card) and clock changes fall on every mode alike; the offloaders live across rounds,
    as in training."""
    hidden = net.blocks[0].gate_proj.in_features
    x = torch.randn(1, tokens, hidden, device="cuda", requires_grad=True)
    grad = torch.randn(1, tokens, hidden, device="cuda")
    offloaders = {"offload": Offloader(), "hooks_idle": Offloader(min_numel=1 << 62)}

    @contextlib.contextmanager
    def turn(mode):
        net.set_recompute(mode == "recompute")
        off = offloaders.get(mode)
        try:
            with (torch.autograd.graph.saved_tensors_hooks(off.pack, off.unpack) if off
                  else contextlib.nullcontext()):
                yield
        finally:
            net.set_recompute(False)

    for mode in modes:
        with turn(mode):
            for _ in range(warmup):
                step(net, x, grad)
    times = {mode: [] for mode in modes}
    peaks = dict.fromkeys(modes, 0)
    for _ in range(rounds):
        for mode in modes:
            with turn(mode):
                for t in (x, *net.parameters()):
                    if t.grad is not None:
                        t.grad.zero_()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(max(1, iters // rounds)):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    step(net, x, grad)
                    end.record()
                    end.synchronize()
                    times[mode].append(start.elapsed_time(end))
                peaks[mode] = max(peaks[mode], torch.cuda.max_memory_allocated() - base)
    for off in offloaders.values():
        off.close()
    return {mode: (min(times[mode]), statistics.median(times[mode]), peaks[mode] / 2**20)
            for mode in modes}


def pinning_and_pcie():
    """Pinned allocation speed (GB/s) and D2H / H2D bandwidth (GB/s) for one gate tensor."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    slabs = [torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    pin = 0.5 / (time.perf_counter() - t0)
    del slabs
    gpu = torch.randn(TOKENS, INTERMEDIATE, device="cuda").half()
    host = torch.empty_like(gpu, device="cpu").pin_memory()
    rates = []
    for dst, src in ((host, gpu), (gpu, host)):
        for _ in range(3):
            dst.copy_(src, non_blocking=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            dst.copy_(src, non_blocking=True)
        end.record()
        end.synchronize()
        rates.append(gpu.numel() * 2 * 10 / start.elapsed_time(end) / 1e6)
    return pin, rates[0], rates[1]


def hook_overhead_us(n=2000):
    """Extra microseconds per saved tensor for forward + backward with the offload hooks on and
    nothing offloaded (a chain of n small multiplications, each saving two tensors)."""
    w = torch.randn(64, device="cuda", requires_grad=True)

    def run():
        x = torch.randn(64, device="cuda", requires_grad=True)
        y = x
        for _ in range(n):
            y = y * w
        y.sum().backward()
        torch.cuda.synchronize()

    idle = Offloader(min_numel=1 << 62)
    hooks = lambda: torch.autograd.graph.saved_tensors_hooks(idle.pack, idle.unpack)  # noqa: E731
    best = {contextlib.nullcontext: float("inf"), hooks: float("inf")}
    for _ in range(10):  # alternating, the minimum of each
        for ctx in best:
            with ctx():
                t0 = time.perf_counter()
                run()
                best[ctx] = min(best[ctx], time.perf_counter() - t0)
    return (best[hooks] - best[contextlib.nullcontext]) / (2 * n) * 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--bnb", action="store_true")
    args = parser.parse_args()

    print(torch.cuda.get_device_name(0), torch.__version__)
    pin, d2h, h2d = pinning_and_pcie()
    print(f"pinned allocation {pin:.2f} GB/s; D2H {d2h:.1f} GB/s, H2D {h2d:.1f} GB/s "
          f"({TOKENS} x {INTERMEDIATE} fp16)")
    print(f"saved-tensor hooks, nothing moved: {hook_overhead_us():.2f} us per saved tensor")
    torch.manual_seed(0)
    net = Net(args.blocks, bnb=args.bnb).cuda()
    kind = "frozen NF4" if args.bnb else "trainable fp16"
    print(f"{args.blocks} blocks ({kind}), {args.tokens} tokens, {args.iters} iterations")
    results = compare(net, ("none", "hooks_idle", "offload", "recompute"), args.tokens, args.iters)
    ref_min, ref_median = results["none"][:2]
    print(f"{'mode':<11} {'min ms':>8} {'vs none':>8} {'median ms':>10} {'vs none':>8} "
          f"{'peak MiB':>9}")
    for mode, (low, median, peak) in results.items():
        print(f"{mode:<11} {low:>8.2f} {low / ref_min - 1:>+8.1%} {median:>10.2f} "
              f"{median / ref_median - 1:>+8.1%} {peak:>9.1f}")


if __name__ == "__main__":
    main()
