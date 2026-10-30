"""Forward + backward time of a stack of decoder-layer-shaped blocks (the seven 4-bit projections of a
Qwen2.5-1.5B layer, LoRA on q and v, 826 tokens as an average step of the task) with the weights
resident and streamed through K = 2 and K = 3 slots; also the bandwidth of one layer's host -> GPU
copy and the setup time. Run on the target GPU: `python -m solution.bench_weight_stream [--layers 8]`.
"""

import argparse
import time

import torch

from solution.nf4 import patch_bnb_matmul4bit
from solution.test_weight_stream import _inputs, make_stack, run_stack
from solution.weight_stream import WeightStreamer


def _time_steps(blocks, x, weight, iters):
    def step():
        for p in blocks.parameters():
            p.grad = None
        out = run_stack(blocks, x.detach().requires_grad_(True))
        (out * weight).sum().backward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _copy_bandwidth(nbytes, iters=20):
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    device = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    device.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        device.copy_(host, non_blocking=True)
    torch.cuda.synchronize()
    seconds = (time.perf_counter() - start) / iters
    return seconds * 1e3, nbytes / seconds / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=826)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    patch_bnb_matmul4bit()
    blocks = make_stack(args.layers)
    x, weight = _inputs(tokens=args.tokens)
    print(f"{torch.cuda.get_device_name()}, {args.layers} layers, {args.tokens} tokens")

    rows = []
    torch.cuda.reset_peak_memory_stats()
    resident = _time_steps(blocks, x, weight, args.iters)
    rows.append(("resident", resident, torch.cuda.max_memory_allocated(), None, None))
    layer_bytes = None
    for slots in (2, 3):
        streamer = WeightStreamer(blocks, slots=slots)
        layer_bytes = streamer.layer_bytes[0]
        try:
            torch.cuda.reset_peak_memory_stats()
            loads, late = streamer.loads, streamer.late_loads
            ms = _time_steps(blocks, x, weight, args.iters)
            per_step = (streamer.loads - loads) / (args.iters + 3)
            rows.append((f"K={slots}", ms, torch.cuda.max_memory_allocated(), streamer.setup_seconds,
                         (per_step, streamer.late_loads - late)))
        finally:
            streamer.restore()
    copy_ms, gbps = _copy_bandwidth(layer_bytes)
    print(f"one layer: {layer_bytes / 2**20:.2f} MiB, host -> GPU copy {copy_ms:.2f} ms ({gbps:.1f} GB/s)")
    print("| mode | fwd+bwd, ms | per layer, ms | overhead | peak allocated, MiB | setup, s | copies/step | late |")
    print("|---|---|---|---|---|---|---|---|")
    for name, ms, peak, setup, loads in rows:
        print(f"| {name} | {ms:.1f} | {ms / args.layers:.2f} | {100 * (ms / resident - 1):+.1f}% "
              f"| {peak / 2**20:.0f} | {'' if setup is None else f'{setup:.3f}'} "
              f"| {'' if loads is None else f'{loads[0]:.1f}'} | {'' if loads is None else loads[1]} |")


if __name__ == "__main__":
    main()
