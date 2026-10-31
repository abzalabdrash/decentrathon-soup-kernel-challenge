"""The output head of one training step on the task's shapes (V = 151,936, D = 1536): the resident head
(fused_ce.py, weight on the GPU, 512-row logits chunks) against the streamed one (head_stream.py,
weight in pinned host memory) for several chunk sizes, forward + backward, with the first chunks
prefetched as a training forward does and without. Prints time, peak allocated memory above the
inputs, PCIe wait and the own kernels' time. Run on the target GPU:
`python -m solution.bench_head_stream [--tokens 354 826 1427] [--rows 4096 8192 16384]`.
"""

import argparse

import torch

from solution.fused_ce import fused_lm_head_loss
from solution.head_stream import HeadStreamer

V, D = 151936, 1536


def _case(tokens, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(tokens, D, device="cuda", generator=gen) * 2.0
    targets = torch.randint(0, V, (tokens,), device="cuda", generator=gen)
    targets[: tokens // 4] = -100  # a prompt part
    return hidden, targets


def _time(fn, iters):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, (torch.cuda.max_memory_allocated() - base) / 2**20


def _step(weight, hidden, targets, prefetch=None):
    def run():
        if prefetch is not None:
            prefetch()
        h = hidden.detach().requires_grad_(True)
        loss, *_ = fused_lm_head_loss(h, weight, targets, None)
        loss.backward()
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[354, 826, 1427])
    parser.add_argument("--rows", type=int, nargs="+", default=[4096, 8192, 16384])
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(0)
    weight = (torch.randn(V, D, device="cuda") * 0.02).half()
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    streamer = HeadStreamer(param, rows=args.rows[0], slots=args.slots)
    streamer.timing = True  # the copy bandwidth and the PCIe wait below (SOUP_SOLUTION_STREAM_TIMING)
    print(f"{torch.cuda.get_device_name()}; weight to pinned host memory in {streamer.setup_seconds:.2f} s; "
          f"fp32-output GEMM: {streamer.fp32_out}")
    print("| tokens | head | ms | peak above inputs, MiB | PCIe wait, ms | chunk copy GB/s |")
    print("|---|---|---|---|---|---|")
    for tokens in args.tokens:
        hidden, targets = _case(tokens)
        ms, peak = _time(_step(weight, hidden, targets), args.iters)
        print(f"| {tokens} | resident (+445 MiB static) | {ms:.1f} | {peak:.0f} | | |")
        for rows in args.rows:
            for prefetch in (False, True):
                streamer.configure(rows=rows, slots=args.slots)
                st0 = streamer.stats()
                ms, peak = _time(_step(param, hidden, targets, streamer.prefetch if prefetch else None),
                                 args.iters)
                st = streamer.stats()
                heads = st["heads"] - st0["heads"]
                wait = (st["wait_ms"] - st0["wait_ms"]) / heads
                gbps = ((st["copy_gbps"] or 0.0))
                print(f"| {tokens} | streamed, {rows} rows x {args.slots}{', prefetched' if prefetch else ''} "
                      f"| {ms:.1f} | {peak:.0f} | {wait:.1f} | {gbps:.1f} |")
    streamer.restore()


if __name__ == "__main__":
    main()
