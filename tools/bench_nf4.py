"""Microbenchmark of solution/nf4.py: the Triton NF4 dequantization against bitsandbytes'
`dequantize_4bit`, and a whole 4-bit linear layer forward + backward, patched against the original.

    python tools/bench_nf4.py            # from the repository root, on a CUDA GPU

Wall times are CUDA events over 50 back-to-back calls after a warm-up, the median of 5 repeats, in ms
(small shapes are bound by the CPU cost of a call). GPU times are the sum of the CUDA kernels of one
call (torch.profiler, 20 calls): bitsandbytes' three kernels (absmax dequantization, offset add, 4-bit
dequantization) against our one.
"""

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SHAPES = [(1536, 1536), (1536, 256), (1536, 8960), (8960, 1536)]  # (in_features, out_features)
TOKENS = 2048
ITERS, REPEATS, WARMUP = 50, 5, 10


def make_layer(k, n, seed=0):
    import bitsandbytes as bnb

    torch.manual_seed(seed)
    layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16, quant_type="nf4",
                              compress_statistics=True)
    weight = (torch.randn(n, k) * 0.02).half()
    layer.weight = bnb.nn.Params4bit(weight, requires_grad=False, quant_type="nf4",
                                     compress_statistics=True, module=layer)
    return layer.cuda()


def timed(fn):
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(REPEATS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / ITERS)
    return statistics.median(times)


def gpu_ms(fn, calls=20):
    """Sum of the CUDA kernel times of one call of `fn`."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    return sum(e.device_time_total for e in prof.key_averages()) / calls / 1000


def main():
    import bitsandbytes.functional as bnb_functional
    from bitsandbytes.autograd import _functions as bnb_autograd

    from solution.nf4 import dequantize_nf4, patch_bnb_matmul4bit

    print(torch.cuda.get_device_name(), f"torch {torch.__version__}")
    print("\ndequantization to fp16, ms (GB/s = bytes read + written / GPU time of the kernel)")
    print(f"{'out x in':>12} {'wall bnb':>9} {'wall ours':>9} {'GPU bnb':>8} {'GPU ours':>8} "
          f"{'GPU speedup':>11} {'GB/s':>5}")
    for k, n in SHAPES:
        layer = make_layer(k, n)
        qs = layer.weight.quant_state
        assert torch.equal(dequantize_nf4(layer.weight, qs), bnb_functional.dequantize_4bit(layer.weight, qs))
        t_bnb = timed(lambda: bnb_functional.dequantize_4bit(layer.weight, qs))
        t_ours = timed(lambda: dequantize_nf4(layer.weight, qs))
        g_bnb = gpu_ms(lambda: bnb_functional.dequantize_4bit(layer.weight, qs))
        g_ours = gpu_ms(lambda: dequantize_nf4(layer.weight, qs))
        moved = n * k * 2 + n * k // 2 + qs.absmax.numel()
        print(f"{f'{n}x{k}':>12} {t_bnb:9.4f} {t_ours:9.4f} {g_bnb:8.4f} {g_ours:8.4f} "
              f"{g_bnb / g_ours:10.2f}x {moved / g_ours / 1e6:5.0f}")

    patch_bnb_matmul4bit()
    cls = bnb_autograd.MatMul4Bit

    class Original(torch.autograd.Function):
        forward = staticmethod(cls._soup_original_forward)
        backward = staticmethod(cls._soup_original_backward)

    print(f"\nLinear4bit forward + backward, x ({TOKENS}, in) fp32 under fp16 autocast, ms")
    print(f"{'out x in':>12} {'original':>9} {'patched':>8} {'speedup':>8}")
    for k, n in SHAPES:
        layer = make_layer(k, n)
        x = torch.randn(TOKENS, k, device="cuda", requires_grad=True)
        grad = torch.randn(TOKENS, n, device="cuda")

        def step():
            with torch.autocast("cuda", dtype=torch.float16):
                out = layer(x)
            out.backward(grad)
            x.grad = None

        bnb_autograd.MatMul4Bit = Original
        t_orig = timed(step)
        bnb_autograd.MatMul4Bit = cls
        t_ours = timed(step)
        print(f"{f'{n}x{k}':>12} {t_orig:9.4f} {t_ours:8.4f} {t_orig / t_ours:7.2f}x")


if __name__ == "__main__":
    main()
