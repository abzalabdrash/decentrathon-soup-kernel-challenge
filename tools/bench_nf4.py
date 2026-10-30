"""Microbenchmark of solution/nf4.py: the Triton NF4 dequantization against bitsandbytes'
`dequantize_4bit`, and a whole 4-bit linear layer forward + backward, patched against the original.

    python tools/bench_nf4.py            # from the repository root, on a CUDA GPU

Dequantization: GPU time of one call, the sum of its CUDA kernels (torch.profiler, mean of 20 calls,
the lowest of 5 rounds):
bitsandbytes' three kernels (absmax dequantization, offset add, 4-bit dequantization), the previous
version of our kernel (commit 9109308, kept below for the comparison) and the current one. The
bandwidth of the current kernel (bytes read + written / time) is given against the better of a device
copy and a fill of 64 MB on the same GPU (the pass is 80% writes), the practical ceiling of a
memory-bound pass.

Layer: wall time (CUDA events over 50 back-to-back steps, the lowest of 5 repeats) of a Linear4bit
forward + backward, patched against the original.
"""

import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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
    return min(times)


def gpu_ms(fn, calls=20, rounds=REPEATS):
    """Sum of the CUDA kernel times of one call of `fn`: the mean over `calls`, the lowest of
    `rounds` (a shared GPU only ever adds time)."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(WARMUP):
        fn()
    best = float("inf")
    for _ in range(rounds):
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                fn()
            torch.cuda.synchronize()
        best = min(best, sum(e.device_time_total for e in prof.key_averages()) / calls / 1000)
    return best


@triton.jit
def previous_nf4_dequant_kernel(packed_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, out_ptr,
                                n, n_blocks, QBLOCK: tl.constexpr, ABLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    """solution/nf4.py at commit 9109308: byte loads, per-element masks, the two nibbles interleaved."""
    blk = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    live = blk < n_blocks
    q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
    scale = tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
    absmax = tl.load(code2_ptr + q) * scale
    absmax = absmax + tl.load(offset_ptr)
    half: tl.constexpr = QBLOCK // 2
    byte = blk[:, None] * half + tl.arange(0, half)[None, :]
    packed = tl.load(packed_ptr + byte, mask=byte * 2 < n, other=0)
    hi = tl.load(nf4_ptr + (packed >> 4).to(tl.int32)) * absmax[:, None]
    lo = tl.load(nf4_ptr + (packed & 15).to(tl.int32)) * absmax[:, None]
    vals = tl.interleave(hi, lo)
    elem = blk[:, None] * QBLOCK + tl.arange(0, QBLOCK)[None, :]
    tl.store(out_ptr + elem, vals.to(out_ptr.dtype.element_ty), mask=elem < n)


def previous_dequantize(weight, qs):
    n_rows, n_cols = qs.shape
    out = torch.empty((n_rows, n_cols), dtype=torch.float16, device=weight.device)
    n_blocks = qs.absmax.numel()
    previous_nf4_dequant_kernel[(triton.cdiv(n_blocks, 32),)](
        weight, qs.absmax, qs.state2.code, qs.state2.absmax, qs.offset, qs.code, out, n_rows * n_cols, n_blocks,
        QBLOCK=64, ABLOCK=256, BLOCKS=32, num_warps=4, enable_fp_fusion=False)
    return out


def device_bandwidth():
    """GB/s of a device-to-device copy and of a fill, 64 MB each (kernel time)."""
    src = torch.empty(32 * 2 ** 20, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    t_copy = gpu_ms(lambda: dst.copy_(src), 50)
    t_fill = gpu_ms(lambda: dst.fill_(1.0), 50)
    return 2 * src.numel() * 2 / t_copy / 1e6, src.numel() * 2 / t_fill / 1e6


def main():
    import bitsandbytes.functional as bnb_functional
    from bitsandbytes.autograd import _functions as bnb_autograd

    from solution.nf4 import dequantize_nf4, patch_bnb_matmul4bit

    print(torch.cuda.get_device_name(), f"torch {torch.__version__}")
    copy_bw, fill_bw = device_bandwidth()
    print(f"device copy {copy_bw:.0f} GB/s, fill {fill_bw:.0f} GB/s")
    print("\ndequantization to fp16, GPU time of one call, ms")
    print(f"{'out x in':>12} {'bnb':>7} {'previous':>8} {'current':>8} {'vs bnb':>7} {'vs prev':>7} "
          f"{'GB/s':>5} {'of ceiling':>10}")
    for k, n in SHAPES:
        layer = make_layer(k, n)
        qs = layer.weight.quant_state
        weight = layer.weight.data
        ref = bnb_functional.dequantize_4bit(layer.weight, qs)
        assert torch.equal(dequantize_nf4(weight, qs), ref) and torch.equal(previous_dequantize(weight, qs), ref)
        g_bnb = gpu_ms(lambda: bnb_functional.dequantize_4bit(layer.weight, qs))
        g_prev = gpu_ms(lambda: previous_dequantize(weight, qs))
        g_ours = gpu_ms(lambda: dequantize_nf4(weight, qs))
        gbs = (n * k * 2 + n * k // 2 + qs.absmax.numel()) / g_ours / 1e6
        print(f"{f'{n}x{k}':>12} {g_bnb:7.4f} {g_prev:8.4f} {g_ours:8.4f} {g_bnb / g_ours:6.2f}x "
              f"{g_prev / g_ours:6.2f}x {gbs:5.0f} {100 * gbs / max(copy_bw, fill_bw):9.0f}%")

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
