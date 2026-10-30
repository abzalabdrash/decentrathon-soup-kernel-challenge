"""Device memory outside torch's allocator on the T4 (measurement only): what nvidia-smi counts beyond
`memory_reserved` at stages of a run like ours (bare context, cuBLAS GEMM, cuBLASLt addmm, the
memory-efficient attention forward and backward, a Triton kernel, bitsandbytes' 4-bit quantization),
and the CUDA limits that reserve memory per thread (stack) or per context (printf FIFO, malloc heap).

`python -m solution.bench_ctx` (`tools/kaggle_run.py --mode bench --bench solution.bench_ctx`)."""

import ctypes
import glob
import os

import torch

LIMITS = {"stack": 0x00, "printf FIFO": 0x01, "malloc heap": 0x02}


def _cudart():
    import nvidia.cuda_runtime as rt  # the runtime torch's wheels ship

    paths = glob.glob(os.path.join(list(rt.__path__)[0], "lib", "libcudart.so*"))
    return ctypes.CDLL(sorted(paths)[-1]) if paths else ctypes.CDLL("libcudart.so")


def report(stage):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    used = (total - free) / 2 ** 20
    reserved = torch.cuda.memory_reserved() / 2 ** 20
    print(f"{stage:48s} used {used:7.1f} MiB  reserved {reserved:6.1f}  outside {used - reserved:6.1f}",
          flush=True)
    return used - reserved


def limits(rt):
    out = {}
    for name, enum in LIMITS.items():
        value = ctypes.c_size_t()
        rt.cudaDeviceGetLimit(ctypes.byref(value), enum)
        out[name] = value.value
    return out


def main():
    torch.empty(1, device="cuda")
    rt = _cudart()
    report("context (one tensor)")
    print("limits:", limits(rt), flush=True)
    a = torch.randn(1440, 1536, device="cuda").half()
    w = torch.randn(8960, 1536, device="cuda").half()
    (a @ w.t()).sum().item()
    report("cuBLAS fp16 GEMM")
    b = torch.randn(1536, device="cuda").half()
    torch.addmm(b, a, w[:1536].t()).sum().item()
    report("addmm with bias (cuBLASLt)")
    q = torch.randn(1, 12, 1440, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 12, 1440, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 12, 1440, 128, device="cuda", dtype=torch.float16, requires_grad=True)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    report("mem-efficient attention forward")
    out.sum().backward()
    report("mem-efficient attention backward")
    print("limits:", limits(rt), flush=True)
    from solution.test_fused_proj import _linear4bit

    layer = _linear4bit(1536, 8960, seed=1)
    report("bitsandbytes NF4 quantization")
    from solution import nf4

    nf4.dequantize_nf4(layer.weight, layer.weight.quant_state, torch.float16).sum().item()
    report("Triton NF4 dequantization")
    print("limits:", limits(rt), flush=True)
    before = report("before lowering the stack limit")
    for size in (512, 256):
        err = rt.cudaDeviceSetLimit(LIMITS["stack"], ctypes.c_size_t(size))
        after = report(f"stack limit set to {size} B (error {err})")
        print(f"    outside the allocator {before - after:+.1f} MiB less", flush=True)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out.sum().backward()
    nf4.dequantize_nf4(layer.weight, layer.weight.quant_state, torch.float16).sum().item()
    (a @ w.t()).sum().item()
    report("after the same kernels again")
    print("limits:", limits(rt), flush=True)


if __name__ == "__main__":
    main()
