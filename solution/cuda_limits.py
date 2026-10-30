"""The CUDA context's per-thread stack reservation, set lower at the start of training.

A context reserves local memory for the stack of every thread that can be resident at once:
`cudaLimitStackSize` (1 KiB by default) x 1,024 threads per SM x 40 SMs = 40 MiB on the T4, counted
by nvidia-smi outside torch's allocator. The kernels of this run do not use that much: cuBLAS and
cuBLASLt GEMMs, the memory-efficient attention forward and backward, torch's elementwise kernels and
our Triton kernels have no recursion and spill little. Measured on the T4 (solution/bench_ctx.py,
soup-night-ctx): device memory outside the allocator 132.9 MiB after a cuBLAS GEMM, 112.9 MiB with
the limit at 512 B, 102.9 MiB at 256 B, still 102.9 MiB after the attention forward/backward, the
NF4 dequantization and the GEMM ran again (none of them grew it back).

SOUP_SOLUTION_STACK_LIMIT: bytes per thread, set once when training starts (offload.py's train
wrapper, after the warm-up); 0 leaves the context alone. The call goes to the CUDA runtime library
torch's wheels ship (the same primary context); any failure leaves the limit as it was. A kernel that
needs more stack than the limit still runs: the driver grows the reservation for it.
"""

import ctypes
import glob
import os

LIMIT_STACK_SIZE = 0x00  # cudaLimitStackSize
_STATE = {"bytes": None, "lib": None}


def _env_bytes():
    try:
        return max(0, int(os.environ.get("SOUP_SOLUTION_STACK_LIMIT", "0") or "0"))
    except ValueError:
        return 0


def _cudart():
    if _STATE["lib"] is None:
        paths = []
        try:
            import nvidia.cuda_runtime as rt  # the runtime torch's wheels ship

            for root in rt.__path__:
                paths += glob.glob(os.path.join(root, "lib", "libcudart.so*"))
        except Exception:  # noqa: BLE001 - another install layout: the loader's search path
            pass
        _STATE["lib"] = ctypes.CDLL(sorted(paths)[-1] if paths else "libcudart.so")
    return _STATE["lib"]


def stack_limit():
    """The current per-thread stack limit in bytes (None when the runtime is not reachable)."""
    try:
        value = ctypes.c_size_t()
        if _cudart().cudaDeviceGetLimit(ctypes.byref(value), LIMIT_STACK_SIZE) != 0:
            return None
        return value.value
    except Exception:  # noqa: BLE001
        return None


def lower_stack_limit(nbytes=None):
    """Sets the per-thread stack limit to `nbytes` (SOUP_SOLUTION_STACK_LIMIT by default) when that
    is lower than the current one; returns (before, after) or None when nothing was changed."""
    nbytes = _env_bytes() if nbytes is None else nbytes
    if not nbytes:
        return None
    before = stack_limit()
    if before is None or before <= nbytes:
        return None
    try:
        import torch

        torch.cuda.synchronize()
        if _cudart().cudaDeviceSetLimit(LIMIT_STACK_SIZE, ctypes.c_size_t(nbytes)) != 0:
            return None
    except Exception:  # noqa: BLE001 - the limit stays as it was
        return None
    _STATE["bytes"] = nbytes
    return before, stack_limit()
