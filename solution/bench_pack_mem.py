"""What the head pack adds to nvidia-smi's memory.used outside torch's allocator (variants showed
403 MiB against 401 with identical reserved/allocated): GPU memory after each step of
    context -> register 334 MiB of host memory (the pack's size) -> register 1,090 MiB (the layers
    and the head) -> 10 pinned slabs of 256 MiB (the offload) -> load head_unpack_kernel's module ->
    load another small Triton module -> unregister the 334 MiB

    python -m solution.bench_pack_mem
"""

import subprocess
import time

import torch
import triton
import triton.language as tl


def smi():
    torch.cuda.synchronize()
    time.sleep(0.3)
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "--id=0"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


@triton.jit
def _other_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=offs < n) * 2, mask=offs < n)


def registered(mib):
    buffer = torch.empty(mib << 20, dtype=torch.uint8)
    buffer.fill_(0)
    assert torch.cuda.cudart().cudaHostRegister(buffer.data_ptr(), buffer.numel(), 0) == 0
    probe = torch.empty(1 << 20, dtype=torch.uint8, device="cuda")
    probe.copy_(buffer[:1 << 20], non_blocking=True)  # the GPU reads it once
    probe.copy_(buffer[-(1 << 20):], non_blocking=True)
    return buffer


def main():
    from .head_pack import decode_chunk

    print(torch.cuda.get_device_name(0), flush=True)
    torch.empty(1, device="cuda")
    steps = [("context", smi())]
    keep = [registered(334)]
    steps.append(("+ register 334 MiB", smi()))
    keep.append(registered(1090))
    steps.append(("+ register 1090 MiB", smi()))
    keep += [torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True) for _ in range(10)]
    steps.append(("+ 10 pinned slabs of 256 MiB", smi()))
    src = torch.zeros(3 * 4096, dtype=torch.uint8, device="cuda")
    out = torch.empty(4096, dtype=torch.float16, device="cuda")
    table = torch.zeros(16, dtype=torch.int32, device="cuda")
    esc = torch.zeros(32, dtype=torch.int32, device="cuda")
    steps.append(("+ small tensors", smi()))
    decode_chunk(src, out, table, esc, esc, 0)
    steps.append(("+ head_unpack_kernel module", smi()))
    x = torch.ones(4096, device="cuda")
    _other_kernel[(1,)](x, x.numel(), BLOCK=4096)
    steps.append(("+ another Triton module", smi()))
    torch.cuda.cudart().cudaHostUnregister(keep[0].data_ptr())
    steps.append(("- unregister 334 MiB", smi()))
    prev = None
    for name, used in steps:
        print(f"{name:32s} {used:6d} MiB" + (f"  ({used - prev:+d})" if prev is not None else ""), flush=True)
        prev = used


if __name__ == "__main__":
    main()
