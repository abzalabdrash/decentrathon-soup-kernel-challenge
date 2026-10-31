"""Second host-cost pass (research R1): pre-bound aten ops, Triton launch paths, LoRA-shaped eager ops."""
import json
import statistics
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl

dev = "cuda"
out = {}


def per_call_us(fn, n, reps=7):
    vals = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        vals.append((t1 - t0) / n * 1e6)
    return round(statistics.median(vals), 2)


def record(name, fn, n):
    fn()
    fn()
    v = per_call_us(fn, n)
    out[name] = v
    print(f"{name:60s} {v:9.2f} us", flush=True)


@triton.jit
def _noop(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(x_ptr + offs, x + 1.0, mask=m)


g = torch.zeros(16, device=dev)
tx = torch.zeros(1024, device=dev)

# pre-bound aten overload vs attribute chain per call
add_ = torch.ops.aten.add_.Tensor
record("cuda aten.add_.Tensor pre-bound", lambda: add_(g, 1.0), 2000)
record("cuda torch.ops.aten.add_.Tensor (chain each call)", lambda: torch.ops.aten.add_.Tensor(g, 1.0), 2000)
record("cuda g.add_(1.0) method", lambda: g.add_(1.0), 2000)
record("cuda torch.empty(16, device=cuda) (alloc + launch-free)", lambda: torch.empty(16, device=dev), 5000)

# Triton: JIT dispatch vs cached compiled kernel
launch_jit = lambda: _noop[(1,)](tx, 1024, BLOCK=1024)
record("triton JIT launch kernel[(1,)](...)", launch_jit, 2000)
try:
    ck = _noop.warmup(tx, 1024, BLOCK=1024, grid=(1,))
    print("warmup returned", type(ck).__name__, flush=True)
    ck[(1, 1, 1)](tx, 1024)
    torch.cuda.synchronize()
    launch_ck = lambda: ck[(1, 1, 1)](tx, 1024)
    record("triton CompiledKernel[(1,1,1)](...) reused handle", launch_ck, 2000)
except Exception as e:  # report API mismatch instead of hiding it
    out["triton CompiledKernel path"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
    print(out["triton CompiledKernel path"], flush=True)

# LoRA-shaped eager ops for one q_proj LoRA module (r=16) on M=832 fp16 tokens, no bnb
x = torch.randn(832, 1536, device=dev, dtype=torch.float16)
res = torch.randn(832, 1536, device=dev, dtype=torch.float16)
lora_A = nn.Linear(1536, 16, bias=False, device=dev, dtype=torch.float16)
lora_B = nn.Linear(16, 1536, bias=False, device=dev, dtype=torch.float16)
drop = nn.Dropout(0.05)
drop.train()


def lora_module_eager():
    r = res.clone()
    o = lora_B(lora_A(drop(x))) * 2.0
    return r + o


def lora_module_no_dropout():
    r = res.clone()
    o = lora_B(lora_A(x)) * 2.0
    return r + o


record("LoRA module eager (clone, dropout, A, B, scale, add) M=832", lora_module_eager, 300)
record("LoRA module eager without dropout M=832", lora_module_no_dropout, 300)

print(json.dumps(out, indent=1))
