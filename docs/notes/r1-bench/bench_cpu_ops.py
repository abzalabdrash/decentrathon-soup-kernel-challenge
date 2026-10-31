"""Per-operation host (CPU) cost of eager PyTorch 2.11 on this machine (research R1).

Each number is the wall time of a Python loop that only enqueues work, divided by the
loop count, median over reps. GPU is synchronised after each rep, so the queue never
holds more than n launches (n is kept below the ~1024-entry launch queue).
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn
import triton
import triton.language as tl

import bitsandbytes as bnb

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
    print(f"{name:52s} {v:9.2f} us", flush=True)


# ---- 1. eager dispatch, CPU tensors (no queue, pure dispatcher + Python) ----
c = torch.zeros(16)
record("cpu add_ scalar (16 el), python method", lambda: c.add_(1.0), 20000)
record("cpu torch.ops.aten.add_.Tensor direct", lambda: torch.ops.aten.add_.Tensor(c, 1.0), 20000)
record("cpu attr read .shape (plain tensor)", lambda: c.shape, 20000)

# ---- 2. eager dispatch, CUDA tiny elementwise (one kernel each) ----
g = torch.zeros(16, device=dev)
record("cuda add_ scalar (16 el)", lambda: g.add_(1.0), 2000)
record("cuda torch.ops.aten.add_.Tensor direct", lambda: torch.ops.aten.add_.Tensor(g, 1.0), 2000)

# ---- 3. Triton kernel launch (own kernel, trivial body) ----
@triton.jit
def _noop(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(x_ptr + offs, x + 1.0, mask=m)


tx = torch.zeros(1024, device=dev)
record("triton launch, 1 CTA, BLOCK 1024", lambda: _noop[(1,)](tx, 1024, BLOCK=1024), 2000)

# ---- 4. cuBLAS GEMM enqueue (fp16, bias epilogue like our qkv) ----
W = torch.randn(1536, 1536, device=dev, dtype=torch.float16)
b = torch.zeros(1536, device=dev, dtype=torch.float16)
x1 = torch.randn(1, 1536, device=dev, dtype=torch.float16)
x832 = torch.randn(832, 1536, device=dev, dtype=torch.float16)
record("cublas addmm M=1 K=N=1536 fp16 (enqueue)", lambda: torch.addmm(b, x1, W.t()), 1000)
record("cublas addmm M=832 K=N=1536 fp16 (enqueue)", lambda: torch.addmm(b, x832, W.t()), 300)

# ---- 5. CUDA events and streams ----
ev = torch.cuda.Event()
s0 = torch.cuda.current_stream()
record("cuda Event.record()", lambda: ev.record(), 5000)
record("current_stream().wait_event(ev) (done event)", lambda: s0.wait_event(ev), 5000)
s1 = torch.cuda.Stream()
def _ctx():
    with torch.cuda.stream(s1):
        pass
record("with torch.cuda.stream(s): pass", _ctx, 5000)

# ---- 6. autograd.Function call vs native autograd op ----
class Ident(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


xg = torch.zeros(16, device=dev, requires_grad=True)
record("autograd.Function.apply (identity, fwd only)", lambda: Ident.apply(xg), 2000)
record("native view_as under grad mode", lambda: xg.view_as(xg), 2000)
with torch.no_grad():
    record("autograd.Function.apply under no_grad", lambda: Ident.apply(xg), 2000)

# ---- 7. __torch_function__ dispatch on attribute reads ----
p_plain = nn.Parameter(torch.zeros(1536, 1536, device=dev, dtype=torch.float16), requires_grad=False)
lin4 = bnb.nn.Linear4bit(1536, 1536, bias=False, compute_dtype=torch.float16, quant_type="nf4").to(dev)
p4 = lin4.weight
record("attr .shape on plain Tensor", lambda: c.shape, 20000)
record("attr .shape on nn.Parameter (__torch_function__ disabled)", lambda: p_plain.shape, 20000)
record("attr .shape on bnb Params4bit (python __torch_function__)", lambda: p4.shape, 20000)
record("attr .requires_grad on Params4bit", lambda: p4.requires_grad, 20000)
record("attr .dtype on Params4bit", lambda: p4.dtype, 20000)
record("attr .quant_state on Params4bit (python __dict__)", lambda: p4.quant_state, 20000)

# ---- 8. bnb Linear4bit forward CPU cost (M=832 and M=1), grad and no-grad paths ----
x832g = torch.randn(832, 1536, device=dev, dtype=torch.float16, requires_grad=True)
x1n = torch.randn(1, 1536, device=dev, dtype=torch.float16)
record("bnb Linear4bit fwd M=832, grad on (MatMul4Bit.apply)", lambda: lin4(x832g), 200)
with torch.no_grad():
    record("bnb Linear4bit fwd M=832, no_grad (gemm_4bit)", lambda: lin4(x832g), 200)
    record("bnb Linear4bit fwd M=1, no_grad (gemm_4bit)", lambda: lin4(x1n), 1000)

# ---- 9. host waits: .item() on an idle GPU (round trip) ----
s_dev = torch.zeros(1, device=dev)
def item_idle():
    s_dev.item()
record(".item() on idle GPU (sync round trip)", item_idle, 2000)

# ---- 10. multi-tensor ops vs per-tensor loop, 450 tensors (our parameter count) ----
ts = [torch.zeros(16, device=dev) for _ in range(450)]
record("loop add_ over 450 tensors (per 450-tensor pass)", lambda: [t.add_(1.0) for t in ts], 20)
record("torch._foreach_add_ over 450 tensors (per pass)", lambda: torch._foreach_add_(ts, 1.0), 20)

print(json.dumps(out, indent=1))
sys.stdout.flush()
