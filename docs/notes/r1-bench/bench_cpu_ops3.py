"""Queue-safe pass (research R1): n kept at 80 so the GPU launch queue never fills during timing."""
import statistics, time
import torch, torch.nn as nn
import bitsandbytes as bnb
dev = "cuda"

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

def record(name, fn, n=80):
    fn(); fn()
    print(f"{name:60s} {per_call_us(fn, n):9.2f} us", flush=True)

x = torch.randn(832, 1536, device=dev, dtype=torch.float16, requires_grad=True)
lin4 = bnb.nn.Linear4bit(1536, 1536, bias=False, compute_dtype=torch.float16, quant_type="nf4").to(dev)
record("bnb Linear4bit fwd M=832 grad on (MatMul4Bit.apply)", lambda: lin4(x))
with torch.no_grad():
    record("bnb Linear4bit fwd M=832 no_grad", lambda: lin4(x))
res = torch.randn(832, 1536, device=dev, dtype=torch.float16)
A = nn.Linear(1536, 16, bias=False, device=dev, dtype=torch.float16)
B = nn.Linear(16, 1536, bias=False, device=dev, dtype=torch.float16)
drop = nn.Dropout(0.05).train()
def lora_eager():
    return res.clone() + B(A(drop(res))) * 2.0
def lora_eager_nodrop():
    return res.clone() + B(A(res)) * 2.0
record("LoRA module eager, dropout, clone, add (M=832)", lora_eager)
record("LoRA module eager, no dropout (M=832)", lora_eager_nodrop)
ts = [torch.zeros(16, device=dev) for _ in range(450)]
record("loop add_ over 450 tensors (per 450-tensor pass)", lambda: [t.add_(1.0) for t in ts], 20)
record("torch._foreach_add_ over 450 tensors (per pass)", lambda: torch._foreach_add_(ts, 1.0), 20)
