"""Calibration (research R1): CPU enqueue time of a frozen-NF4 bnb stack with Qwen2.5-1.5B shapes,
28 layers, M=832 tokens, forward + backward to the input (no LoRA, no own kernels).
Per-op numbers in bench_cpu_ops*.py say what this should cost; this measures it."""
import statistics, time
import torch, torch.nn as nn, torch.nn.functional as F
import bitsandbytes as bnb
import sys
dev, dt, L = "cuda", torch.float16, 28
M = int(sys.argv[1])
H, I, KV = 1536, 8960, 256

def q4(i, o):
    return bnb.nn.Linear4bit(i, o, bias=False, compute_dtype=dt, quant_type="nf4").to(dev)

class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = q4(H, H), q4(H, KV), q4(H, KV), q4(H, H)
        self.gate, self.up, self.down = q4(H, I), q4(H, I), q4(I, H)
    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        a = self.o(q) + x + (k.sum() + v.sum()) * 0
        m = self.down(F.silu(self.gate(a)) * self.up(a))
        return m + a

layers = nn.ModuleList(Layer() for _ in range(L)).to(dev)
for p in layers.parameters():
    p.requires_grad_(False)
x0 = torch.randn(M, H, device=dev, dtype=dt, requires_grad=True)

def step():
    t0 = time.perf_counter()
    h = x0
    for lyr in layers:
        h = lyr(h)
    loss = h.float().sum()
    t1 = time.perf_counter()
    loss.backward()
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1

def gpu_ms():
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); a.record()
    step()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b)

for _ in range(3):
    step(); torch.cuda.synchronize()
rows = []
for _ in range(5):
    torch.cuda.synchronize()
    tf, tb = step()
    torch.cuda.synchronize()
    rows.append((tf, tb))
f = statistics.median(r[0] for r in rows) * 1e3
b = statistics.median(r[1] for r in rows) * 1e3
print(f"28 layers, M={M}: forward enqueue {f:.1f} ms, backward enqueue {b:.1f} ms, total CPU {f+b:.1f} ms")
print(f"per layer fwd+bwd CPU: {(f+b)/L*1e3:.0f} us")
print(f"GPU wall of one step (events, sync'd): {statistics.median(gpu_ms() for _ in range(3)):.1f} ms")
t0 = time.perf_counter()
torch.cuda.synchronize()
print("sync tail ms", round((time.perf_counter()-t0)*1e3, 2))
