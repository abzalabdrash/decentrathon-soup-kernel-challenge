import time
import torch
import torch.nn.functional as F

assert torch.cuda.is_available(), "no CUDA"
dev = torch.device("cuda")
print("torch", torch.__version__, "cuda", torch.version.cuda, "|",
      torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)

N, D, P = 512, 1536, 0.05
LAYERS = 12  # 6 kernels per layer -> 72 launches per block, close to one transformer layer's ~70
torch.manual_seed(0)
ws = [torch.randn(D, D, device=dev, dtype=torch.float16) * 0.02 for _ in range(2 * LAYERS)]
x = torch.randn(N, D, device=dev, dtype=torch.float16)


def block(x, layers=LAYERS):
    h = x
    for i in range(layers):
        a = F.layer_norm(h, (D,))
        a = F.linear(a, ws[2 * i])
        a = F.dropout(a, P, True)
        a = F.silu(a)
        a = F.linear(a, ws[2 * i + 1])
        h = h + a
    return h


def warmup(fn, n=3):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()


def cpu_us(fn, reps=200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) / reps * 1e6, (t2 - t0) / reps * 1e6


def gpu_us(fn, reps=100):
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps * 1e3


cap_stream = torch.cuda.Stream()
# ---- 1. capture cost: torch.cuda.graph (with __enter__ extras) vs raw capture_begin/end
with torch.cuda.stream(cap_stream):
    for _ in range(3):
        block(x)
torch.cuda.synchronize()
for name in ["enter", "raw"]:
    times = []
    for _ in range(5):
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        if name == "enter":
            t0 = time.perf_counter()
            with torch.cuda.graph(g):
                out = block(x)
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            with torch.cuda.stream(cap_stream):
                g.capture_begin()
                out = block(x)
                g.capture_end()
            t1 = time.perf_counter()
        times.append((t1 - t0) * 1e3)
        del out
    print(f"capture {name:5s} ms per capture (72 launches): " + " ".join(f"{t:.1f}" for t in times), flush=True)

# cost of the extra work torch.cuda.graph.__enter__ does, measured alone
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    torch.cuda.empty_cache()
    t2 = time.perf_counter()
    torch._C._host_emptyCache()
    t3 = time.perf_counter()
    ts.append(((t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3))
print("enter extras ms (sync, empty_cache, host_emptyCache): " +
      " | ".join(f"{a:.2f},{b:.2f},{c:.2f}" for a, b, c in ts), flush=True)

# ---- 2. replay vs eager: CPU issue time and GPU time per block
warmup(lambda: block(x))
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out_static = block(x)
g.replay()
torch.cuda.synchronize()
eager_issue, eager_wall = cpu_us(lambda: block(x))
replay_issue, replay_wall = cpu_us(g.replay)
eager_gpu = gpu_us(lambda: block(x))
replay_gpu = gpu_us(g.replay)
print(f"72-launch block: eager issue {eager_issue:.0f} us, eager wall {eager_wall:.0f} us, "
      f"eager GPU {eager_gpu:.0f} us", flush=True)
print(f"72-launch block: replay issue {replay_issue:.0f} us, replay wall {replay_wall:.0f} us, "
      f"replay GPU {replay_gpu:.0f} us", flush=True)

# ---- 3. dropout masks: eager sequence vs replay sequence, and interleaved
K = 6
def run_eager():
    torch.cuda.manual_seed(1234)
    return [block(x).clone() for _ in range(K)]

def run_replay():
    torch.cuda.manual_seed(1234)
    res = []
    for _ in range(K):
        g.replay()
        res.append(out_static.clone())
    return res

def run_mixed():
    torch.cuda.manual_seed(1234)
    res = []
    for k in range(K):
        if k % 2:
            g.replay()
            res.append(out_static.clone())
        else:
            res.append(block(x).clone())
    return res

ref = run_eager()
ref2 = run_eager()
rep = run_replay()
mix = run_mixed()
print("eager run twice identical:", all(torch.equal(a, b) for a, b in zip(ref, ref2)))
print("consecutive eager calls differ (dropout active):", not torch.equal(ref[0], ref[1]))
print("replay == eager, bitwise, per call:", [torch.equal(a, b) for a, b in zip(ref, rep)])
print("mixed (eager+replay interleaved) == eager, bitwise:", [torch.equal(a, b) for a, b in zip(ref, mix)])
print("max abs diff replay vs eager:", max((a.float() - b.float()).abs().max().item() for a, b in zip(ref, rep)))

# ---- 4. capture cost vs graph size
for L in [1, 2, 4, 12]:
    warmup(lambda: block(x, L))
    tt = []
    for _ in range(3):
        gg = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(gg):
            o = block(x, L)
        tt.append((time.perf_counter() - t0) * 1e3)
        del o
    print(f"capture {6*L:3d} launches: {min(tt):.1f} ms (min of 3)", flush=True)
