import statistics as st
import time
import torch
import torch.nn.functional as F

dev = torch.device("cuda")
print("torch", torch.__version__, "|", torch.cuda.get_device_name(0), flush=True)
P, LAYERS = 0.05, 12  # 6 launches per layer, 72 per block


def make(N, D):
    torch.manual_seed(0)
    ws = [torch.randn(D, D, device=dev, dtype=torch.float16) * 0.02 for _ in range(2 * LAYERS)]
    x = torch.randn(N, D, device=dev, dtype=torch.float16)

    def block(x):
        h = x
        for i in range(LAYERS):
            a = F.layer_norm(h, (D,))
            a = F.linear(a, ws[2 * i])
            a = F.dropout(a, P, True)
            a = F.silu(a)
            a = F.linear(a, ws[2 * i + 1])
            h = h + a
        return h
    return x, block


def med_us(fn, reps=200, trials=5):
    """median over trials of (issue, wall) per call, microseconds"""
    iss, wal = [], []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        iss.append((t1 - t0) / reps * 1e6)
        wal.append((t2 - t0) / reps * 1e6)
    return st.median(iss), st.median(wal)


def gpu_us(fn, reps=100):
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps * 1e3


for N, D in [(64, 256), (512, 1536)]:
    x, block = make(N, D)
    cap = torch.cuda.Stream()
    with torch.cuda.stream(cap):
        for _ in range(3):
            block(x)
    torch.cuda.synchronize()

    caps = []
    for _ in range(5):
        g = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.stream(cap):
            g.capture_begin()
            out = block(x)
            g.capture_end()
        caps.append((time.perf_counter() - t0) * 1e3)
        del out
    g.replay()
    torch.cuda.synchronize()

    ei, ew = med_us(lambda: block(x))
    ri, rw = med_us(g.replay)
    eg = gpu_us(lambda: block(x))
    rg = gpu_us(g.replay)
    print(f"N={N} D={D}: capture(72 launches) ms median {st.median(caps):.2f} (all {', '.join(f'{c:.1f}' for c in caps)})", flush=True)
    print(f"  eager : issue {ei:7.0f} us/block ({ei/72:5.1f} us/launch), wall {ew:7.0f} us, GPU {eg:7.0f} us", flush=True)
    print(f"  replay: issue {ri:7.0f} us/block ({ri/72:5.1f} us/launch), wall {rw:7.0f} us, GPU {rg:7.0f} us", flush=True)
    del g
    torch.cuda.empty_cache()
