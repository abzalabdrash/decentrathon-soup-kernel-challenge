import time, statistics, torch, bitsandbytes as bnb
lin = bnb.nn.Linear4bit(1536, 1536, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
p = lin.weight
d = p.data
print("type of p.data:", type(d).__name__, "| has quant_state:", hasattr(d, "quant_state"), "| is p.data is p.data:", p.data is p.data)
def t(fn, n=20000):
    vals=[]
    for _ in range(5):
        t0=time.perf_counter()
        for _ in range(n): fn()
        vals.append((time.perf_counter()-t0)/n*1e6)
    return round(statistics.median(vals),2)
print("p.data read us:", t(lambda: p.data))
print("p.grad read us:", t(lambda: p.grad))
print("p.is_cuda read us:", t(lambda: p.is_cuda))
print("p.ndim read us:", t(lambda: p.ndim))
