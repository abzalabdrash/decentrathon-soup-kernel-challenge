"""Micro-check: first-call (cold, lazy module load) vs second-call cost of torch ops the step uses
but warmup.py does not warm; out_dtype availability; tokenizer save time. Local card only."""
import os, sys, time
t0 = time.perf_counter()
import torch
torch.empty(1, device="cuda")
print("CUDA_MODULE_LOADING", os.environ.get("CUDA_MODULE_LOADING"), "ctx", round(time.perf_counter() - t0, 2))
half = torch.float16

def timed(name, fn, reps=2):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter() - s) * 1e3)
    print(f"{name:40s} first {out[0]:8.2f} ms  second {out[1]:7.2f} ms")
    return out[0] - out[1]

N, H, KV, D = 830, 12, 2, 128
q = torch.ones(1, N, H, D, device="cuda", dtype=half, requires_grad=True)
k = torch.ones(1, N, H, D, device="cuda", dtype=half, requires_grad=True)
v = torch.ones(1, N, H, D, device="cuda", dtype=half, requires_grad=True)
cu = torch.tensor([0, 400, 830], device="cuda", dtype=torch.int32)
total = 0.0

def attn_fwd_bwd():
    o, lse, *_ = torch.ops.aten._efficient_attention_forward(q, k, v, None, cu, cu, 512, 512, 0.0, 1, True, scale=None)
    g = torch.ops.aten._efficient_attention_backward(torch.ones_like(o), q, k, v, None, o, cu, cu, 512, 512, lse, 0.0,
                                                    torch.empty((), dtype=torch.int64, device="cuda"),
                                                    torch.empty((), dtype=torch.int64, device="cuda"), 1, False)
total += timed("efficient attention fwd+bwd", attn_fwd_bwd)
x = torch.ones(N, 1536, device="cuda", dtype=half)
xf = torch.ones(N, 1536, device="cuda", dtype=torch.float32)
total += timed("cast half->float, add_, mul", lambda: (x.float().add_(1).mul(2)))
total += timed("isfinite/where", lambda: torch.where(torch.isfinite(xf), xf, xf * 0))
total += timed("index_select / gather", lambda: x.index_select(0, cu[:2].long()))
total += timed("nonzero/cumsum", lambda: (torch.nonzero(xf[:, 0] > 0), torch.cumsum(cu, 0)))
total += timed("sum/amax fp32", lambda: (xf.sum(), xf.amax(1)))
total += timed("repeat_interleave (GQA expand)", lambda: k[:, :, :2].repeat_interleave(6, dim=2))
total += timed("copy_ strided", lambda: torch.empty(1536, N, device="cuda", dtype=half).copy_(x.t()))
print(f"sum of first-call excess: {total:.1f} ms")

# out_dtype availability
a = torch.ones(4, 64, 32, device="cuda", dtype=half); b = torch.ones(4, 32, 16, device="cuda", dtype=half)
for name, fn in [("bmm out_dtype", lambda: torch.bmm(a, b, out_dtype=torch.float32)),
                 ("mm out_dtype", lambda: torch.mm(a[0], b[0], out_dtype=torch.float32)),
                 ("addmm out_dtype", lambda: torch.addmm(torch.zeros(64, 16, device="cuda"), a[0], b[0], out_dtype=torch.float32))]:
    try:
        r = fn(); print(name, "OK", r.dtype)
    except Exception as e:
        print(name, "FAIL", type(e).__name__, str(e)[:100])

# tokenizer save time (Qwen2.5 tokenizer from the local HF cache)
try:
    from transformers import AutoTokenizer
    import tempfile
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    for i in range(3):
        with tempfile.TemporaryDirectory() as d:
            s = time.perf_counter(); tok.save_pretrained(d); e = time.perf_counter() - s
            print(f"tokenizer.save_pretrained #{i}: {e*1e3:.0f} ms, files {sorted(os.listdir(d))}")
except Exception as e:
    print("tokenizer: skipped", type(e).__name__, str(e)[:120])
