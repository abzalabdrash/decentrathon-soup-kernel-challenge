"""Per saved-tensor cost of Python autograd saved_tensors_hooks (research R1), CPU tensors of 16 elements.
CPU tensors keep the GPU out of the timing: the numbers are pure host cost of the hook calls."""
import statistics, time
import torch
dev = "cpu"
x = torch.randn(16, device=dev, dtype=torch.float32, requires_grad=True)
N = 2000  # chain of N elementwise muls -> N saved tensors (mul saves its other operand only when it is a tensor)
w = torch.randn(16, device=dev)
def chain():
    h = x
    for _ in range(N):
        h = h * w          # saves w (a non-requires-grad tensor) -> one saved tensor per mul
    return h.sum()
def run(hooks):
    vals = []
    for _ in range(5):
        
        t0 = time.perf_counter()
        if hooks:
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                loss = chain()
        else:
            loss = chain()
        loss.backward()
        t1 = time.perf_counter()
        
        vals.append((t1 - t0) / N * 1e6)
    return round(statistics.median(vals), 2)
run(False); run(True)
print("per mul fwd+bwd CPU, no hooks    :", run(False), "us")
print("per mul fwd+bwd CPU, python hooks:", run(True), "us")
