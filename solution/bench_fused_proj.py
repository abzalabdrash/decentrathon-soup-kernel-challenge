"""Microbenchmark of the fused projections (fused_proj.py) against the path they replace (main's unpad
attention with rope and nf4, mlp16's MLP), one Qwen2.5-1.5B layer on x (1, 826, 1536) fp32 in four
rows, LoRA r=16 on q and v, under fp16 autocast.

    python solution/bench_fused_proj.py              # LoRA dropout 0.05 (Soup's default) and 0
    python solution/bench_fused_proj.py --iters 50

Per module and path, forward and backward apart: "gpu" ms as the summed time of the CUDA kernels
of one call from torch.profiler (what the call costs inside a training step where the CPU runs
ahead), "cpu" ms to issue the call from Python (perf_counter from an idle GPU, median of `--iters`;
what the call costs where the CPU is the bottleneck), and the number of kernel launches. The two
paths alternate call by call, so clock changes and other work on the GPU hit both alike.
"""

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile


def _setup(dropout):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj
    from solution import test_fused_proj as t
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rope import patch_qwen2_rope
    from solution.unpad import patch_qwen2_attention

    patch_qwen2_attention()
    patch_qwen2_rope()
    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    fused_proj.patch_qwen2_attention_fused()
    fused_proj.patch_qwen2_mlp_fused()
    config = t._config()
    attn = t._quantize_attention(mq.Qwen2Attention(config, 0), seed=3, dropout=dropout).cuda()
    mlp = t._quantize_mlp(mq.Qwen2MLP(config), seed=20).cuda()
    rotary = mq.Qwen2RotaryEmbedding(config=config).cuda()
    attn.train()
    mlp.train()
    x, pe, grad = t._attention_inputs(rotary)
    return t, attn, mlp, x, pe, grad


def _profile(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return len(events), sum(e.device_time for e in events) / 1000.0


def _cpu(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed * 1000.0


class _Case:
    """One module on one path: forward and backward as separate calls, LoRA grads set to None
    before each forward (as zero_grad does), the input a non-leaf as in the model."""

    def __init__(self, module, forward, x, grad):
        self.module, self.forward, self.x, self.grad = module, forward, x, grad
        self.params = [p for p in module.parameters() if p.requires_grad]
        self.out = None

    def fwd(self):
        for p in self.params:
            p.grad = None
        inp = self.x.detach().requires_grad_(True) * 1.0
        self.out = self.forward(self.module, inp)

    def bwd(self):
        self.out.backward(self.grad.to(self.out.dtype))
        self.out = None


def _measure(cases, iters):
    """{path: (gpu fwd, gpu bwd, cpu fwd, cpu bwd, launches fwd, launches bwd)}."""
    from solution import fused_proj

    rows = {on: {"gf": [], "gb": [], "cf": [], "cb": []} for on in cases}
    for _ in range(3):
        for on, case in cases.items():
            fused_proj._STATE["enabled"] = on
            case.fwd()
            case.bwd()
    counts = {}
    for i in range(iters):
        for on, case in cases.items():
            fused_proj._STATE["enabled"] = on
            rows[on]["cf"].append(_cpu(case.fwd))
            rows[on]["cb"].append(_cpu(case.bwd))
            if i < 5:
                n_fwd, gpu_fwd = _profile(case.fwd)
                n_bwd, gpu_bwd = _profile(case.bwd)
                rows[on]["gf"].append(gpu_fwd)
                rows[on]["gb"].append(gpu_bwd)
                counts[on] = (n_fwd, n_bwd)
    med = statistics.median
    return {on: (med(r["gf"]), med(r["gb"]), med(r["cf"]), med(r["cb"])) + counts[on]
            for on, r in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    from solution import fused_proj

    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, {args.iters} iterations")
    print("| module | dropout | path | fwd gpu | bwd gpu | fwd cpu | bwd cpu | launches fwd/bwd |")
    print("|---|---|---|---|---|---|---|---|")
    for dropout in (0.05, 0.0):
        t, attn, mlp, x, pe, grad = _setup(dropout)
        modules = [("attention", attn, lambda m, v: t._attention_forward(m, v, pe))]
        if dropout:
            modules.append(("mlp", mlp, t._mlp_forward))
        for name, module, forward in modules:
            cases = {on: _Case(module, forward, x, grad) for on in (False, True)}
            for on, (gf, gb, cf, cb, nf, nb) in _measure(cases, args.iters).items():
                print(f"| {name} | {dropout} | {'fused' if on else 'main'} | {gf:.3f} | {gb:.3f} | "
                      f"{cf:.3f} | {cb:.3f} | {nf}/{nb} |")
    fused_proj._STATE["enabled"] = True


if __name__ == "__main__":
    main()
