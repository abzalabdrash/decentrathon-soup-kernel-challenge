"""Microbenchmark of the fused MLP's memory modes (fused_proj.set_mlp_recompute): one Qwen2.5-1.5B
MLP on NF4 weights, x (1, N, 1536) fp32 under fp16 autocast, (gate|up) saved against x16 saved and
(gate|up) recomputed in the backward, with the gate|up weights in one buffer or split.

    python solution/bench_mlp_memory.py                 # N = 767, 826, 1427
    python solution/bench_mlp_memory.py --tokens 832 --iters 30

Per case: GPU ms of the forward and of the backward (summed CUDA kernel time of one call from
torch.profiler, median of `--iters`), the bytes the forward saves and the peak allocated above the
input during forward + backward. The modes alternate call by call. Saved-mode numbers here exclude
the offload's PCIe copies and allocator stalls (measured on the T4 by the diag run).
"""

import argparse
import os
import statistics

import torch
from torch.profiler import ProfilerActivity, profile


def _mlp():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj
    from solution import test_fused_proj as t
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    fused_proj.patch_qwen2_mlp_fused()
    return t._quantize_mlp(mq.Qwen2MLP(t._config()), seed=20).cuda().train()


def _gpu_ms(fn):
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = fn()
        torch.cuda.synchronize()
    ms = sum(e.device_time for e in prof.events()
             if e.device_type == torch.autograd.DeviceType.CUDA) / 1000
    return ms, result


def _forward(mlp, x):
    with torch.autocast("cuda", dtype=torch.float16):
        return mlp(x)


def _one(mlp, x, grad):
    saved = []

    def pack(t):
        saved.append(t.numel() * t.element_size())
        return t

    inp = x.detach().requires_grad_(True) * 1.0  # not a leaf, as the norm output in the model
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fwd_ms, out = _gpu_ms(lambda: _forward(mlp, inp))
    bwd_ms, _ = _gpu_ms(lambda: out.backward(grad))
    peak = torch.cuda.max_memory_allocated() - base
    return fwd_ms, bwd_ms, sum(saved), peak


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, nargs="*", default=[767, 826, 1427])
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from solution import fused_proj

    mlp = _mlp()
    print(f"{torch.cuda.get_device_name()}; fwd / bwd GPU ms (median), saved MiB, peak MiB above input")
    for n in args.tokens:
        g = torch.Generator("cuda").manual_seed(n)
        x = torch.randn(1, n, 1536, device="cuda", generator=g)
        grad = torch.randn(1, n, 1536, device="cuda", generator=g, dtype=torch.float16)
        for split in ("0", "1"):
            os.environ["SOUP_SOLUTION_FUSED_GU_SPLIT"] = split
            rows = {False: [], True: []}
            for _ in range(args.iters + 2):  # 2 warm-up rounds
                for recompute in (False, True):
                    fused_proj.set_mlp_recompute(recompute)
                    rows[recompute].append(_one(mlp, x, grad))
            for recompute, values in rows.items():
                values = values[2:]
                fwd = statistics.median(v[0] for v in values)
                bwd = statistics.median(v[1] for v in values)
                print(f"N {n:5d} split {split} {'recompute' if recompute else 'saved    '}: fwd "
                      f"{fwd:6.3f} bwd {bwd:6.3f} total {fwd + bwd:6.3f} ms; saved "
                      f"{values[-1][2] / 2**20:5.1f} MiB, peak {values[-1][3] / 2**20:6.1f} MiB")
    os.environ.pop("SOUP_SOLUTION_FUSED_GU_SPLIT", None)
    fused_proj.set_mlp_recompute(True)


if __name__ == "__main__":
    main()
