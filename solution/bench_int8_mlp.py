"""The fused MLP's backward at the 1.5B shapes, fp16 against int8 (int8_bwd.py): GPU time of every
kernel (torch.profiler) and the wall time per call, for M tokens in --m. Measurement only.

`python -m solution.bench_int8_mlp [--m 832 1440] [--iters 20]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_int8_mlp` runs it on Kaggle)."""

import argparse
import collections
import statistics

import torch


def _mlp():
    from solution.test_fused_proj import _config, _quantize_mlp
    from transformers.models.qwen2 import modeling_qwen2 as mq

    return _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()


def _patch():
    from solution.fused_proj import patch_qwen2_mlp_fused
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    patch_qwen2_mlp_fused()


def _step(mlp, x, grad):
    xg = x.detach().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.float16):
        out = mlp(xg)
    out.backward(grad.to(out.dtype))
    return xg.grad


def main():
    from solution import int8_bwd

    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[832, 1440])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tiles", nargs="*", default=[],
                        help="SwiGLU-backward tile configs TIxTMxGROUPS to time (int8 only)")
    args = parser.parse_args()
    _patch()
    mlp = _mlp()
    for m in args.m:
        gen = torch.Generator("cuda").manual_seed(m)
        x = torch.randn(1, m, 1536, device="cuda", generator=gen)
        grad = (torch.randn(1, m, 1536, device="cuda", generator=gen) * 1e-2).half()
        configs = [(False, None), (True, None), (False, None), (True, None)]
        configs += [(True, tuple(int(v) for v in t.split("x"))) for t in args.tiles]
        for on, tiles in configs:
            int8_bwd._STATE["on"] = on
            if tiles is not None:
                int8_bwd.TILE_I, int8_bwd.TILE_M, int8_bwd.GROUPS = tiles
            for _ in range(3):
                _step(mlp, x, grad)
            torch.cuda.synchronize()
            times = []
            for _ in range(args.iters):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                _step(mlp, x, grad)
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end))
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                _step(mlp, x, grad)
                torch.cuda.synchronize()
            kernels = collections.Counter()
            for evt in prof.events():
                if evt.device_type.name == "CUDA":
                    kernels[evt.name[:70]] += evt.device_time_total / 1000.0
            total = sum(kernels.values())
            label = ("int8" if on else "fp16") + (f" tiles {tiles}" if tiles else "")
            print(f"M={m} {label}: fwd+bwd {statistics.median(times):.3f} ms per call "
                  f"(GPU kernels {total:.3f} ms)", flush=True)
            for name, ms in kernels.most_common(12):
                print(f"    {ms:7.3f} ms  {name}", flush=True)


if __name__ == "__main__":
    main()
