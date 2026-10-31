"""offload_int8.py measured: (1) per layer at M tokens, the GPU time of the quantization and the
rebuild against the PCIe time saved (the D2H and H2D of the half (gate|up) and of its int8 buffer,
pinned host memory, CUDA events); (2) the error it adds to the LoRA gradients of a Qwen2.5-1.5B-shaped
block (fused attention with LoRA on q and v, then the fused MLP, both residual, NF4 weights, the
training path's offload with the early copy) against the half offload, for the fp16 and the int8 MLP
backward and the feature groups in --groups. Measurement only.

`python -m solution.bench_offload_int8 [--m 416 832 1440 2048] [--groups 64 128 256]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_offload_int8` runs it on Kaggle)."""

import argparse
import statistics

import torch


def _time(fn, iters=30, inner=1):
    """Median GPU ms of fn; `inner` calls between the events (a short kernel's launch from Python
    would otherwise be timed too: in training the CPU runs ahead of the GPU)."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / inner)
    return statistics.median(times)


def tiles(ms, configs):
    """quant and rebuild per layer for (tokens per program) x (warps) configurations."""
    from solution import offload_int8

    rows = 2 * 8960
    for config in configs:
        offload_int8.TILE_M, offload_int8.WARPS = (int(v) for v in config.split("x"))
        line = []
        for m in ms:
            x = (torch.randn(rows, m, device="cuda") * 2).half()
            spare = torch.empty((rows // 2, m), dtype=x.dtype, device="cuda")
            wire, codec = offload_int8.encode(x, spare)
            quant = _time(lambda: offload_int8.encode(x, spare), inner=10)
            rebuild = _time(lambda: codec.decode(wire), inner=10)
            line.append(f"M={m}: quant {quant:.3f} rebuild {rebuild:.3f}")
        print(f"tile {config:8s} " + "  ".join(line), flush=True)


def kernels_and_copies(ms, layers=28):
    from solution import offload_int8

    offload_int8.compile_now()
    rows = 2 * 8960
    print(f"per layer ((gate|up) of {rows} rows, group {offload_int8.GROUP}); x{layers} = per step")
    print(f"{'M':>6s} {'quant':>7s} {'rebuild':>8s} {'D2H f16':>8s} {'D2H i8':>7s} {'H2D f16':>8s} "
          f"{'H2D i8':>7s} {'PCIe saved':>11s} {'GPU added':>10s}  (ms)")
    for m in ms:
        x = (torch.randn(rows, m, device="cuda") * 2).half()
        wire, codec = offload_int8.encode(x)
        host16 = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        host8 = torch.empty(wire.shape, dtype=wire.dtype, pin_memory=True)
        quant = _time(lambda: offload_int8.encode(x), inner=10)
        rebuild = _time(lambda: codec.decode(wire), inner=10)
        d2h16 = _time(lambda: host16.copy_(x, non_blocking=True))
        d2h8 = _time(lambda: host8.copy_(wire, non_blocking=True))
        h2d16 = _time(lambda: x.copy_(host16, non_blocking=True))
        h2d8 = _time(lambda: wire.copy_(host8, non_blocking=True))
        saved = d2h16 - d2h8 + h2d16 - h2d8
        added = quant + rebuild
        print(f"{m:6d} {quant:7.3f} {rebuild:8.3f} {d2h16:8.3f} {d2h8:7.3f} {h2d16:8.3f} {h2d8:7.3f} "
              f"{saved:11.3f} {added:10.3f}   per step: PCIe -{layers * saved:.1f} ms, GPU "
              f"+{layers * added:.1f} ms, D2H {wire.numel() / 2**20:.1f} MiB instead of "
              f"{x.numel() * 2 / 2**20:.1f}", flush=True)
        del x, wire, host16, host8


def _patch():
    from solution import fused_proj
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm
    from solution.rope import patch_qwen2_rope
    from solution.unpad import patch_qwen2_attention

    patch_qwen2_attention()
    patch_qwen2_rope()
    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    patch_qwen2_rmsnorm()
    patch_qwen2_decoder_layer()
    fused_proj.patch_qwen2_attention_fused()
    fused_proj.patch_qwen2_mlp_fused()
    fused_proj.set_mlp_recompute(False)


def gradient_error(lengths_list, groups):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import int8_bwd, offload, offload_int8
    from solution.test_fused_proj import (_config, _lora_params, _positions, _quantize_attention,
                                          _quantize_mlp, _rel, _spans)
    from solution.unpad import varlen

    _patch()
    config = _config()
    attn = _quantize_attention(mq.Qwen2Attention(config, 0), seed=3).cuda().train()
    mlp = _quantize_mlp(mq.Qwen2MLP(config), seed=20).cuda().train()
    rotary = mq.Qwen2RotaryEmbedding(config=config).cuda()
    weights = [*attn.parameters(), *mlp.parameters()]
    int8_bwd.compile_now()
    offload_int8.compile_now()

    def run(x, pe, grad, lengths, bwd_int8, gu_int8):
        int8_bwd._STATE["on"] = bwd_int8
        offload_int8._STATE["on"] = gu_int8
        for p in weights:
            p.grad = None
        x = x.detach().requires_grad_(True)
        with offload.offload_saved_tensors(weights=weights) as off, \
                torch.autocast("cuda", dtype=torch.float16), varlen(_spans(lengths)):
            a, _ = attn(x, pe, None)
            h = x + a
            out = h + mlp(h)
            out.backward(grad)
        torch.cuda.synchronize()
        assert off.stats["int8"] == (1 if gu_int8 else 0), off.summary()
        return x.grad, [p.grad.clone() for p in _lora_params(attn)]

    print("\nrelative error against fp16 MLP backward + half offload (dL/dx; LoRA A_q B_q A_v B_v)")
    for lengths in lengths_list:
        g = torch.Generator("cuda").manual_seed(sum(lengths))
        x = torch.randn(1, sum(lengths), 1536, device="cuda", generator=g)
        pe = rotary(x, _positions(lengths))
        grad = torch.randn(1, sum(lengths), 1536, device="cuda", generator=g)
        ref_dx, ref_lora = run(x, pe, grad, lengths, False, False)
        rows = [("int8 MLP backward, half offload", True, False, None)]
        for group in groups:
            rows += [(f"fp16 backward, int8 offload g{group}", False, True, group),
                     (f"int8 backward, int8 offload g{group}", True, True, group)]
        for label, bwd, gu, group in rows:
            if group is not None:
                offload_int8.GROUP = group
                offload_int8.TILE_M = 64 if group <= 128 else 32
            dx, lora = run(x, pe, grad, lengths, bwd, gu)
            errs = [_rel(a, b) for a, b in zip(lora, ref_lora)]
            print(f"  {sum(lengths):5d} tokens  {label:36s} dx {_rel(dx, ref_dx):.4f}  LoRA "
                  + " ".join(f"{e:.4f}" for e in errs), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[416, 832, 1440, 2048])
    parser.add_argument("--groups", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--skip-error", action="store_true")
    parser.add_argument("--tiles", nargs="*", default=[], metavar="TMxWARPS")
    args = parser.parse_args()
    if args.tiles:
        tiles(args.m, args.tiles)
    kernels_and_copies(args.m)
    if not args.skip_error:
        gradient_error([[212, 187, 310, 117], [512, 512, 512, 512]], args.groups)


if __name__ == "__main__":
    main()
