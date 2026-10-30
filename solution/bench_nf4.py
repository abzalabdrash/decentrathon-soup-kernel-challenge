"""NF4 dequantization bandwidth on the T4 (measurement only): nf4_dequant_kernel at the 1.5B weight
shapes for a sweep of blocks per program and warps, every output checked bit for bit against the
default launch, next to torch's own copy and fill of the same bytes. The GPU is heated with fp16
GEMMs first and the configurations alternate in rounds, so clocks are close to the training's.

`python -m solution.bench_nf4 [--iters 40] [--rounds 3]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_nf4` runs it on Kaggle)."""

import argparse
import statistics
import time

import torch
import triton
import triton.language as tl


def _time(fn, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


@triton.jit
def nf4_dequant_3d_kernel(words_ptr, qabsmax_ptr, code2_ptr, absmax2_ptr, offset_ptr, nf4_ptr, out_ptr,
                          n_blocks, QBLOCK: tl.constexpr, ABLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    # nf4.py's kernel without the reshape: the (blocks, words, 8) codes are stored through a 3-D
    # index, 8 consecutive fp16 per word (a candidate for fewer layout conversions)
    blk = tl.program_id(0) * BLOCKS + tl.arange(0, BLOCKS)
    live = blk < n_blocks
    q = tl.load(qabsmax_ptr + blk, mask=live, other=0).to(tl.int32)
    scale = tl.load(absmax2_ptr + blk // ABLOCK, mask=live, other=0.0)
    absmax = tl.load(code2_ptr + q) * scale
    absmax = absmax + tl.load(offset_ptr)
    words: tl.constexpr = QBLOCK // 8
    wi = tl.arange(0, words)
    w = tl.load(words_ptr + blk[:, None] * words + wi[None, :], mask=live[:, None], other=0)
    j = tl.arange(0, 8)
    codes = (w[:, :, None] >> ((j // 2) * 8 + (1 - j % 2) * 4)[None, None, :]) & 15
    vals = tl.load(nf4_ptr + codes) * absmax[:, None, None]
    elem = blk[:, None, None] * QBLOCK + wi[None, :, None] * 8 + j[None, None, :]
    tl.store(out_ptr + elem, vals.to(out_ptr.dtype.element_ty), mask=live[:, None, None])


def _heat(seconds):
    a = torch.randn(4096, 4096, device="cuda").half()
    b = torch.randn(4096, 4096, device="cuda").half()
    stop = time.perf_counter() + seconds
    while time.perf_counter() < stop:
        for _ in range(20):
            a @ b
        torch.cuda.synchronize()


def main():
    from solution import nf4
    from solution.test_fused_proj import _linear4bit

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--heat", type=float, default=8.0)
    args = parser.parse_args()
    shapes = {"gate/up (8960x1536)": (1536, 8960), "down (1536x8960)": (8960, 1536),
              "o (1536x1536)": (1536, 1536)}
    layers = {name: _linear4bit(k, n, seed=i) for i, (name, (k, n)) in enumerate(shapes.items())}
    configs = [(b, w) for b in (16, 32, 64, 128) for w in (2, 4, 8)]
    configs += [(b, w, "3d") for b in (32, 64) for w in (4, 8)]

    def launch(layer, out, blocks, warps, kind="2d"):
        qs = layer.weight.quant_state
        n_blocks = qs.absmax.numel()
        kernel = nf4.nf4_dequant_kernel if kind == "2d" else nf4_dequant_3d_kernel
        kernel[(triton.cdiv(n_blocks, blocks),)](
            layer.weight.data.view(-1).view(torch.int32), qs.absmax, qs.state2.code, qs.state2.absmax,
            qs.offset, qs.code, out, n_blocks, QBLOCK=nf4.QBLOCK, ABLOCK=nf4.ABLOCK, BLOCKS=blocks,
            num_warps=warps, enable_fp_fusion=False)

    refs, outs = {}, {}
    for name, layer in layers.items():
        refs[name] = nf4.dequantize_nf4(layer.weight, layer.weight.quant_state, torch.float16)
        outs[name] = torch.empty_like(refs[name])
        for cfg in configs:
            outs[name].zero_()
            launch(layer, outs[name], *cfg)
            assert torch.equal(outs[name], refs[name]), (name, cfg)
    print("every configuration bit-equal to the default launch", flush=True)
    _heat(args.heat)
    times = {(name, cfg): [] for name in layers for cfg in configs + ["copy", "fill"]}
    for rnd in range(args.rounds):
        order = configs if rnd % 2 == 0 else configs[::-1]
        for name, layer in layers.items():
            out = outs[name]
            src = torch.empty(out.numel() * 5 // 4, dtype=torch.float16, device="cuda")
            for cfg in order:
                times[(name, cfg)].append(_time(lambda: launch(layer, out, *cfg), args.iters))
            # references: write the same fp16 bytes reading 1.25x as many (copy), write only (fill)
            times[(name, "copy")].append(_time(lambda: out.view(-1).copy_(src[:out.numel()]), args.iters))
            times[(name, "fill")].append(_time(lambda: out.fill_(1.0), args.iters))
        _heat(2.0)
    for name, layer in layers.items():
        n = refs[name].numel()
        moved = n * 2 + n // 2 + n // 64  # fp16 out, packed codes, absmax codes
        print(f"\n{name}: {n / 1e6:.2f} M weights, {moved / 1e6:.1f} MB moved per dequantization")
        rows = []
        for cfg in configs + ["copy", "fill"]:
            ms = statistics.median(times[(name, cfg)])
            if cfg == "copy":
                gbs = n * 2 * 2 / ms / 1e6
            elif cfg == "fill":
                gbs = n * 2 / ms / 1e6
            else:
                gbs = moved / ms / 1e6
            rows.append((ms, cfg, gbs))
        for ms, cfg, gbs in sorted(rows, key=lambda r: r[0]):
            label = cfg if isinstance(cfg, str) else (f"BLOCKS={cfg[0]:3d} warps={cfg[1]}"
                                                      + (" 3-D store" if len(cfg) > 2 else ""))
            mark = "  <- default" if cfg == (nf4.BLOCKS_PER_PROGRAM, 4) else ""
            print(f"    {ms * 1000:8.1f} us  {gbs:6.1f} GB/s  {label}{mark}", flush=True)


if __name__ == "__main__":
    main()
