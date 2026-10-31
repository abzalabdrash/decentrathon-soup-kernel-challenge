"""B3, step 1 (measurement only): our tensor-core GEMMs (solution/mma_gemm.py) against cuBLAS on the
T4, on the training's shapes.

For every shape and M: cuBLAS (torch.mm in fp16, as fused_proj runs it), our tc_matmul_kernel over
a small set of tile configurations (the best one is reported, all are logged), and, for the frozen
layers, "NF4 decode + cuBLAS" (solution/nf4.py's kernel writing the fp16 weight, then torch.mm, as
training does today) against tc_nf4_matmul_kernel (decode inside the tile). int8: torch._int_mm
(cuBLASLt IMMA) against tc_matmul_i8_kernel. Each compiled kernel's PTX is checked for
mma.sync.aligned.m16n8k8 / m8n8k16 and for the absence of cp.async; registers and spills are
printed. Correctness: max |ours - reference| / max |reference|.

`python -m solution.bench_mma_gemm [--m 512 832 1440 2048] [--iters 10] [--rounds 5] [--out mma.json]`
(`tools/kaggle_run.py --mode bench --bench solution.bench_mma_gemm` runs it on Kaggle).
"""

import argparse
import json
import re
import time

import torch

from . import mma_gemm
from .bench_int8 import _Clock, _time

D, INTER, QKV, HEAD_CHUNK = 1536, 8960, 2048, 4096

FWD = [("gate_up", D, 2 * INTER), ("down", INTER, D), ("qkv", D, QKV), ("o", D, D), ("head", D, HEAD_CHUNK)]
BWD = [("gate_up_bwd", 2 * INTER, D), ("down_bwd", D, INTER), ("qkv_bwd", QKV, D), ("o_bwd", D, D)]

CFG16 = [(128, 128, 32, 4, 8), (128, 128, 32, 8, 8), (128, 256, 32, 8, 8), (256, 128, 32, 8, 8),
         (64, 128, 32, 4, 8), (128, 64, 32, 4, 8), (128, 128, 64, 8, 8), (64, 64, 32, 4, 8)]
CFG8 = [(128, 128, 64, 4, 8), (128, 128, 64, 8, 8), (128, 256, 64, 8, 8), (256, 128, 64, 8, 8),
        (64, 128, 64, 4, 8), (128, 128, 128, 8, 8)]
CFGNF4 = [(128, 128, 64, 4, 8), (128, 128, 64, 8, 8), (128, 256, 64, 8, 8), (256, 128, 64, 8, 8),
          (64, 128, 64, 4, 8), (64, 64, 64, 4, 8)]


def tflops(m, k, n, us):
    return 2.0 * m * k * n / (us * 1e-6) / 1e12


def ptx_report(compiled):
    ptx = compiled.asm.get("ptx", "")
    mma = len(re.findall(r"mma\.sync\.aligned\.(m16n8k8|m8n8k16)", ptx))
    fma = len(re.findall(r"\bfma\.rn\.f32\b", ptx))
    return {"mma": mma, "fma_f32": fma, "cp_async": "cp.async" in ptx, "ldmatrix": "ldmatrix" in ptx,
            "regs": getattr(compiled, "n_regs", None), "spills": getattr(compiled, "n_spills", None),
            "shared": compiled.metadata.shared}


def rel_err(x, ref):
    return float((x.float() - ref.float()).abs().max() / ref.float().abs().max().clamp_min(1e-12))


def best_of(fn_for_cfg, cfgs, iters, rounds, check):
    """Times every config (compile errors are logged and skipped); returns (best_us, best_cfg, log)."""
    log, best = [], (float("inf"), None)
    for cfg in cfgs:
        try:
            compiled, out = fn_for_cfg(cfg)
            torch.cuda.synchronize()
            err = check(out)
            us = _time(lambda: fn_for_cfg(cfg), iters, rounds)
            rep = ptx_report(compiled)
        except Exception as exc:  # noqa: BLE001 - a config that does not compile or fit is skipped
            log.append({"cfg": cfg, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            continue
        log.append({"cfg": cfg, "us": round(us, 1), "err": err, **rep})
        if us < best[0]:
            best = (us, cfg)
    return best[0], best[1], log


def bench_fp16(ms, iters, rounds, gen, results):
    for name, k, n in FWD + BWD:
        fwd = name in dict((f[0], 1) for f in FWD)
        for m in ms:
            a = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).half()
            # the weight as stored: (out, in) row-major; forward x @ W.T (k = in), backward g @ W (k = out)
            w = (torch.randn((n, k) if fwd else (k, n), device="cuda", generator=gen) * 0.05).half()
            b = w.t() if fwd else w
            ref = torch.mm(a, b)
            with _Clock() as clock:
                us_cublas = _time(lambda: torch.mm(a, b), iters, rounds)
                us, cfg, log = best_of(lambda c: mma_gemm.matmul(a, b, c), CFG16, iters, rounds,
                                       lambda out: rel_err(out, ref))
                us_cublas2 = _time(lambda: torch.mm(a, b), iters, rounds)
            cub = min(us_cublas, us_cublas2)
            mhz, watts = clock.mean()
            row = {"kind": "fp16", "shape": name, "m": m, "k": k, "n": n, "cublas_us": round(cub, 1),
                   "cublas_tflops": round(tflops(m, k, n, cub), 2), "tc_us": round(us, 1),
                   "tc_tflops": round(tflops(m, k, n, us), 2) if cfg else None, "tc_cfg": cfg,
                   "ratio": round(cub / us, 3) if cfg else None, "mhz": mhz, "watts": watts, "configs": log}
            results.append(row)
            print(f"fp16 {name:12s} M={m:5d}  cuBLAS {cub:8.1f} us {row['cublas_tflops']:6.2f} TF | "
                  f"tc {us:8.1f} us {row['tc_tflops']} TF cfg {cfg} | ours/cuBLAS speed {row['ratio']} | "
                  f"{mhz} MHz", flush=True)
            for entry in log:
                print("    ", entry, flush=True)


def bench_nf4(ms, iters, rounds, gen, results):
    import bitsandbytes.functional as bnbf

    from .nf4 import dequantize_nf4

    for name, k, n in FWD[:4] + BWD:
        fwd = name in dict((f[0], 1) for f in FWD)
        rows, cols = (n, k) if fwd else (k, n)  # the stored (out, in) weight
        w16 = (torch.randn(rows, cols, device="cuda", generator=gen) * 0.05).half()
        packed, qs = bnbf.quantize_4bit(w16, quant_type="nf4", compress_statistics=True, blocksize=64)
        packed = packed.view(-1)
        wdq = dequantize_nf4(packed, qs)  # bit-for-bit bitsandbytes' fp16 weight
        buf = torch.empty_like(wdq)
        for m in ms:
            a = (torch.randn(m, k, device="cuda", generator=gen) * 0.5).half()
            ref = torch.mm(a, wdq.t() if fwd else wdq)

            def today():
                wd = dequantize_nf4(packed, qs, out=buf)
                return torch.mm(a, wd.t() if fwd else wd)

            with _Clock() as clock:
                us_today = _time(today, iters, rounds)
                us_deq = _time(lambda: dequantize_nf4(packed, qs, out=buf), iters, rounds)
                us, cfg, log = best_of(
                    lambda c: mma_gemm.matmul_nf4(a, packed, qs, n, fwd, c), CFGNF4, iters, rounds,
                    lambda out: rel_err(out, ref))
                us_today2 = _time(today, iters, rounds)
            tod = min(us_today, us_today2)
            mhz, watts = clock.mean()
            row = {"kind": "nf4", "shape": name, "m": m, "k": k, "n": n, "today_us": round(tod, 1),
                   "dequant_us": round(us_deq, 1), "tc_nf4_us": round(us, 1), "tc_cfg": cfg,
                   "gain_us": round(tod - us, 1) if cfg else None, "mhz": mhz, "watts": watts,
                   "configs": log}
            results.append(row)
            print(f"nf4  {name:12s} M={m:5d}  dequant+cuBLAS {tod:8.1f} us (dequant {us_deq:6.1f}) | "
                  f"tc_nf4 {us:8.1f} us cfg {cfg} | gain {row['gain_us']} us | {mhz} MHz", flush=True)
            for entry in log:
                print("    ", entry, flush=True)


def bench_int8(ms, iters, rounds, gen, results):
    for name, k, n in [("gate_up", D, 2 * INTER), ("down", INTER, D), ("gate_up_bwd", 2 * INTER, D),
                       ("down_bwd", D, INTER)]:
        for m in ms:
            a = torch.randint(-127, 128, (m, k), device="cuda", dtype=torch.int8, generator=gen)
            w = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=gen)
            ref = torch._int_mm(a, w.t())
            with _Clock() as clock:
                us_cub = _time(lambda: torch._int_mm(a, w.t()), iters, rounds)
                us, cfg, log = best_of(lambda c: mma_gemm.matmul_i8(a, w, c), CFG8, iters, rounds,
                                       lambda out: float((out != ref).sum()))
                us_cub2 = _time(lambda: torch._int_mm(a, w.t()), iters, rounds)
            cub = min(us_cub, us_cub2)
            mhz, watts = clock.mean()
            row = {"kind": "int8", "shape": name, "m": m, "k": k, "n": n, "cublas_us": round(cub, 1),
                   "cublas_tops": round(tflops(m, k, n, cub), 2), "tc_us": round(us, 1),
                   "tc_tops": round(tflops(m, k, n, us), 2) if cfg else None, "tc_cfg": cfg,
                   "ratio": round(cub / us, 3) if cfg else None, "mhz": mhz, "configs": log}
            results.append(row)
            print(f"int8 {name:12s} M={m:5d}  _int_mm {cub:8.1f} us {row['cublas_tops']:6.2f} TOPS | "
                  f"tc {us:8.1f} us {row['tc_tops']} TOPS cfg {cfg} | ours/cuBLAS speed {row['ratio']}",
                  flush=True)
            for entry in log:
                print("    ", entry, flush=True)


CFGBWD = [(128, 128, 64, 8, 8), (128, 128, 64, 4, 8), (256, 128, 64, 8, 8), (128, 256, 64, 8, 8),
          (64, 128, 64, 4, 8), (128, 64, 64, 4, 8)]


def bench_swiglu_bwd(ms, iters, rounds, gen, results):
    """int8_bwd.py steps 3-4: torch._int_mm + int8_swiglu_bwd_kernel against the fused kernel."""
    import triton

    from . import int8_bwd

    inter, d = 8960, 1536
    for m in ms:
        w = torch.randint(-127, 128, (inter, d), device="cuda", dtype=torch.int8, generator=gen)
        g = torch.randint(-127, 128, (m, d), device="cuda", dtype=torch.int8, generator=gen)
        w_scale = torch.rand(inter, device="cuda", generator=gen) * 1e-3
        g_scale = torch.rand(m, device="cuda", generator=gen) * 1e-3
        gate_up0 = (torch.randn(2 * inter, m, device="cuda", generator=gen) * 2).half()
        gate_up = gate_up0.clone()
        groups = int8_bwd.GROUPS
        pmax = torch.empty((groups, m), dtype=torch.float32, device="cuda")

        def today():
            acc = torch._int_mm(w, g.t())
            int8_bwd.int8_swiglu_bwd_kernel[(triton.cdiv(m, int8_bwd.TILE_M), groups)](
                acc, w_scale, g_scale, gate_up, pmax, inter, m, inter // groups,
                TI=int8_bwd.TILE_I, TM=int8_bwd.TILE_M)

        ref = gate_up0.clone()
        gate_up.copy_(gate_up0)
        today()
        ref.copy_(gate_up)
        log, best = [], (float("inf"), None)
        with _Clock() as clock:
            us_today = _time(today, iters, rounds)
            for cfg in CFGBWD:
                try:
                    gate_up.copy_(gate_up0)
                    mma_gemm.int8_swiglu_bwd(w, g, w_scale, g_scale, gate_up, cfg)
                    torch.cuda.synchronize()
                    same = bool(torch.equal(gate_up.view(torch.int16), ref.view(torch.int16)))
                    us = _time(lambda: mma_gemm.int8_swiglu_bwd(w, g, w_scale, g_scale, gate_up, cfg),
                               iters, rounds)
                except Exception as exc:  # noqa: BLE001
                    log.append({"cfg": cfg, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                    continue
                log.append({"cfg": cfg, "us": round(us, 1), "bitwise": same})
                if us < best[0]:
                    best = (us, cfg)
            us_today2 = _time(today, iters, rounds)
        tod = min(us_today, us_today2)
        mhz, watts = clock.mean()
        row = {"kind": "swiglu_bwd", "m": m, "today_us": round(tod, 1), "fused_us": round(best[0], 1),
               "cfg": best[1], "gain_us": round(tod - best[0], 1), "int32_mib": round(inter * m * 4 / 2**20, 1),
               "mhz": mhz, "configs": log}
        results.append(row)
        print(f"swiglu_bwd M={m:5d}  _int_mm+kernel {tod:8.1f} us | fused {best[0]:8.1f} us cfg {best[1]} | "
              f"gain {row['gain_us']} us | int32 buffer saved {row['int32_mib']} MiB | {mhz} MHz", flush=True)
        for entry in log:
            print("    ", entry, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[512, 832, 1440, 2048])
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--parts", default="fp16,nf4,int8")
    parser.add_argument("--out", default="runs_bench/mma_gemm.json")
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    import triton

    print("triton", triton.__version__, flush=True)
    gen = torch.Generator(device="cuda").manual_seed(0)
    results = []
    t0 = time.time()
    parts = args.parts.split(",")
    try:
        if "fp16" in parts:
            bench_fp16(args.m, args.iters, args.rounds, gen, results)
        if "nf4" in parts:
            bench_nf4(args.m, args.iters, args.rounds, gen, results)
        if "bwd" in parts:
            bench_swiglu_bwd(args.m, args.iters, args.rounds, gen, results)
        if "int8" in parts:
            bench_int8([m for m in args.m if m >= 832] or args.m, args.iters, args.rounds, gen, results)
    finally:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out} ({len(results)} rows) in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
