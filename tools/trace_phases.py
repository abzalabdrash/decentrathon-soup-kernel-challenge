"""GPU side of a diag chrome trace (solution/diag.py, SOUP_SOLUTION_DIAG_TRACE_CPU=0), per step:

- the compute stream's idle time by phase (forward layers / head / backward layers / optimizer / step
  boundary = from the last optimizer kernel of a step to the first kernel of the next one) and by gap
  length (< 50 us, 50-200 us, 0.2-1 ms, 1-5 ms, > 5 ms), each gap split as in trace_waits.py into CPU
  (the next kernel's launch call had not returned when the gap began) and a dependency wait (the rest);
- the GPU work by category (GEMM fp16 / int8, NF4, our int8 kernels, RMSNorm, RoPE, head, SwiGLU,
  attention, torch elementwise / reductions, optimizer, H2D / D2H copies), ms and launches per step,
  split forward / backward for the compute stream;
- the torch elementwise kernels by name, ms and launches per step and phase.

    python tools/trace_phases.py <diag_trace_*.json.gz> [--skip-gap-ms 1000] [--top 40]

Steps are found as in trace_waits.py (the optimizer's multi_tensor_apply cluster after a head).
"""

import argparse
import collections
import gzip
import json
import re

BUCKETS = ((50, "<50us"), (200, "50-200us"), (1000, "0.2-1ms"), (5000, "1-5ms"), (float("inf"), ">5ms"))
PHASES = ("forward", "head", "backward", "optimizer", "boundary")


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def category(name):
    n = name
    if "Memcpy" in n or "memcpy" in n:
        return "copy H2D" if "HtoD" in n else "copy D2H" if "DtoH" in n else "copy other"
    if "Memset" in n or "memset" in n:
        return "memset"
    if "i8816" in n or "imma" in n or "s8_" in n and "gemm" in n:
        return "GEMM int8"
    if "gemm" in n.lower() or "splitK" in n or "cutlass" in n and "fmha" not in n:
        return "GEMM fp16"
    if "fmha" in n or "Attention" in n:
        return "attention"
    if "multi_tensor_apply" in n:
        return "optimizer"
    if n.startswith("gu_quant") or n.startswith("gu_dequant"):
        return "ours offload int8"
    if n.startswith("nf4_int8") or n.startswith("int8_") or n.startswith("quant_") or "int8" in n:
        return "ours int8"
    if n.startswith("nf4_"):
        return "ours NF4"
    if n.startswith("rms_norm") or "rmsnorm" in n:
        return "ours RMSNorm"
    if n.startswith("rope"):
        return "ours RoPE"
    if n.startswith("head_") or "masked_pair_sum" in n or "fused_ce" in n:
        return "ours head"
    if "swiglu" in n:
        return "SwiGLU (example)"
    if "at::native" in n or "at_cuda_detail" in n or "c10::" in n:
        return "torch elementwise/reduce"
    return "other"


FAMILIES = (("nf4_dequant4", "NF4 attention q|k|v|o (dequant3/4)"), ("nf4_dequant3", "NF4 attention q|k|v|o (dequant3/4)"),
            ("nf4_dequant_kernel", "NF4 MLP matrix (dequant)"), ("nf4_int8_t", "NF4 -> int8 decode (bwd MLP)"),
            ("gu_dequant", "offload (gate|up) int8 unpack"), ("fmha", "attention fmha"),
            ("head_", "head"), ("rms_norm", "RMSNorm"), ("rope", "RoPE"), ("multi_tensor_apply", "optimizer"))


def family(name):
    for prefix, label in FAMILIES:
        if prefix in name:
            return label
    c = category(name)
    return c if c != "other" else "other"


def short_torch(name):
    """A readable key for a torch kernel: the functor / kernel family."""
    for pat in ("fused_dropout_kernel", "direct_copy_kernel_cuda", "float16_copy_kernel_cuda",
                "bfloat16_copy_kernel_cuda", "FillFunctor<[^>]*>", "launch_clamp_scalar", "clamp",
                "floor_kernel_cuda", "CUDAFunctorOnSelf_add<[^>]*>", "CUDAFunctor_add<[^>]*>",
                "BinaryFunctor<[^,]*, [^,]*, [^,]*, at::native::[A-Za-z_]+",
                "AUnaryFunctor<[^,]*, [^,]*, [^,]*, at::native::[A-Za-z_]+",
                "BUnaryFunctor<[^,]*, [^,]*, [^,]*, at::native::[A-Za-z_]+",
                "abs_kernel_cuda", "where_kernel", "compare_scalar_kernel", "MaxOps", "MinOps",
                "func_wrapper_t<[^,]*, at::native::sum_functor", "sum_functor", "norm", "index_elementwise_kernel",
                "vectorized_gather_kernel", "CatArrayBatchedCopy", "radixSort", "maximum", "minimum",
                "mul_kernel", "div_true_kernel", "MulFunctor", "DivFunctor", "neg_kernel", "sqrt",
                "reduce_kernel<[^>]*>", "elementwise_kernel", "unrolled_elementwise_kernel"):
        m = re.search(pat, name)
        if m:
            kind = ("unrolled/" if "unrolled_elementwise" in name else
                    "vectorized/" if "vectorized_elementwise" in name else
                    "elementwise/" if "elementwise_kernel<" in name else "")
            return kind + m.group(0)
    return name[:90]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--skip-gap-ms", type=float, default=1000.0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--debug-recompute", action="store_true")
    args = parser.parse_args()
    events = load(args.trace)
    runtime = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            runtime[e["args"]["correlation"]] = (e["ts"], e["ts"] + e.get("dur", 0))
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset")]
    copies = [e for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy"]
    busy = collections.Counter()
    for k in kernels:
        busy[k["args"].get("stream")] += k["dur"]
    compute = busy.most_common(1)[0][0]
    stream = sorted((k for k in kernels if k["args"].get("stream") == compute), key=lambda k: k["ts"])
    # steps: [first kernel after the previous optimizer cluster, end of this optimizer cluster]
    clusters, seen_head = [], False  # (first optimizer kernel ts, end)
    for k in stream:
        if k["name"].startswith("head_"):
            seen_head = True
        elif "multi_tensor_apply" in k["name"] and seen_head:
            if clusters and k["ts"] - clusters[-1][1] < 20_000:
                clusters[-1][1] = k["ts"] + k["dur"]
            else:
                clusters.append([k["ts"], k["ts"] + k["dur"]])
            seen_head = False
    other = [e for e in kernels + copies if e["args"].get("stream") != compute]
    idle = collections.defaultdict(float)       # (phase, cpu|dep) -> us
    hist = collections.defaultdict(float)       # (phase, bucket, cpu|dep) -> us
    hist_n = collections.Counter()              # (phase, bucket) -> gaps
    work = collections.defaultdict(float)       # (category, phase-ish) -> us
    launches = collections.Counter()
    torch_k = collections.defaultdict(float)
    torch_n = collections.Counter()
    before = collections.defaultdict(float)  # (family, phase, cpu|dep) -> us of the gap before it
    before_n = collections.Counter()
    recomp = collections.defaultdict(float)  # category -> us of the attention recompute's kernels
    recomp_span = recomp_n = 0.0
    spans, nsteps = [], 0
    start = stream[0]["ts"]
    for i, (o0, end) in enumerate(clusters):
        ks = [k for k in stream if start <= k["ts"] < end]
        lo, start = start, end
        heads = [k for k in ks if k["name"].startswith("head_")]
        if len(ks) < 50 or not heads:
            continue
        nxt_first = next((k for k in stream if k["ts"] >= end), None)
        if nxt_first is None or i + 1 >= len(clusters):
            continue  # no boundary after the last traced step
        h0, h1 = heads[0]["ts"], heads[-1]["ts"] + heads[-1]["dur"]
        nsteps += 1
        spans.append((nxt_first["ts"] - ks[0]["ts"]) / 1e3)

        def phase_of(t):
            return "forward" if t <= h0 else "head" if t <= h1 else "backward" if t < o0 else "optimizer"

        pairs = list(zip(ks, ks[1:])) + [(ks[-1], nxt_first)]
        for j, (prev, nxt) in enumerate(pairs):
            g0, g1 = prev["ts"] + prev["dur"], nxt["ts"]
            gap = g1 - g0
            if gap <= 0:
                continue
            if gap > args.skip_gap_ms * 1e3:
                continue
            ph = "boundary" if j == len(pairs) - 1 else phase_of(g1)
            submitted = runtime.get(nxt["args"].get("correlation"), (None, None))[1]
            cpu = 0.0 if submitted is None else min(gap, max(0.0, submitted - g0))
            b = next(label for lim, label in BUCKETS if gap < lim)
            idle[(ph, "cpu")] += cpu
            idle[(ph, "dep")] += gap - cpu
            hist[(ph, b, "cpu")] += cpu
            hist[(ph, b, "dep")] += gap - cpu
            hist_n[(ph, b)] += 1
            fam = family(nxt["name"])
            before[(fam, ph, "cpu")] += cpu
            before[(fam, ph, "dep")] += gap - cpu
            before_n[(fam, ph)] += 1
        # the attention recompute in the backward: from the x16 rebuild before each forward attention
        # kernel to the kernel before the post-attention norm's backward (the o_proj branch included)
        for f, k in enumerate(ks):
            if "fmha_cutlassF" not in k["name"] or phase_of(k["ts"]) != "backward":
                continue
            a = f  # back to the q|k|v|o dequantization (and the x16 rebuild right before it)
            while a > 0 and not ks[a]["name"].startswith(("nf4_dequant3", "nf4_dequant4")) \
                    and "fmha" not in ks[a - 1]["name"] and f - a < 40:
                a -= 1
            if a > 0 and ks[a - 1]["name"].startswith("rms_norm_rows_rebuild"):
                a -= 1
            b = f
            while b + 1 < len(ks) and not ks[b + 1]["name"].startswith("rms_norm_rows_bwd") \
                    and "fmha" not in ks[b + 1]["name"]:
                b += 1
            if args.debug_recompute and recomp_n < 2:
                for kk in ks[a:b + 1]:
                    print(f"    recompute {kk['dur']:7.1f} {kk['name'][:90]}")
            for kk in ks[a:b + 1]:
                recomp[family(kk["name"])] += kk["dur"]
            recomp_span += ks[b]["ts"] + ks[b]["dur"] - ks[a]["ts"]
            recomp_n += 1
        for k in ks:
            ph = phase_of(k["ts"])
            ph2 = "fwd" if ph == "forward" else "bwd" if ph == "backward" else ph
            c = category(k["name"])
            work[(c, ph2)] += k["dur"]
            launches[(c, ph2)] += 1
            if c == "torch elementwise/reduce":
                key = (short_torch(k["name"]), ph2)
                torch_k[key] += k["dur"]
                torch_n[key] += 1
        for e in other:
            if lo <= e["ts"] < end:
                c = category(e["name"])
                work[(c, "side")] += e["dur"]
                launches[(c, "side")] += 1
                if c == "torch elementwise/reduce":
                    key = (short_torch(e["name"]), "side")
                    torch_k[key] += e["dur"]
                    torch_n[key] += 1
    if not nsteps:
        print("no complete step found")
        return
    n = nsteps
    print(f"{n} steps, span (first kernel to next step's first kernel) ms: "
          + ", ".join(f"{s:.1f}" for s in spans) + f"; mean {sum(spans) / n:.1f}")
    print("\ncompute-stream idle per step, ms (CPU = launch not yet returned / dependency = queued, waiting)")
    print(f"{'phase':10s} {'CPU':>7s} {'dep':>7s} {'total':>7s}   " +
          "  ".join(f"{label:>17s}" for _, label in BUCKETS))
    for ph in PHASES:
        c, d = idle[(ph, "cpu")] / n / 1e3, idle[(ph, "dep")] / n / 1e3
        cells = []
        for _, label in BUCKETS:
            cells.append(f"{hist[(ph, label, 'cpu')] / n / 1e3:5.1f}+{hist[(ph, label, 'dep')] / n / 1e3:5.1f}"
                         f" ({hist_n[(ph, label)] / n:4.0f})")
        print(f"{ph:10s} {c:7.2f} {d:7.2f} {c + d:7.2f}   " + "  ".join(f"{x:>17s}" for x in cells))
    tot_c = sum(v for (p, k), v in idle.items() if k == "cpu") / n / 1e3
    tot_d = sum(v for (p, k), v in idle.items() if k == "dep") / n / 1e3
    print(f"{'all':10s} {tot_c:7.2f} {tot_d:7.2f} {tot_c + tot_d:7.2f}   (bucket cells: CPU+dep ms (gaps per step))")
    print("\nGPU work per step by category, ms (launches): compute stream fwd / head / bwd / optimizer; side streams")
    cats = sorted({c for c, _ in work}, key=lambda c: -sum(v for (cc, _), v in work.items() if cc == c))
    cols = ("fwd", "head", "bwd", "optimizer", "side")
    print(f"{'category':26s} " + " ".join(f"{c:>15s}" for c in cols) + f" {'total':>15s}")
    for c in cats:
        cells = [f"{work[(c, p)] / n / 1e3:7.2f} ({launches[(c, p)] / n:5.0f})" for p in cols]
        tot = sum(work[(c, p)] for p in cols) / n / 1e3
        totn = sum(launches[(c, p)] for p in cols) / n
        print(f"{c:26s} " + " ".join(f"{x:>15s}" for x in cells) + f" {tot:7.2f} ({totn:5.0f})")
    comp = sum(v for (c, p), v in work.items() if p != "side") / n / 1e3
    print(f"compute stream kernels {comp:.1f} ms per step; launches {sum(v for (c, p), v in launches.items() if p != 'side') / n:.0f}")
    if recomp_n:
        tot = sum(recomp.values()) / n / 1e3
        print(f"\nattention recompute in the backward: {recomp_n / n:.0f} per step, kernels {tot:.2f} ms per step, "
              f"span {recomp_span / n / 1e3:.2f} ms (rebuild .. o_proj branch):")
        for f, us in sorted(recomp.items(), key=lambda kv: -kv[1]):
            print(f"  {us / n / 1e3:6.2f}  {f}")
    print("\ngaps before a kernel family, ms per step CPU + dependency (gaps per step)")
    fams = sorted({f for f, _, _ in before}, key=lambda f: -sum(v for (ff, p, k), v in before.items() if ff == f))
    for f in fams:
        cells = []
        for ph in ("forward", "head", "backward", "optimizer", "boundary"):
            if before_n[(f, ph)]:
                c, d = before[(f, ph, "cpu")] / n / 1e3, before[(f, ph, "dep")] / n / 1e3
                cells.append(f"{ph} {c:5.2f}+{d:5.2f} ({before_n[(f, ph)] / n:.0f})")
        print(f"  {f:38s} " + "   ".join(cells))
    print(f"\ntorch elementwise / reduce kernels per step (top {args.top}): ms, launches, phase")
    for (name, ph), us in sorted(torch_k.items(), key=lambda kv: -kv[1])[:args.top]:
        print(f"  {us / n / 1e3:6.2f} {torch_n[(name, ph)] / n:6.1f}  {ph:9s} {name}")
    tk = sum(torch_k.values()) / n / 1e3
    tn = sum(torch_n.values()) / n
    print(f"  total {tk:.2f} ms, {tn:.0f} launches per step")


if __name__ == "__main__":
    main()
