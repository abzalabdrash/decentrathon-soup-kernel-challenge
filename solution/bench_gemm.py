"""cuBLAS GEMM efficiency on the training step's exact shapes and layouts (T19, measurement only).

On the T4 our GEMMs take 150-170 ms per step at 30-33 achieved TFLOPS, about half of the card's 65,
with 128x256-tile kernels (`turing_fp16_s1688gemm_fp16_128x256_ldg8_f2f_tn` ...) whose tile counts
suggest wave quantization (a 1,536-wide output at M = 832 is 42 tiles of 128x256 on 40 SMs). This
script times every GEMM of a decoder layer as solution/fused_proj.py issues it (gate|up split, the
offload backward, M a multiple of 16) and alternatives of each:

- library: cuBLAS as torch picks it by default, or `torch.backends.cuda.preferred_blas_library("cublaslt")`;
- form: `as_trained` (the training's call), `bias` (addmm with a zero bias vector: torch's cuBLASLt
  gemm_and_bias path), `transposed` (out^T = B^T @ A^T, the transposed output), `splitk2` (two
  K-halves into one fp32 output with addmm(beta=1); one extra fp32 rounding of the partial sums, and a
  cast back to fp16 where the training's output is fp16);
- `streams2`: the two independent GEMMs of a pair (gate and up forward; gate and up backward into two
  outputs added after) on two streams against one after the other.

For each: the time per call (median of rounds of CUDA-event-timed loops), the achieved TFLOPS and the
kernel names (torch.profiler). At the end, per M: the step total over the 28 layers (forward and
backward) as trained and with the fastest variant of each GEMM. Run on the target GPU:
`python -m solution.bench_gemm [--m 832 1440] [--iters 30] [--rounds 5] [--out gemm.json]`
(`tools/kaggle_run.py --mode bench` runs it on Kaggle). Nothing here touches the training path.
"""

import argparse
import contextlib
import json
import re
import statistics

import torch

D, INTER, QKV, LAYERS = 1536, 8960, 2048, 28
HALF = torch.float16


def _w(rows, cols, gen):
    return (torch.randn(rows, cols, generator=gen, device="cuda") * 0.02).to(HALF)


def make_operands(m, seed=0):
    """Random operands in the training's shapes: weights as rebuilt in fp16, activations token-major
    (M, ·), the MLP's (2I, M) gate|up and (I, M) hidden as fused_proj keeps them."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, D, generator=gen, device="cuda").to(HALF)
    return {
        "x": x, "x_t": x.t(),
        "w_qkv": _w(QKV, D, gen), "b_qkv": torch.zeros(QKV, dtype=HALF, device="cuda"),
        "w_o": _w(D, D, gen), "w_gate": _w(INTER, D, gen), "w_up": _w(INTER, D, gen),
        "w_down": _w(D, INTER, gen),
        "hidden": torch.randn(INTER, m, generator=gen, device="cuda").to(HALF),  # (I, M)
        "g_qkv": torch.randn(m, QKV, generator=gen, device="cuda").to(HALF),
        "g_d": torch.randn(m, D, generator=gen, device="cuda").to(HALF),
        "g_gate_up": torch.randn(2 * INTER, m, generator=gen, device="cuda").to(HALF),  # (2I, M)
        "c32": torch.zeros(m, D, dtype=torch.float32, device="cuda"),
    }


class Gemm:
    """One GEMM of the layer: out = a @ b (a, b as the training passes them, views included),
    out dtype fp16 or fp32, `accumulate` for an addmm(beta=1) into an fp32 C."""

    def __init__(self, name, phase, a, b, out32=False, accumulate=False, bias=None):
        self.name, self.phase, self.a, self.b = name, phase, a, b
        self.out32, self.accumulate, self.bias = out32, accumulate, bias
        self.m, self.k = a.shape
        self.n = b.shape[1]
        self.c32 = torch.zeros(self.m, self.n, dtype=torch.float32, device="cuda")  # fp32 C / split-K out

    @property
    def flops(self):
        return 2.0 * self.m * self.n * self.k

    def as_trained(self):
        if self.accumulate:
            return torch.addmm(self.c32, self.a, self.b, out_dtype=torch.float32, out=self.c32)
        if self.out32:
            return torch.mm(self.a, self.b, out_dtype=torch.float32)
        if self.bias is not None:
            return torch.addmm(self.bias, self.a, self.b)
        return torch.mm(self.a, self.b)

    def with_bias(self):
        if self.out32 or self.accumulate:
            return None  # the fp32-out calls have no fp16 bias form
        bias = self.bias if self.bias is not None else torch.zeros(self.n, dtype=HALF, device="cuda")
        return torch.addmm(bias, self.a, self.b)

    def transposed(self):
        # out^T = b^T @ a^T: the same product, the transposed output (operand layouts swap roles)
        if self.accumulate:
            c = self.c32.t()
            return torch.addmm(c, self.b.t(), self.a.t(), out_dtype=torch.float32, out=c)
        if self.out32:
            return torch.mm(self.b.t(), self.a.t(), out_dtype=torch.float32)
        return torch.mm(self.b.t(), self.a.t())

    def splitk2(self):
        h = self.k // 2
        c = self.c32
        if self.accumulate:
            torch.addmm(c, self.a[:, :h], self.b[:h], out_dtype=torch.float32, out=c)
        else:
            torch.addmm(c, self.a[:, :h], self.b[:h], beta=0, out_dtype=torch.float32, out=c)
        torch.addmm(c, self.a[:, h:], self.b[h:], out_dtype=torch.float32, out=c)
        return c if (self.out32 or self.accumulate) else c.to(HALF)

    FORMS = ("as_trained", "bias", "transposed", "splitk2")

    def call(self, form):
        return {"as_trained": self.as_trained, "bias": self.with_bias, "transposed": self.transposed,
                "splitk2": self.splitk2}[form]


def layer_gemms(ops):
    """The GEMMs of one decoder layer in the order fused_proj.py runs them (gate|up split on, the
    offload backward: no recompute GEMMs)."""
    x, x_t, hid, ggu = ops["x"], ops["x_t"], ops["hidden"], ops["g_gate_up"]
    return [
        Gemm("qkv", "fwd", x, ops["w_qkv"].t(), bias=ops["b_qkv"]),  # addmm(bias, x16, w.t())
        Gemm("o", "fwd", x, ops["w_o"].t()),  # mm(x16, w.t())
        Gemm("gate", "fwd", ops["w_gate"], x_t),  # mm(w, x16.t(), out=gate_up[:I]) -> (I, M)
        Gemm("up", "fwd", ops["w_up"], x_t),
        Gemm("down", "fwd", hid.t(), ops["w_down"].t()),  # mm(hidden.t(), w.t()) -> (M, D)
        Gemm("down", "bwd", ops["w_down"].t(), ops["g_d"].t()),  # mm(w.t(), grad16.t()) -> (I, M)
        Gemm("gate", "bwd", ggu[:INTER].t(), ops["w_gate"], out32=True),  # mm(..., out_dtype=fp32)
        Gemm("up", "bwd", ggu[INTER:].t(), ops["w_up"], accumulate=True),  # addmm(result, ..., fp32)
        Gemm("o", "bwd", ops["g_d"], ops["w_o"]),  # mm(grad, w)
        Gemm("qkv", "bwd", ops["g_qkv"], ops["w_qkv"], accumulate=True),  # addmm(lora_x, g, w, fp32)
    ]


@contextlib.contextmanager
def blas(library):
    previous = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(library)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(previous)


def time_us(fn, iters, rounds):
    """Median over rounds of the mean time of `iters` back-to-back calls, in microseconds."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(times)


def kernel_names(fn):
    """The CUDA kernels one call launches (names, GEMM-like first), from torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = []
    for event in prof.events():
        if event.device_type.name == "CUDA" and event.name not in names:
            names.append(event.name)
    return names


def _short(name):
    """A kernel's name without its argument list: the cutlass template's name, or the cuBLAS one."""
    found = re.search(r"cutlass::Kernel2?<(\w+)", name)
    if found:
        return found.group(1)
    return name.replace("void ", "").split("(")[0].split("<")[0][:90]


def two_streams(first, second, iters, rounds):
    """(sequential, on two streams) time of the pair, microseconds per pair."""
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    main = torch.cuda.current_stream()

    def concurrent():
        s1.wait_stream(main)
        s2.wait_stream(main)
        with torch.cuda.stream(s1):
            first()
        with torch.cuda.stream(s2):
            second()
        main.wait_stream(s1)
        main.wait_stream(s2)

    def sequential():
        first()
        second()

    return time_us(sequential, iters, rounds), time_us(concurrent, iters, rounds)


def bench_m(m, iters, rounds, libraries):
    ops = make_operands(m)
    gemms = layer_gemms(ops)
    rows = []
    for gemm in gemms:
        row = {"gemm": f"{gemm.name}_{gemm.phase}", "m": gemm.m, "n": gemm.n, "k": gemm.k,
               "a": "T" if gemm.a.stride(0) == 1 else "N", "b": "T" if gemm.b.stride(0) == 1 else "N",
               "out": "fp32+=" if gemm.accumulate else ("fp32" if gemm.out32 else "fp16"), "variants": {}}
        for library in libraries:
            with blas(library):
                for form in Gemm.FORMS:
                    fn = gemm.call(form)
                    if fn() is None:
                        continue
                    us = time_us(fn, iters, rounds)
                    row["variants"][f"{library}/{form}"] = {
                        "us": us, "tflops": gemm.flops / us / 1e6,
                        "kernels": [_short(k) for k in kernel_names(fn)]}
        rows.append(row)

    pairs = {}
    by_key = {f"{g.name}_{g.phase}": g for g in gemms}
    gate_f, up_f = by_key["gate_fwd"], by_key["up_fwd"]
    up_out = torch.empty(INTER, m, dtype=HALF, device="cuda")
    pairs["gate|up fwd"] = two_streams(gate_f.as_trained, lambda: torch.mm(up_f.a, up_f.b, out=up_out),
                                       iters, rounds)
    gate_b, up_b = by_key["gate_bwd"], by_key["up_bwd"]
    other = torch.zeros(m, D, dtype=torch.float32, device="cuda")

    def up_alone():
        torch.addmm(other, up_b.a, up_b.b, beta=0, out_dtype=torch.float32, out=other)

    seq, conc = two_streams(gate_b.as_trained, up_alone, iters, rounds)
    add = time_us(lambda: other.add_(gate_b.c32), iters, rounds)
    pairs["gate|up bwd (two outputs + add)"] = (seq, conc + add)
    return rows, pairs


def report(results):
    for m, (rows, pairs) in results.items():
        print(f"\n## M = {m}\n")
        variants = list(dict.fromkeys(v for r in rows for v in r["variants"]))
        print("| GEMM | M x N x K | layout a,b -> out | " + " | ".join(variants) + " | best |")
        print("|---|---|---|" + "---|" * (len(variants) + 1))
        trained = best = 0.0
        for r in rows:
            vs = r["variants"]
            base = vs.get("default/as_trained") or next(iter(vs.values()))
            win = min(vs.items(), key=lambda kv: kv[1]["us"])
            trained += base["us"]
            best += win[1]["us"]
            cells = [f"{vs[v]['us']:.0f} us {vs[v]['tflops']:.1f} TF" if v in vs else "" for v in variants]
            print(f"| {r['gemm']} | {r['m']} x {r['n']} x {r['k']} | {r['a']},{r['b']} -> {r['out']} | "
                  + " | ".join(cells) + f" | {win[0]} ({100 * (1 - win[1]['us'] / base['us']):+.0f}%) |")
        print(f"\nper layer as trained {trained / 1000:.2f} ms, with the best variant of each "
              f"{best / 1000:.2f} ms; x{LAYERS} layers: {LAYERS * trained / 1000:.1f} -> "
              f"{LAYERS * best / 1000:.1f} ms per step ({LAYERS * (trained - best) / 1000:.1f} ms saved)")
        for name, (seq, conc) in pairs.items():
            print(f"two streams, {name}: {seq:.0f} us one after the other, {conc:.0f} us concurrently "
                  f"({100 * (1 - conc / seq):+.0f}%)")
        print("\nkernels (default / cublaslt, as trained):")
        for r in rows:
            for lib in ("default", "cublaslt"):
                v = r["variants"].get(f"{lib}/as_trained")
                if v:
                    print(f"  {r['gemm']:10s} {lib:8s} {', '.join(v['kernels'][:3])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[832, 1440])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--libraries", nargs="+", default=["default", "cublaslt"])
    parser.add_argument("--out")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"{torch.cuda.get_device_name()}, torch {torch.__version__}, cuBLAS "
          f"{torch.backends.cuda.preferred_blas_library()} by default, "
          f"fp16 reduced-precision reduction {torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}")
    results = {m: bench_m(m, args.iters, args.rounds, args.libraries) for m in args.m}
    report(results)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({str(m): {"rows": rows, "pairs": pairs} for m, (rows, pairs) in results.items()}, f,
                      indent=1)


if __name__ == "__main__":
    main()
