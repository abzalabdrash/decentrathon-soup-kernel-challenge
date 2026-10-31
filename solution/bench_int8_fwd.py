"""The fused MLP's forward on int8 tensor cores (int8_fwd.py, task I1) against its fp16 forward:
timing at the 1.5B shapes and the quantization error on the real model's activations. Measurement
only.

`python -m solution.bench_int8_fwd [--m 256 512 832 1440 2048] [--iters 30] [--rounds 5]`:
GPU time per MLP forward (CUDA events around `iters` calls of fused_proj._MLP.forward, the variants
interleaved round by round: the T4 throttles as it heats), fp16 against int8 "all" (gate|up and down)
and "gu" (gate|up only), plus the kernels of one call (torch.profiler).

`python -m solution.bench_int8_fwd --errors [--rows 32] [--data data/train.jsonl]`: Qwen2.5-1.5B in
NF4 as Soup loads it (double quant, fp16 compute, fp32 norms, autocast fp16), the first rows of the
training file (alpaca -> chat template, at most 512 tokens) one at a time. Every MLP computes the
fp16 forward (fused_proj's math) and both int8 variants on the same input; per layer the relative
error (Frobenius) of gate|up, of the SwiGLU output and of the MLP output, the worst token's error and
its position, and the error of int8 down alone on the fp16 SwiGLU output. Then the loss of every row
(assistant tokens) with all MLP forwards in fp16, in int8 "all" and in int8 "gu", and with the
layers of --fp16-layers kept in fp16."""

import argparse
import collections
import json
import statistics
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _patch():
    from solution.fused_proj import patch_qwen2_mlp_fused
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit

    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    patch_qwen2_mlp_fused()


def _ops(*bases):
    from solution.fused_proj import _nf4_args

    return tuple(_nf4_args(b, torch.float16)[1] for b in bases)


def _profile(fn):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = collections.Counter()
    for evt in prof.events():
        if evt.device_type.name == "CUDA":
            kernels[evt.name[:60]] += evt.device_time_total / 1000.0
    return kernels


def timing(args):
    from solution import fused_proj, int8_fwd
    from solution.test_fused_proj import _config, _quantize_mlp
    from transformers.models.qwen2 import modeling_qwen2 as mq

    _patch()
    int8_fwd._STATE["on"] = True
    int8_fwd.compile_now()
    for mode in ("all", "gu"):  # both variants' kernels
        int8_fwd.MODE = mode
        int8_fwd._warm()
    mlp = _quantize_mlp(mq.Qwen2MLP(_config()), seed=20).cuda().train()
    ops = _ops(mlp.gate_proj, mlp.up_proj, mlp.down_proj)
    ctx = SimpleNamespace(save_for_backward=lambda *a: None)
    results = {}
    for m in args.m:
        x = torch.randn(1, m, 1536, device="cuda", generator=torch.Generator("cuda").manual_seed(m))
        plan = fused_proj._mlp_plan(mlp, x)
        variants = {"fp16": None, "all": ops, "gu": ops}

        def call(name):
            plan.int8 = variants[name]
            int8_fwd.MODE = name if name != "fp16" else int8_fwd.MODE
            return fused_proj._MLP.forward(ctx, x, plan)

        for name in variants:
            for _ in range(3):
                call(name)
        torch.cuda.synchronize()
        times = {name: [] for name in variants}
        for _ in range(args.rounds):
            for name in variants:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iters):
                    call(name)
                end.record()
                end.synchronize()
                times[name].append(start.elapsed_time(end) / args.iters)
        med = {name: statistics.median(t) for name, t in times.items()}
        results[m] = med
        print(f"M={m}: fp16 {med['fp16']:.3f} ms, int8 all {med['all']:.3f} ms "
              f"(x{med['fp16'] / med['all']:.2f}), int8 gu {med['gu']:.3f} ms "
              f"(x{med['fp16'] / med['gu']:.2f})", flush=True)
        if m in args.profile:
            for name in variants:
                kernels = _profile(lambda: call(name))
                print(f"  {name}: GPU kernels {sum(kernels.values()):.3f} ms", flush=True)
                for kname, ms in kernels.most_common(10):
                    print(f"    {ms:7.3f} ms  {kname}", flush=True)
    print("JSON " + json.dumps(results))


def _time_fn(fn, iters, rounds):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / iters)
    return statistics.median(out)


def sweep(args):
    """Tiles of the two bandwidth-bound int8 kernels: the gate|up epilogue (TI, TM, PG, warps; mode
    gu, the SwiGLU output written) at each M of --m, and the NF4 -> int8 decode of gate|up (ROWS,
    CB, warps). us per launch and the effective bandwidth."""
    import triton

    from solution import int8_fwd
    from solution.test_fused_proj import _linear4bit

    inter, d = 8960, 1536
    for m in args.m:
        acc = torch.randint(-2 ** 20, 2 ** 20, (2 * inter, m), dtype=torch.int32, device="cuda")
        s_w = torch.rand(2 * inter, device="cuda") * 1e-4
        s_x = torch.rand(m, device="cuda") * 1e-2
        gate_up = torch.empty((2 * inter, m), dtype=torch.float16, device="cuda")
        hidden = torch.empty((inter, m), dtype=torch.float16, device="cuda")
        moved = acc.numel() * 4 + gate_up.numel() * 2 + hidden.numel() * 2
        for ti, tm, pg, warps in [(32, 64, 448, 4), (32, 64, 896, 4), (16, 64, 448, 4),
                                  (64, 64, 448, 4), (32, 128, 448, 4), (32, 128, 448, 8),
                                  (16, 128, 448, 4), (64, 32, 448, 4), (32, 64, 448, 8),
                                  (32, 64, 224, 4), (16, 64, 224, 2), (64, 64, 448, 8)]:
            grid = (triton.cdiv(m, tm), inter // pg)

            def fn():
                int8_fwd.int8_gate_up_kernel[grid](acc, s_w, s_x, gate_up, hidden, inter, inter, m,
                                                   0, pg, 0, TI=ti, TM=tm, HIDDEN=True,
                                                   num_warps=warps)

            us = _time_fn(fn, args.iters, args.rounds)
            print(f"epilogue M={m} TI={ti} TM={tm} PG={pg} warps={warps}: {us:7.1f} us "
                  f"({moved / us / 1e3:.0f} GB/s)", flush=True)
    gate, up = _linear4bit(d, inter, seed=4), _linear4bit(d, inter, seed=5)
    ops = _ops(gate, up)
    out = torch.empty((2 * inter, d), dtype=torch.int8, device="cuda")
    scales = torch.empty(2 * inter, dtype=torch.float32, device="cuda")
    moved = out.numel() + out.numel() // 2 + out.numel() // 64
    for rows, cb, warps in [(8, 8, 4), (4, 8, 4), (16, 8, 4), (8, 8, 2), (8, 8, 8), (16, 8, 8),
                            (4, 8, 2), (8, 4, 4), (32, 8, 8), (2, 8, 1)]:
        programs = inter // rows

        def fn():
            int8_fwd.nf4_int8_gu_kernel[(2 * programs,)](
                *ops[0][:6], *ops[1][:6], out, scales, programs, d // 64, d, inter,
                ROWS=rows, CB=cb, QBLOCK=64, ABLOCK=256, num_warps=warps, enable_fp_fusion=False)

        us = _time_fn(fn, args.iters, args.rounds)
        print(f"decode ROWS={rows} CB={cb} warps={warps}: {us:7.1f} us "
              f"({moved / us / 1e3:.0f} GB/s)", flush=True)


# ---------------------------------------------------------------------------------------------


def _rows(path, tokenizer, count, max_len):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instr, extra = row["instruction"], row.get("input") or ""
            user = f"{instr}\n{extra}".strip() if extra else instr
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": user}],
                                                   tokenize=False, add_generation_prompt=True)
            full = prompt + row["output"] + "<|im_end|>\n"
            ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
            n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            rows.append((torch.tensor(ids), n_prompt))
            if len(rows) == count:
                break
    return rows


class _Probe:
    """Qwen2MLP.forward for the error run: fp16 reference and both int8 variants on the same input;
    `use` picks which output the model goes on with."""

    def __init__(self, fp16_layers=()):
        self.use = "fp16"
        self.record = False
        self.stats = collections.defaultdict(lambda: collections.defaultdict(list))
        self.fp16_layers = set(fp16_layers)
        self.only = None  # with a layer index: only that layer's MLP runs int8

    def forward(self, mlp, x):
        from solution import int8_fwd
        from solution.nf4 import dequantize_nf4

        layer = mlp._probe_layer
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        n = x2.shape[0]
        padded = -(-n // 16) * 16
        inter, d = mlp.gate_proj.out_features, shape[-1]
        if (self.use == "fp16" or layer in self.fp16_layers or n <= 16
                or (self.only is not None and layer != self.only)):
            return self._reference(mlp, x2)[2][:n].view(*shape[:-1], d)
        ops = _ops(mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        buffer = torch.empty(inter * d, dtype=torch.float16, device=x.device)
        x_q, s_x = int8_fwd.quantize_tokens(x2, n, padded)
        if self.use == "all":
            gate_up, pmax = int8_fwd.gate_up_forward(x_q, s_x, ops[0], ops[1], buffer,
                                                     torch.float16, inter)
            out = int8_fwd.down_forward(gate_up, pmax, ops[2], buffer, torch.float16, d)
        else:
            _, h = int8_fwd.gate_up_forward(x_q, s_x, ops[0], ops[1], buffer, torch.float16,
                                            inter, hidden=True)
            w_d = dequantize_nf4(mlp.down_proj.weight, mlp.down_proj.weight.quant_state,
                                 torch.float16)
            out = torch.mm(h.t(), w_d.t())
        return out[:n].view(*shape[:-1], d)

    def _reference(self, mlp, x2):
        from solution.nf4 import dequantize_nf4

        def w(lin):
            return dequantize_nf4(lin.weight, lin.weight.quant_state, torch.float16)

        x16 = x2.half()
        gate_up = torch.mm(torch.cat([w(mlp.gate_proj), w(mlp.up_proj)]), x16.t())
        inter = mlp.gate_proj.out_features
        g, u = gate_up[:inter].float(), gate_up[inter:].float()
        h = (g * torch.sigmoid(g) * u).half()
        out = torch.mm(h.t(), w(mlp.down_proj).t())
        if self.record:
            self._measure(mlp, x2, gate_up, h, out)
        return gate_up, h, out

    def _measure(self, mlp, x2, gate_up, h, out):
        from solution import int8_fwd
        from solution.nf4 import dequantize_nf4

        n, d = x2.shape
        if n <= 16:
            return
        padded = -(-n // 16) * 16
        inter = mlp.gate_proj.out_features
        ops = _ops(mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        buffer = torch.empty(inter * d, dtype=torch.float16, device=x2.device)
        x_q, s_x = int8_fwd.quantize_tokens(x2, n, padded)
        gu8, h8 = int8_fwd.gate_up_forward(x_q, s_x, ops[0], ops[1], buffer, torch.float16, inter,
                                           hidden=True)
        gu8, h8 = gu8[:, :n], h8[:, :n]
        w_d = dequantize_nf4(mlp.down_proj.weight, mlp.down_proj.weight.quant_state, torch.float16)
        out_gu = torch.mm(h8.t(), w_d.t())
        gu8b, pmax = int8_fwd.gate_up_forward(x_q, s_x, ops[0], ops[1], buffer, torch.float16,
                                              inter)
        out_all = int8_fwd.down_forward(gu8b, pmax, ops[2], buffer, torch.float16, d)[:n]
        # int8 down alone, on the fp16 SwiGLU output (its (gate|up) padded, one group of maxima)
        gu_ref = torch.zeros((2 * inter, padded), dtype=torch.float16, device=x2.device)
        gu_ref[:, :n] = gate_up
        g, u = gu_ref[:inter].float(), gu_ref[inter:].float()
        hp = g * torch.sigmoid(g) * u
        out_down = int8_fwd.down_forward(gu_ref, hp.abs().amax(0, keepdim=True), ops[2], buffer,
                                         torch.float16, d)[:n]
        st = self.stats[mlp._probe_layer]
        st["gate_up"].append(_rel(gu8, gate_up))
        st["hidden"].append(_rel(h8, h))
        st["out_gu"].append(_rel(out_gu, out))
        st["out_all"].append(_rel(out_all, out))
        st["out_down"].append(_rel(out_down, out))
        tok_err = ((out_all.float() - out.float()).norm(dim=1)
                   / out.float().norm(dim=1).clamp_min(1e-6))
        st["tok_worst"].append(tok_err.max().item())
        st["tok_worst_pos"].append(int(tok_err.argmax().item()))
        xa = x2.float().abs()
        st["x_peak_ratio"].append((xa.amax(1) / xa.pow(2).mean(1).sqrt().clamp_min(1e-12)).max().item())
        hf = h.float().abs()
        st["h_peak_ratio"].append((hf.amax(0) / hf.pow(2).mean(0).sqrt().clamp_min(1e-12)).max().item())


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def errors(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import int8_fwd

    int8_fwd._STATE["on"] = True
    int8_fwd.compile_now()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant,
                                                 dtype=torch.float32, device_map={"": 0})
    model.eval()
    probe = _Probe(args.fp16_layers)
    for i, layer in enumerate(model.model.layers):
        layer.mlp._probe_layer = i
    mq.Qwen2MLP.forward = lambda self, x: probe.forward(self, x)
    rows = _rows(args.data, tokenizer, args.rows, args.max_len)
    print(f"{len(rows)} rows, {sum(len(r[0]) for r in rows)} tokens", flush=True)
    losses = collections.defaultdict(list)
    uses = ["fp16", "all", "gu"] + (["all_skip", "gu_skip"] if args.fp16_layers else [])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for ids, n_prompt in rows:
            ids = ids.cuda()[None]
            for use in uses:
                probe.use = use.replace("_skip", "")
                probe.fp16_layers = set(args.fp16_layers) if use.endswith("_skip") else set()
                probe.record = use == "fp16"
                logits = model(input_ids=ids).logits.float()
                loss = F.cross_entropy(logits[0, n_prompt - 1:-1], ids[0, n_prompt:])
                losses[use].append(loss.item())
    print("\nlayer  gate_up  hidden  out_gu  out_all  down_only  worst_tok(pos)  x_peak/rms  h_peak/rms")
    summary = {}
    for layer in sorted(probe.stats):
        st = probe.stats[layer]
        mean = {k: statistics.mean(v) for k, v in st.items()}
        summary[layer] = mean
        print(f"{layer:5d}  {mean['gate_up']:.4f}  {mean['hidden']:.4f}  {mean['out_gu']:.4f}  "
              f"{mean['out_all']:.4f}  {mean['out_down']:.4f}     {max(st['tok_worst']):.3f}"
              f"({collections.Counter(st['tok_worst_pos']).most_common(1)[0][0]})"
              f"        {max(st['x_peak_ratio']):7.1f}  {max(st['h_peak_ratio']):8.1f}", flush=True)
    base = losses["fp16"]
    print(f"\nloss fp16 {statistics.mean(base):.5f}")
    for use in uses[1:]:
        rel = [abs(a - b) / b for a, b in zip(losses[use], base)]
        shift = (statistics.mean(losses[use]) - statistics.mean(base)) / statistics.mean(base)
        print(f"loss int8 {use}: {statistics.mean(losses[use]):.5f} (mean {shift * 100:+.3f} %, "
              f"mean |rel| per row {statistics.mean(rel) * 100:.3f} %, max {max(rel) * 100:.3f} %)",
              flush=True)
    single = {}
    if args.single:
        # one layer at a time in int8 "gu", the others fp16: each layer's own share of the shift
        probe.record, probe.fp16_layers = False, set()
        print(f"\nsingle layers in int8 {args.single}: loss shift, mean |rel| per row", flush=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for layer in range(len(model.model.layers)):
                probe.use, probe.only = args.single, layer
                got = []
                for ids, n_prompt in rows[:args.single_rows]:
                    ids = ids.cuda()[None]
                    logits = model(input_ids=ids).logits.float()
                    got.append(F.cross_entropy(logits[0, n_prompt - 1:-1], ids[0, n_prompt:]).item())
                ref = base[:len(got)]
                shift = (statistics.mean(got) - statistics.mean(ref)) / statistics.mean(ref)
                rel = statistics.mean(abs(a - b) / b for a, b in zip(got, ref))
                single[layer] = (shift, rel)
                print(f"  layer {layer:2d}: {shift * 100:+.3f} %  {rel * 100:.3f} %", flush=True)
        probe.only = None
    print("JSON " + json.dumps({"layers": summary, "losses": losses, "single": single}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[256, 512, 832, 1440, 2048])
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile", type=int, nargs="*", default=[832, 2048])
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="tiles of the epilogue and the decode")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--fp16-layers", type=int, nargs="*", default=[])
    parser.add_argument("--single", choices=["gu", "all"], default=None,
                        help="also the loss with one layer at a time in this int8 variant")
    parser.add_argument("--single-rows", type=int, default=12)
    args = parser.parse_args()
    if args.errors:
        errors(args)
    elif args.sweep:
        sweep(args)
    else:
        timing(args)


if __name__ == "__main__":
    main()
