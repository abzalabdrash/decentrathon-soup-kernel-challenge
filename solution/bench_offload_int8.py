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


def _example_swiglu(x, hidden):
    import triton

    from solution.swiglu import example_swiglu_fwd_kernel

    inter, m = hidden.shape
    example_swiglu_fwd_kernel[(triton.cdiv(inter * m, 1024),)](x[:inter], x[inter:], hidden, inter * m,
                                                               BLOCK=1024)


def tiles(ms, configs):
    """quant and rebuild per layer for (tokens per program) x (warps) configurations."""
    from solution import offload_int8

    rows = 2 * 8960
    for config in configs:
        offload_int8.TILE_M, offload_int8.WARPS = (int(v) for v in config.split("x"))
        line = []
        for m in ms:
            x = (torch.randn(rows, m, device="cuda") * 2).half()
            hidden = torch.empty((rows // 2, m), dtype=x.dtype, device="cuda")
            wire, codec = offload_int8.encode(x)
            swiglu = _time(lambda: _example_swiglu(x, hidden), inner=10)
            quant = _time(lambda: offload_int8.swiglu_quant(x, hidden), inner=10)
            rebuild = _time(lambda: codec.decode(wire), inner=10)
            line.append(f"M={m}: SwiGLU {swiglu:.3f} SwiGLU+int8 {quant:.3f} rebuild {rebuild:.3f}")
        print(f"tile {config:8s} " + "  ".join(line), flush=True)


def kernels_and_copies(ms, layers=28):
    from solution import offload_int8

    offload_int8.compile_now()
    rows = 2 * 8960
    print(f"per layer ((gate|up) of {rows} rows, group {offload_int8.GROUP}); x{layers} = per step")
    print(f"{'M':>6s} {'+quant':>7s} {'rebuild':>8s} {'D2H f16':>8s} {'D2H i8':>7s} {'H2D f16':>8s} "
          f"{'H2D i8':>7s} {'PCIe saved':>11s} {'GPU added':>10s}  (ms)")
    for m in ms:
        x = (torch.randn(rows, m, device="cuda") * 2).half()
        wire, codec = offload_int8.encode(x)
        host16 = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
        host8 = torch.empty(wire.shape, dtype=wire.dtype, pin_memory=True)
        hidden = torch.empty((rows // 2, m), dtype=x.dtype, device="cuda")
        # what the int8 path adds in the forward: SwiGLU + int8 form against the example SwiGLU alone
        quant = (_time(lambda: offload_int8.swiglu_quant(x, hidden), inner=10)
                 - _time(lambda: _example_swiglu(x, hidden), inner=10))
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


def _chat_ids(tokenizer, row, limit=512):
    """Soup's alpaca row as Qwen's chat: a user turn of instruction (+ input), an assistant turn of
    output, cut at the task's 512 tokens."""
    user = row["instruction"] + (f"\n\n{row['input']}" if row["input"] else "")
    ids = tokenizer.apply_chat_template([{"role": "user", "content": user},
                                         {"role": "assistant", "content": row["output"]}],
                                        tokenize=True)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    return list(ids)[:limit]


def real_gradient_error(layer_ids, data):
    """The LoRA gradients of real Qwen2.5-1.5B layers on real activations: the model in NF4 as Soup
    loads it (double quant, fp16 compute), four rows of the task's training data through the plain
    transformers model for every layer's input, then that layer (its own NF4 weights, LoRA r=16 on q
    and v with random A and B) through solution's fused path with its norms: fp16 / int8 MLP backward x
    half / int8 offload of (gate|up). Errors against the fp16 backward with the half offload, and of
    P1 (int8 offload) against main (int8 backward, half offload)."""
    import json

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from solution import int8_bwd, offload, offload_int8
    from solution.test_fused_proj import _lora, _lora_params, _positions, _rel, _spans
    from solution.unpad import varlen

    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0})
    model.eval()
    with open(data, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle][:64]
    seqs = [_chat_ids(tokenizer, row) for row in rows]
    batches = {"rows 0-3": list(range(4)),
               "the 4 longest of rows 0-63": sorted(range(64), key=lambda i: -len(seqs[i]))[:4]}
    inputs = {}  # (batch, layer) -> real input of the layer, (1, N, 1536) fp32
    with torch.no_grad():
        for name, picked in batches.items():
            per_row = [model(input_ids=torch.tensor([seqs[i]], device="cuda"), output_hidden_states=True)
                       .hidden_states for i in picked]
            for layer in layer_ids:
                inputs[name, layer] = torch.cat([hs[layer] for hs in per_row], dim=1).float()
    _patch()
    int8_bwd.compile_now()
    offload_int8.compile_now()
    eps = model.config.rms_norm_eps

    def rms(x, weight):
        return weight.float() * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    print("\nreal Qwen2.5-1.5B layers on real activations: relative error of dL/dx and of the LoRA "
          "gradients (A_q B_q A_v B_v)")
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        attn, mlp = layer.self_attn, layer.mlp
        attn.q_proj = _lora(attn.q_proj, seed=100 + layer_id)
        attn.v_proj = _lora(attn.v_proj, seed=200 + layer_id)
        attn.train()
        mlp.train()
        weights = [*attn.parameters(), *mlp.parameters()]
        for name, picked in batches.items():
            lengths = [len(seqs[i]) for i in picked]
            x0 = inputs[name, layer_id]
            pe = model.model.rotary_emb(x0, _positions(lengths))
            grad = torch.randn(x0.shape, device="cuda",
                               generator=torch.Generator("cuda").manual_seed(layer_id))

            def run(bwd_int8, gu_int8):
                int8_bwd._STATE["on"] = bwd_int8
                offload_int8._STATE["on"] = gu_int8
                for p in weights:
                    p.grad = None
                x = x0.detach().requires_grad_(True)
                with offload.offload_saved_tensors(weights=weights) as off, \
                        torch.autocast("cuda", dtype=torch.float16), varlen(_spans(lengths)):
                    a, _ = attn(rms(x, layer.input_layernorm.weight), pe, None)
                    h = x + a
                    out = h + mlp(rms(h, layer.post_attention_layernorm.weight))
                    out.backward(grad)
                torch.cuda.synchronize()
                assert off.stats["int8"] == (1 if gu_int8 else 0), off.summary()
                return [x.grad] + [p.grad.clone() for p in _lora_params(attn)]

            ref = run(False, False)
            main_path = run(True, False)
            results = {"fp16 backward, int8 offload": run(False, True),
                       "int8 backward, half offload (main)": main_path,
                       "int8 backward, int8 offload (P1)": run(True, True)}
            for label, got in results.items():
                errs = [_rel(a, b) for a, b in zip(got, ref)]
                print(f"  layer {layer_id:2d}, {name} ({sum(lengths)} tokens), {label:34s} dx {errs[0]:.4f}"
                      f"  LoRA " + " ".join(f"{e:.4f}" for e in errs[1:]), flush=True)
            errs = [_rel(a, b) for a, b in zip(results["int8 backward, int8 offload (P1)"], main_path)]
            print(f"  layer {layer_id:2d}, {name}, P1 against main{'':19s} dx {errs[0]:.4f}  LoRA "
                  + " ".join(f"{e:.4f}" for e in errs[1:]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=[416, 832, 1440, 2048])
    parser.add_argument("--groups", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--skip-error", action="store_true")
    parser.add_argument("--skip-kernels", action="store_true")
    parser.add_argument("--tiles", nargs="*", default=[], metavar="TMxWARPS")
    parser.add_argument("--real", type=int, nargs="*", default=None, metavar="LAYER",
                        help="the real model's layers on real activations (needs data/train.jsonl)")
    parser.add_argument("--data", default="data/train.jsonl")
    args = parser.parse_args()
    if args.real is not None:
        real_gradient_error(args.real or [2, 13, 27], args.data)
        return
    if args.tiles:
        tiles(args.m, args.tiles)
    if not args.skip_kernels:
        kernels_and_copies(args.m)
    if not args.skip_error:
        gradient_error([[212, 187, 310, 117], [512, 512, 512, 512]], args.groups)


if __name__ == "__main__":
    main()
