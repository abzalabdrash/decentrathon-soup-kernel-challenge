"""Soup's setup of the task model (Qwen2.5-1.5B-Instruct, 4-bit NF4 with double quantization, fp16)
with and without the load-time moves of load_stream.py: the load's time, the GPU memory after the
load and after peft's kbit preparation, the setup's peak (allocated and reserved), and one training
step on a small batch whose loss and LoRA gradients must be the same bits either way. Each mode runs
in a process of its own (clean peaks). Run on a GPU with the model in the HF cache:
`python -m solution.bench_load_stream [--modes on off]`.
"""

import argparse
import json
import os
import subprocess
import sys
import time

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MIB = 2 ** 20


def _mem():
    import torch

    return {"alloc": torch.cuda.memory_allocated() / MIB, "reserved": torch.cuda.memory_reserved() / MIB}


def run_one(mode):
    os.environ["SOUP_SOLUTION_LOAD_STREAM"] = "1" if mode == "on" else "0"
    import peft
    import torch
    from peft import LoraConfig, get_peft_model
    from soup_cli.trainer import sft as soup_sft
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    from solution import fused_ce
    from solution.unpad import patch_qwen2_attention
    from solution.weight_stream import patch_weight_stream

    torch.cuda.init()
    marks = {}

    def setup(self):
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant, dtype=torch.float16,
                                                     device_map={"": 0})
        torch.cuda.synchronize()
        marks["load_s"] = time.perf_counter() - started
        marks["after_load"] = _mem()
        marks["load_peak"] = {"alloc": torch.cuda.max_memory_allocated() / MIB,
                              "reserved": torch.cuda.max_memory_reserved() / MIB}
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        marks["after_prepare"] = _mem()
        torch.manual_seed(1)
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                 target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        gen = torch.Generator().manual_seed(2)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data.copy_(torch.randn(param.shape, generator=gen) * 0.02)
        self.model = model

    soup_sft.SFTTrainerWrapper._setup_transformers = setup
    fused_ce.patch_qwen2_causal_lm()
    patch_qwen2_attention()
    patch_weight_stream(slots=2)
    wrapper = soup_sft.SFTTrainerWrapper.__new__(soup_sft.SFTTrainerWrapper)
    started = time.perf_counter()
    wrapper._setup_transformers()
    torch.cuda.synchronize()
    marks["setup_s"] = time.perf_counter() - started
    marks["after_setup"] = _mem()
    marks["setup_peak"] = {"alloc": torch.cuda.max_memory_allocated() / MIB,
                           "reserved": torch.cuda.max_memory_reserved() / MIB}

    model = wrapper.model
    gen = torch.Generator().manual_seed(3)
    lengths = (120, 77, 43)
    ids = torch.randint(0, 151000, (3, max(lengths)), generator=gen).cuda()
    mask = (torch.arange(ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]).long().cuda()
    labels = ids.masked_fill(mask == 0, -100)
    labels[:, :10] = -100
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
    (out.loss * 1024.0).backward()
    grads = [p.grad.double().sum().item() for _, p in model.named_parameters() if p.requires_grad]
    marks["loss"] = out.loss.item()
    marks["grad_sums"] = grads[:4] + [sum(grads)]
    return marks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["on", "off"])
    parser.add_argument("--one", choices=["on", "off"])
    args = parser.parse_args()
    if args.one:
        print("RESULT " + json.dumps(run_one(args.one)), flush=True)
        return
    results = {}
    for mode in args.modes:
        out = subprocess.run([sys.executable, "-m", "solution.bench_load_stream", "--one", mode],
                             capture_output=True, text=True, check=False)
        line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
        if not line:
            print(out.stdout[-3000:], out.stderr[-3000:])
            raise SystemExit(f"mode {mode} failed")
        for x in out.stdout.splitlines():
            if x.startswith("[solution]"):
                print(f"  {mode}: {x}")
        results[mode] = json.loads(line[0][len("RESULT "):])
    print("| mode | load, s | whole setup, s | after load alloc/reserved, MiB | load peak alloc/reserved "
          "| after prepare alloc | setup peak alloc/reserved | loss |")
    print("|---|---|---|---|---|---|---|---|")
    for mode, r in results.items():
        print(f"| {mode} | {r['load_s']:.2f} | {r['setup_s']:.2f} "
              f"| {r['after_load']['alloc']:.0f} / {r['after_load']['reserved']:.0f} "
              f"| {r['load_peak']['alloc']:.0f} / {r['load_peak']['reserved']:.0f} "
              f"| {r['after_prepare']['alloc']:.0f} | {r['setup_peak']['alloc']:.0f} / "
              f"{r['setup_peak']['reserved']:.0f} | {r['loss']:.6f} |")
    if len(results) == 2:
        a, b = results.values()
        print("same step:", a["loss"] == b["loss"] and a["grad_sums"] == b["grad_sums"], a["grad_sums"], b["grad_sums"])


if __name__ == "__main__":
    main()
