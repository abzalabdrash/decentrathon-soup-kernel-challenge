"""The int8 head against the fp16 streamed head on real final hidden states of Qwen2.5-1.5B-Instruct
(fp16 model, rows of the task's data in batches of 4 as the run's steps): per step the logged loss,
TRL's entropy (all real positions) and accuracy (labelled positions), Soup's rewind row losses, the
argmax flips, and dL/dh against the fp16 head's for the int8 head's split (P W_q - W_q[y], P's
positive part quantized) and for a plain P - Y quantization (the -1 of the target in the same int8
row, emulated with the same int8 logits).

    python -m solution.bench_head_int8_real [--data data/train.jsonl] [--steps 4] [--rows-from 0]
"""

import argparse
import json
import statistics

import torch


def batches(path, steps, first, tok, device, batch=4):
    rows = [json.loads(line) for line in open(path, encoding="utf-8")][first:first + steps * batch]
    for i in range(0, len(rows), batch):
        texts, prompts = [], []
        for r in rows[i:i + batch]:
            user = r["instruction"] + (("\n" + r["input"]) if r.get("input") else "")
            prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                             add_generation_prompt=True)
            texts.append(prompt + r["output"] + tok.eos_token)
            prompts.append(prompt)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        labels = enc.input_ids.clone()
        for b, p in enumerate(prompts):
            labels[b, :len(tok(p).input_ids)] = -100
        labels[enc.attention_mask == 0] = -100
        yield enc, labels


def plain_dh(h16, t, n_lab, int8, streamer, col_inv):
    """dL/dh with P - Y quantized per row and chunk as one int8 row (s_v folded in), from the same
    int8 logits as the head, normalized by the int8 lse: the variant the split avoids."""
    from .head_int8 import quant_rows

    dev, d = h16.device, h16.shape[1]
    hq = torch.empty(h16.shape, dtype=torch.int8, device=dev)
    sh = quant_rows(h16, hq, col_inv).double()
    chunks = []
    for c in range(int8.n_chunks):
        _, vc, _, _ = int8.chunk(c)
        raw = int8.host_chunk(c).to(dev)
        chunks.append((raw[:vc * d].view(torch.int8).view(vc, d), raw[vc * d:].view(torch.float32)))
    lab = slice(0, n_lab)
    logits = [(hq[lab].double() @ q.double().t()) * sh[lab, None] * sv.double()[None, :] for q, sv in chunks]
    lse = torch.logsumexp(torch.cat(logits, 1), 1)
    out = torch.zeros(n_lab, d, dtype=torch.float64, device=dev)
    c0 = 0
    for (q, sv), x in zip(chunks, logits):
        vc = q.shape[0]
        g = torch.exp(x - lse[:, None])
        local = t[:n_lab] - c0
        hit = (local >= 0) & (local < vc)
        g[hit.nonzero().flatten(), local[hit]] -= 1.0
        u = g * sv.double()[None, :]
        amax = u.abs().amax(1)
        inv = torch.where(amax > 0, 127.0 / amax, torch.zeros_like(amax))
        uq = torch.floor(u * inv[:, None] + 0.5).clamp(-127, 127)
        out += (uq @ q.double()) * (amax / 127.0)[:, None]
        c0 += vc
    return out * col_inv.double()[None, :] / n_lab


def step_gaps(streamer, int8, h16, t, seq, n_seq):
    """(loss, entropy, accuracy, rewind rows) relative gaps of the int8 head against the fp16 head on
    one step's sorted rows, as the run logs them."""
    n_lab = int((t != -100).sum())
    lab = t != -100
    out = {}
    for mode in ("fp16", "int8"):
        int8.calls = 10**6 if mode == "int8" else -10**6
        streamer.prefetch(h16.shape[0])
        out[mode] = streamer.head(h16, t, n_lab, 1.0 / n_lab)
    (l16, e16, c16, _), (l8, e8, c8, _) = out["fp16"], out["int8"]
    rows16 = [l16[lab & (seq == b)].mean().item() for b in range(n_seq)]
    rows8 = [l8[lab & (seq == b)].mean().item() for b in range(n_seq)]
    return (abs((l8[lab].sum() / l16[lab].sum()).item() - 1), abs(e8.mean().item() / e16.mean().item() - 1),
            abs(c8[lab].sum().item() / max(c16[lab].sum().item(), 1) - 1),
            statistics.mean(abs(a / b - 1) for a, b in zip(rows8, rows16)))


def hidden_steps(model, tok, path, steps, first=0):
    """(sorted hidden rows fp16, sorted targets, their sequence index, sequences) per step of 4 rows."""
    dev = next(model.parameters()).device
    for enc, labels in batches(path, steps, first, tok, dev):
        with torch.no_grad():
            h = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        tgt = torch.full_like(labels, -100)
        tgt[:, :-1] = labels[:, 1:]
        keep = enc.attention_mask.bool()
        seq = torch.arange(h.shape[0], device=dev)[:, None].expand_as(keep)[keep]
        hh, tt = h[keep].half(), tgt[keep]
        order = torch.argsort((tt != -100).to(torch.int8), descending=True, stable=True)
        yield hh[order].contiguous(), tt[order].contiguous(), seq[order], h.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--rows-from", type=int, default=0)
    parser.add_argument("--deltas", default="0/0.5/1/2", help="rescoring thresholds to compare (0: off)")
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import re

    from . import head_int8
    from .head_stream import HeadStreamer

    deltas = [float(x) for x in re.split(r"[,/]", args.deltas)]

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16).to(dev).eval()
    weight = model.get_input_embeddings().weight.detach()
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    streamer = HeadStreamer(param, rows=4096, slots=2, norm=model.model.norm)
    int8 = streamer.start_int8(defer=False)
    int8.join()
    assert int8.ready, int8.error
    gaps = {k: [] for k in ("loss", "entropy", "accuracy", "rewind")}
    acc_gaps = {dl: [] for dl in deltas}
    for step, (enc, labels) in enumerate(batches(args.data, args.steps, args.rows_from, tok, dev)):
        with torch.no_grad():
            h = model.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
        tgt = torch.full_like(labels, -100)
        tgt[:, :-1] = labels[:, 1:]
        keep = enc.attention_mask.bool()
        seq = torch.arange(h.shape[0], device=dev)[:, None].expand_as(keep)[keep]
        hh, tt = h[keep].half(), tgt[keep]
        order = torch.argsort((tt != -100).to(torch.int8), descending=True, stable=True)
        h16, t = hh[order].contiguous(), tt[order].contiguous()
        seq = seq[order]
        n_lab = int((t != -100).sum())
        res = {}
        int8.calls = -10**6
        streamer.prefetch(h16.shape[0])
        res["fp16"] = streamer.head(h16, t, n_lab, 1.0 / n_lab)
        int8.calls = 10**6
        flips_by = {}
        head_int8.LOSS_FIX = False
        streamer.prefetch(h16.shape[0])
        nofix = streamer.head(h16, t, n_lab, 1.0 / n_lab)
        head_int8.LOSS_FIX = True
        lab0 = t != -100
        nofix_loss = (nofix[0][lab0].sum() / res["fp16"][0][lab0].sum()).item() - 1
        rows16_ = [res["fp16"][0][lab0 & (seq == b)].mean().item() for b in range(h.shape[0])]
        rowsnf = [nofix[0][lab0 & (seq == b)].mean().item() for b in range(h.shape[0])]
        print(f"step {step}: without the loss correction: loss {nofix_loss * 100:+.3f}%, rewind rows mean |rel| "
              f"{statistics.mean(abs(a / b - 1) for a, b in zip(rowsnf, rows16_)) * 100:.3f}%")
        for dl in deltas:
            head_int8.DELTA = dl
            streamer.prefetch(h16.shape[0])
            out = streamer.head(h16, t, n_lab, 1.0 / n_lab)
            c = out[2]
            flips_by[dl] = int((c[t != -100] != res["fp16"][2][t != -100]).sum())
            acc_gaps[dl].append(abs((c.sum() / n_lab).item() / max((res["fp16"][2].sum() / n_lab).item(), 1e-9) - 1))
            res["int8"] = out
        (l16, e16, c16, d16), (l8, e8, c8, d8) = res["fp16"], res["int8"]
        lab = t != -100
        stat = {
            "loss": (l8[lab].sum() / n_lab).item() / (l16[lab].sum() / n_lab).item() - 1,
            "entropy": e8.mean().item() / e16.mean().item() - 1,
            "accuracy": (c8.sum() / n_lab).item() / max((c16.sum() / n_lab).item(), 1e-9) - 1,
        }
        rows16 = [l16[lab & (seq == b)].mean().item() for b in range(h.shape[0])]
        rows8 = [l8[lab & (seq == b)].mean().item() for b in range(h.shape[0])]
        stat["rewind"] = statistics.mean(abs(a / b - 1) for a, b in zip(rows8, rows16))
        flips = int((c8[lab] != c16[lab]).sum())
        for k in gaps:
            gaps[k].append(abs(stat[k]))
        ref = d16[:n_lab].double()
        err_split = ((d8[:n_lab].double() - ref).norm() / ref.norm()).item()
        err_plain = ((plain_dh(h16, t, n_lab, int8, streamer, int8.inv_col) - ref).norm() / ref.norm()).item()
        print(f"step {step}: argmax flips by rescoring threshold " + ", ".join(f"{k}: {v}" for k, v in flips_by.items()))
        print(f"step {step}: N {h16.shape[0]} labelled {n_lab}: int8 vs fp16 loss {stat['loss'] * 100:+.3f}%, "
              f"entropy {stat['entropy'] * 100:+.3f}%, accuracy {stat['accuracy'] * 100:+.3f}% ({flips} argmax "
              f"flips of {n_lab}), rewind rows mean |rel| {stat['rewind'] * 100:.3f}%; dL/dh rel. error vs fp16: "
              f"split (ours) {err_split:.4f}, plain P - Y {err_plain:.4f}", flush=True)
    print("mean |gap| over steps: " + ", ".join(f"{k} {statistics.mean(v) * 100:.3f}%" for k, v in gaps.items())
          + f" (accuracy with the last threshold, {deltas[-1]})", flush=True)
    print("accuracy mean |gap| by rescoring threshold: " + ", ".join(
        f"{k}: {statistics.mean(v) * 100:.3f}%" for k, v in acc_gaps.items()), flush=True)
    streamer.restore()


if __name__ == "__main__":
    main()
