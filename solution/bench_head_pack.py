"""The 12-bit packed head weight on the real checkpoint (head_pack.py): ratio, build time, copy and
decode time per chunk, and the streamed head's time and memory with the fp16 ring against the packed
ring at the run's token counts.

    python -m solution.bench_head_pack [--rows 4096] [--tokens 300/650/830/1100/1400/1970] [--reps 6]

(lists separated by "," or "/": tools/kaggle_run.py --bench takes no commas)

The weight is Qwen2.5-1.5B-Instruct's tied embedding, cast bf16 -> fp16 as the run loads it. The head
is called right after its prefetch (no layer compute to hide the first copies behind: the worst case
for the copies), alternating fp16 and packed, CUDA-synchronized wall time per head.
"""

import argparse
import re
import statistics
import time

import torch


def load_weight():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", allow_patterns=["model.safetensors"])
    with safe_open(f"{path}/model.safetensors", "pt") as handle:
        return handle.get_tensor("model.embed_tokens.weight").half()


def time_copies(streamer, pack, reps=3):
    dev = streamer.device
    stream = torch.cuda.Stream(dev)
    raw = torch.empty(streamer.rows, streamer.hidden, dtype=streamer.dtype, device=dev)
    slot = torch.empty(pack.slot_bytes, dtype=torch.uint8, device=dev)
    out = torch.empty_like(raw)
    res = {"raw_copy": [], "packed_copy": [], "decode": []}
    for _ in range(reps):
        for c in range(streamer.n_chunks - 1):  # the full chunks
            c0, vc, _, nb = pack.chunk(c)
            for key, fn in (("raw_copy", lambda: raw[:vc].copy_(streamer.host[c0:c0 + vc], non_blocking=True)),
                            ("packed_copy", lambda: slot[:nb].copy_(pack.host_chunk(c), non_blocking=True)),
                            ("decode", lambda: pack.decode(c, slot, out[:vc]))):
                with torch.cuda.stream(stream if key != "decode" else torch.cuda.current_stream(dev)):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    fn()
                    end.record()
                end.synchronize()
                res[key].append(begin.elapsed_time(end))
    return {k: statistics.median(v) for k, v in res.items()}


def sweep_decode(streamer, pack, configs=((512, 4), (1024, 4), (1024, 8), (2048, 4), (2048, 8), (4096, 8))):
    """Decode time of a full chunk per (pairs per program, warps), median of 20 after 3 warm-ups."""
    from .head_pack import decode_chunk

    dev = streamer.device
    slot = torch.empty(pack.slot_bytes, dtype=torch.uint8, device=dev)
    out = torch.empty(streamer.rows, streamer.hidden, dtype=streamer.dtype, device=dev)
    slot.copy_(pack.host_chunk(1))
    res = {}
    for block, warps in configs:
        times = []
        for i in range(23):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            decode_chunk(slot, out, pack.tables[1], pack.esc_pos[1], pack.esc_val[1], pack.n_esc[1],
                         block=block, warps=warps)
            end.record()
            end.synchronize()
            if i >= 3:
                times.append(begin.elapsed_time(end))
        c0, vc = 1 * streamer.rows, streamer.rows
        exact = torch.equal(out.view(torch.int16), streamer.host[c0:c0 + vc].to(dev).view(torch.int16))
        res[(block, warps)] = (statistics.median(times), exact)
    return res


def time_heads(streamer, pack, n, reps):
    dev, d = streamer.device, streamer.hidden
    gen = torch.Generator(device=dev).manual_seed(n)
    hidden = (torch.randn(n, d, device=dev, generator=gen)).half()
    targets = torch.randint(0, streamer.vocab, (n,), device=dev, generator=gen)
    targets[: int(0.3 * n)] = -100
    order = torch.argsort((targets != -100).to(torch.int8), descending=True, stable=True)
    h16, t = hidden[order].contiguous(), targets[order].contiguous()
    n_lab = int((t != -100).sum())
    out = {"fp16": [], "packed": []}
    peak = {}
    outputs = {}
    for r in range(reps + 1):
        for mode in (("fp16", "packed") if r % 2 == 0 else ("packed", "fp16")):
            streamer._pack = pack if mode == "packed" else None
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            started = time.perf_counter()
            streamer.prefetch()
            result = streamer.head(h16, t, n_lab, 1.0 / max(n_lab, 1))
            torch.cuda.synchronize(dev)
            if r:  # the first round warms up
                out[mode].append((time.perf_counter() - started) * 1000)
            peak[mode] = (torch.cuda.max_memory_allocated(dev) - base) / 2**20
            outputs[mode] = result
    streamer._pack = pack
    same = all(torch.equal(a, b) for a, b in zip(outputs["fp16"], outputs["packed"]))
    return {k: statistics.median(v) for k, v in out.items()}, peak, same


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", default="4096")
    parser.add_argument("--tokens", default="300,650,830,1100,1400,1970")
    parser.add_argument("--reps", type=int, default=6)
    args = parser.parse_args()
    from .head_stream import HeadStreamer

    print(torch.cuda.get_device_name(0), flush=True)
    started = time.perf_counter()
    weight = load_weight()
    print(f"weight {tuple(weight.shape)} {weight.dtype} loaded in {time.perf_counter() - started:.1f} s", flush=True)
    for rows in [int(r) for r in re.split(r"[,/]", args.rows)]:
        param = torch.nn.Parameter(weight.cuda(), requires_grad=False)
        streamer = HeadStreamer(param, rows=rows, slots=2)
        del param
        torch.cuda.empty_cache()
        started = time.perf_counter()
        pack = streamer.start_pack(log=print)
        pack.join()
        print(f"rows {rows}: pack ready {pack.ready} ({pack.error}) in {time.perf_counter() - started:.2f} s, "
              f"{pack.total_bytes / 2**20:.1f} MiB vs {streamer.host_bytes / 2**20:.1f} MiB "
              f"({pack.total_bytes / streamer.host_bytes * 100:.2f}%), escapes {pack.escapes} "
              f"(max per chunk {max(pack.n_esc)})", flush=True)
        if not pack.ready:
            continue
        per = time_copies(streamer, pack)
        print(f"rows {rows} per full chunk (median ms): fp16 copy {per['raw_copy']:.3f} "
              f"({rows * streamer.hidden * 2 / per['raw_copy'] / 1e6:.2f} GB/s), packed copy {per['packed_copy']:.3f} "
              f"({pack.slot_bytes / per['packed_copy'] / 1e6:.2f} GB/s), decode {per['decode']:.3f} "
              f"({rows * streamer.hidden * 3.5 / per['decode'] / 1e6:.0f} GB/s read+write); per head of "
              f"{streamer.n_chunks} chunks: fp16 copies {per['raw_copy'] * streamer.n_chunks:.1f} ms, packed copies "
              f"{per['packed_copy'] * streamer.n_chunks:.1f} ms + decode {per['decode'] * streamer.n_chunks:.1f} ms",
              flush=True)
        for (block, warps), (ms, exact) in sweep_decode(streamer, pack).items():
            print(f"rows {rows} decode BLOCK {block} pairs, {warps} warps: {ms * 1000:.1f} us per chunk "
                  f"({rows * streamer.hidden * 3.5 / ms / 1e6:.0f} GB/s), exact {exact}", flush=True)
        for n in [int(x) for x in re.split(r"[,/]", args.tokens)]:
            times, peak, same = time_heads(streamer, pack, n, args.reps)
            print(f"rows {rows} N {n:5d}: head fp16 {times['fp16']:7.2f} ms, packed {times['packed']:7.2f} ms "
                  f"({times['packed'] - times['fp16']:+.2f}); head memory above the start fp16 {peak['fp16']:.1f} "
                  f"MiB, packed {peak['packed']:.1f} MiB; bitwise equal {same}", flush=True)
        st = streamer.stats()
        print(f"rows {rows}: {st}", flush=True)
        streamer.restore()


if __name__ == "__main__":
    main()
