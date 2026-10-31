"""Exponent statistics and round trips for the streamed embedding (docs/notes/research-head-compress.md).

Usage:
  python research-head-compress-stats.py extract <model.safetensors> <out.bin>   (stdlib only; WSL python3 is fine)
  python research-head-compress-stats.py stats <out.bin>                         (numpy)

`extract` copies only the bytes of `model.embed_tokens.weight` (the header is read per the safetensors
spec: u64 little-endian header size, JSON header, then the data). `stats` casts the bf16 values to fp16
the way torch does (through fp32, round to nearest even), prints the exponent statistics and round-trips
two layouts chunk by chunk (4096 rows): 13-bit fixed and 3-bit codeword + fallback.
"""
import heapq
import json
import struct
import sys

V, D, ROWS, TILE = 151936, 1536, 4096, 512


def extract(src, dst, name="model.embed_tokens.weight", step=16 << 20):
    with open(src, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        start, end = json.loads(f.read(n))[name]["data_offsets"]
        f.seek(8 + n + start)
        left = end - start
        with open(dst, "wb") as out:
            while left:
                block = f.read(min(step, left))
                if not block:
                    raise SystemExit("short read")
                out.write(block)
                left -= len(block)


def fp16_chunks(raw):
    import numpy as np

    for lo in range(0, V, ROWS):
        hi = min(lo + ROWS, V)
        b = raw[lo * D:hi * D].astype(np.uint32)
        yield (b << 16).view(np.float32).astype(np.float16).view(np.uint16).astype(np.int32)


def entropy(counts):
    import numpy as np

    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_avg(counts):
    counts = [int(c) for c in counts if c > 0]
    heap = [(c, i, [i]) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    depth = [0] * len(counts)
    nxt = len(counts)
    while len(heap) > 1:
        c1, _, g1 = heapq.heappop(heap)
        c2, _, g2 = heapq.heappop(heap)
        for s in g1 + g2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, nxt, g1 + g2))
        nxt += 1
    return sum(c * d for c, d in zip(counts, depth)) / sum(counts)


def stats(path):
    import numpy as np

    raw = np.fromfile(path, dtype=np.uint16)
    assert raw.size == V * D
    hist = np.zeros(32, np.int64)
    for h in fp16_chunks(raw):
        hist += np.bincount((h >> 10) & 31, minlength=32)
    total = hist.sum()
    print(f"fp16 exponent field (5 bits): distinct {np.count_nonzero(hist)}, entropy {entropy(hist):.4f} bits, "
          f"Huffman {huffman_avg(hist):.4f} bits")
    best = max(range(26), key=lambda s: hist[s:s + 7].sum())
    print(f"best contiguous 7-window: exponents {best}..{best + 6}, fallback {100 * (1 - hist[best:best + 7].sum() / total):.3f}%")

    # layout A: 5-bit symbol (normal e in 1..13 -> e-1; e = 0 -> 13 + low three mantissa bits) + sign|mantissa byte
    # layout B: 3-bit codeword (window value k in 1..7, 0 = fallback with the full 16 bits), sign|mantissa byte
    #           for every element, u32 fallback base per 512-element tile
    bad_a = bad_b = fb_max = 0
    bits_b = 0
    for h in fp16_chunks(raw):
        e, m, s = (h >> 10) & 31, h & 1023, h >> 15
        assert e.max() <= 13 and np.all((m & 7 == 0) | (e == 0))
        ca = np.where(e > 0, e - 1, 13 + (m & 7))
        sm = (s << 7) | (m >> 3)
        planes = [np.packbits((ca.reshape(-1, 8) >> k) & 1, axis=1, bitorder="little").reshape(-1) for k in range(5)]
        cd = np.zeros(ca.size, np.int32)
        for k in range(5):
            cd |= np.unpackbits(planes[k], bitorder="little").astype(np.int32) << k
        ed = np.where(cd < 13, cd + 1, 0)
        bad_a += int(np.count_nonzero(((sm & 0x80) << 8 | (sm & 0x7F) << 3 | ed << 10 | np.where(cd >= 13, cd - 13, 0)) != h))

        k = np.where((e >= best) & (e < best + 7), e - best + 1, 0)
        high = k > 0
        fb = h[~high].astype(np.uint16)
        k8 = k.reshape(-1, TILE)
        tile_fb = (k8 == 0).sum(axis=1)
        base = np.concatenate(([0], np.cumsum(tile_fb)[:-1]))
        fb_max = max(fb_max, fb.size)
        bits_b += 3 * k.size + 8 * k.size + 16 * fb.size + 32 * base.size
        kd = np.zeros(k.size, np.int32)
        for p in range(3):
            kd |= np.unpackbits(np.packbits((k8.reshape(-1) >> p) & 1, bitorder="little"), bitorder="little").astype(np.int32) << p
        kd = kd.reshape(-1, TILE)
        isfb = kd == 0
        rank = np.cumsum(isfb, axis=1) - isfb
        idx = np.minimum(base[:, None] + rank, max(fb.size - 1, 0)).reshape(-1)
        normal = (sm.astype(np.int32) & 0x80) << 8 | (sm.astype(np.int32) & 0x7F) << 3 | (np.where(kd.reshape(-1) > 0, kd.reshape(-1) - 1 + best, 0) << 10)
        out = np.where(kd.reshape(-1) > 0, normal, fb[idx].astype(np.int32))
        bad_b += int(np.count_nonzero(out != h))
    n = V * D
    print(f"layout A (13 bits): mismatching elements {bad_a} of {n}; {13 / 16 * 100:.2f}% of fp16")
    print(f"layout B (3-bit + fallback): mismatching elements {bad_b} of {n}; {bits_b / n:.4f} bits = {100 * bits_b / n / 16:.2f}% "
          f"of fp16; largest chunk fallback {100 * fb_max / (ROWS * D):.2f}% (buffer sizing)")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "stats":
        stats(sys.argv[2])
    else:
        raise SystemExit(__doc__)
