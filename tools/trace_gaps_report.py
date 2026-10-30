"""Where the GPU sat idle inside a training step, from a torch.profiler chrome trace.

    python tools/trace_gaps_report.py runs_diag/solution_1/diag_trace_steps8-9.json.gz [--min-gap-us 150] [--top 30]

solution/diag.py traces two steps with CPU and CUDA activities. This script takes every GPU event
(kernels, memcpy, memset on any stream), merges them into busy intervals, and reports the idle
gaps between them: their total, and the longest ones with the CPU operators that were running
when each gap began (what the CPU was doing while the GPU waited). It also sums the GPU time by
kernel name family, so the two steps can be compared with the harness's profile table.
"""

import argparse
import collections
import gzip
import json
import re
import sys


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["traceEvents"] if isinstance(data, dict) else data


def family(name):
    name = re.sub(r"<.*", "", name)
    name = re.sub(r"\(.*", "", name)
    return name[:70]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace")
    parser.add_argument("--min-gap-us", type=float, default=150.0)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    events = load(args.trace)

    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    if not gpu:
        sys.exit("no GPU events in the trace")
    gpu.sort(key=lambda e: e["ts"])
    cpu.sort(key=lambda e: e["ts"])
    start, end = gpu[0]["ts"], max(e["ts"] + e["dur"] for e in gpu)
    span = end - start

    # merge the busy intervals over all streams
    busy = []
    for e in gpu:
        s, t = e["ts"], e["ts"] + e["dur"]
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], t)
        else:
            busy.append([s, t])
    busy_total = sum(t - s for s, t in busy)
    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)]
    gaps = [(s, t) for s, t in gaps if t - s >= args.min_gap_us]
    print(f"GPU span {span / 1000:.1f} ms: busy {busy_total / 1000:.1f} ms, idle {(span - busy_total) / 1000:.1f} ms "
          f"({100 * (span - busy_total) / span:.1f}%); {len(gpu)} GPU events; "
          f"{len(gaps)} gaps >= {args.min_gap_us:.0f} us summing {sum(t - s for s, t in gaps) / 1000:.1f} ms")

    def cpu_at(ts):
        """CPU operators running at ts, innermost (shortest) first."""
        running = [e for e in cpu if e["ts"] <= ts <= e["ts"] + e["dur"]]
        running.sort(key=lambda e: e["dur"])
        return running[:4]

    print(f"\nlongest gaps (what the CPU ran when each began):")
    for s, t in sorted(gaps, key=lambda g: g[0] - g[1])[:args.top]:
        names = [f"{e['name'][:60]} ({e['dur'] / 1000:.2f} ms)" for e in cpu_at(s)]
        print(f"  {(t - s) / 1000:7.2f} ms at +{(s - start) / 1000:8.1f} ms: " + " < ".join(names))

    # idle attributed by the innermost CPU op at the gap start
    by_op = collections.Counter()
    for s, t in gaps:
        running = cpu_at(s)
        by_op[running[0]["name"][:60] if running else "(no CPU op)"] += t - s
    print(f"\nidle by the CPU operator at the gap start:")
    for name, us in by_op.most_common(20):
        print(f"  {us / 1000:7.2f} ms  {name}")

    kernels = collections.Counter()
    counts = collections.Counter()
    for e in gpu:
        kernels[family(e["name"])] += e["dur"]
        counts[family(e["name"])] += 1
    print(f"\nGPU time by kernel family (both steps):")
    for name, us in kernels.most_common(30):
        print(f"  {us / 1000:7.2f} ms {counts[name]:6d}x  {name}")


if __name__ == "__main__":
    sys.exit(main())
