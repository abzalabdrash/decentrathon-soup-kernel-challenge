"""Where the compute stream of a diag chrome trace (solution/diag.py) waits on something other than the
CPU: every idle gap between two kernels of the busiest stream whose second kernel the CPU had already
launched before the gap began (cudaLaunchKernel earlier than the end of the kernel before it) is a
dependency wait (an event of a copy, another stream). Sums them per next kernel and per step, next to
the copies of the other streams.

    python tools/trace_waits.py <diag_trace_steps*.json.gz> [--top 12]
"""

import argparse
import collections
import gzip
import json


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--min-us", type=float, default=5.0, help="gaps shorter than this are ignored")
    args = parser.parse_args()
    events = load(args.trace)
    launches = {}  # correlation -> CPU timestamp of the launch call
    for e in events:
        if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = e["ts"]
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset")]
    copies = [e for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy"]
    busy = collections.Counter()
    for k in kernels:
        busy[k["args"].get("stream")] += k["dur"]
    compute = busy.most_common(1)[0][0]
    stream = sorted((k for k in kernels if k["args"].get("stream") == compute), key=lambda k: k["ts"])
    steps = sorted((e for e in events if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")),
                   key=lambda e: e["ts"])
    if not steps:  # no step annotations: one window over the whole trace
        steps = [{"name": "trace", "ts": stream[0]["ts"], "dur": stream[-1]["ts"] + stream[-1]["dur"] - stream[0]["ts"]}]
    for step in steps:
        lo, hi = step["ts"], step["ts"] + step["dur"]
        ks = [k for k in stream if lo <= k["ts"] < hi]
        if len(ks) < 2:
            continue
        waits, cpu_gaps = collections.Counter(), 0.0
        count = collections.Counter()
        for prev, nxt in zip(ks, ks[1:]):
            gap = nxt["ts"] - (prev["ts"] + prev["dur"])
            if gap < args.min_us:
                continue
            launched = launches.get(nxt["args"].get("correlation"))
            if launched is not None and launched < prev["ts"] + prev["dur"]:
                name = nxt["name"][:60]
                waits[name] += gap
                count[name] += 1
            else:
                cpu_gaps += gap
        span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
        work = sum(k["dur"] for k in ks)
        h2d = [c for c in copies if lo <= c["ts"] < hi and "HtoD" in c["name"]]
        d2h = [c for c in copies if lo <= c["ts"] < hi and "DtoH" in c["name"]]
        print(f"{step['name']}: compute stream {compute}: span {span / 1e3:.1f} ms, kernels {work / 1e3:.1f} ms, "
              f"dependency waits {sum(waits.values()) / 1e3:.2f} ms in {sum(count.values())} gaps, "
              f"CPU-bound gaps {cpu_gaps / 1e3:.2f} ms; copies H2D {sum(c['dur'] for c in h2d) / 1e3:.1f} ms "
              f"({sum(c['args'].get('bytes', 0) for c in h2d) / 2**20:.0f} MiB), D2H "
              f"{sum(c['dur'] for c in d2h) / 1e3:.1f} ms ({sum(c['args'].get('bytes', 0) for c in d2h) / 2**20:.0f} MiB)")
        for name, us in waits.most_common(args.top):
            print(f"    {us / 1e3:7.2f} ms in {count[name]:3d} waits before {name}")


if __name__ == "__main__":
    main()
