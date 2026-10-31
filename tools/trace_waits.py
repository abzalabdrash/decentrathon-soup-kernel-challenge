"""Why the compute stream of a diag chrome trace (solution/diag.py, best with
SOUP_SOLUTION_DIAG_TRACE_CPU=0 and the memory snapshot outside the traced steps) is idle: per step and
phase (forward / head / backward), every gap between two kernels of the busiest stream is split into

- CPU: the next kernel's launch call had not returned yet when the gap began (the CPU had not
  submitted it): the part of the gap until it did;
- a dependency wait: the rest of the gap (the kernel was queued and still did not run). It is charged
  to the copy that ended last inside it (the one the stream was waiting for), by kind (from its size
  and direction: NF4 matrix, attention group, head chunk, int8 (gate|up), layer input, D2H) and by why
  that copy ended so late:
  - "issued late": its copy call returned less than its own transfer time before the wait began
    (nothing could have hidden it: the CPU or the program's structure issued it too late);
  - "engine queue": it started more than 50 us after its copy call, right behind another copy in the
    same direction (the engine was busy with copies issued before it);
  - "waited for its slot / an event": it started later than its call, but not right behind another copy
    (its stream waited for an event: the slot it overwrites was still being read, a recorded order);
  - "transfer": it started on time and the transfer itself outlasted the work in front of the wait;
- no copy ended inside the wait: another stream's kernel or an event.

    python tools/trace_waits.py <diag_trace_*.json.gz> [--skip-gap-ms 1000]

Steps are told apart by the optimizer's multi_tensor_apply kernels (one cluster per step); a gap longer
than --skip-gap-ms (a memory snapshot being written) separates windows and is not counted.
"""

import argparse
import collections
import gzip
import json


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)["traceEvents"]


def copy_kind(c):
    mib = c["args"].get("bytes", 0) / 2**20
    if "DtoH" in c["name"]:
        return f"D2H {mib:.1f} MiB" if mib < 1 else "D2H (gate|up form / layer input)"
    if "HtoD" not in c["name"]:
        return "other copy"
    if abs(mib - 12.0) < 0.05 or abs(mib - 1.125) < 0.01:
        return "H2D head chunk"
    if abs(mib - 6.77) < 0.05:
        return "H2D NF4 MLP matrix"
    if abs(mib - 2.71) < 0.05:
        return "H2D NF4 attention group"
    if mib < 0.5:
        return "H2D small"
    return "H2D activation (int8 gate|up or layer input)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--skip-gap-ms", type=float, default=1000.0)
    parser.add_argument("--min-us", type=float, default=3.0)
    args = parser.parse_args()
    events = load(args.trace)
    runtime = {}  # correlation -> (ts, end) of the CUDA runtime call
    for e in events:
        # runtime launches (torch, cuBLAS) and driver launches (Triton's cuLaunchKernel)
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {}):
            runtime[e["args"]["correlation"]] = (e["ts"], e["ts"] + e.get("dur", 0))
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset")]
    copies = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") == "gpu_memcpy"),
                    key=lambda e: e["ts"])
    busy = collections.Counter()
    for k in kernels:
        busy[k["args"].get("stream")] += k["dur"]
    compute = busy.most_common(1)[0][0]
    stream = sorted((k for k in kernels if k["args"].get("stream") == compute), key=lambda k: k["ts"])
    # step ends: the last optimizer kernel of each cluster that follows a head (forward + backward)
    ends, seen_head = [], False
    for k in stream:
        if k["name"].startswith("head_"):
            seen_head = True
        elif "multi_tensor_apply" in k["name"] and seen_head:
            if ends and k["ts"] - ends[-1] < 20_000:
                ends[-1] = k["ts"] + k["dur"]
            else:
                ends.append(k["ts"] + k["dur"])
            seen_head = False
    # previous copy in the same direction, for the engine-queue test
    by_dir = collections.defaultdict(list)
    for c in copies:
        by_dir["HtoD" in c["name"]].append(c)
    prev_same = {}
    for lst in by_dir.values():
        for a, b in zip(lst, lst[1:]):
            prev_same[id(b)] = a
    start = stream[0]["ts"]
    total = collections.Counter()
    nsteps = 0
    for end in ends:
        ks = [k for k in stream if start <= k["ts"] < end]
        lo, start = start, end
        if len(ks) < 50:
            continue
        heads = [k for k in ks if k["name"].startswith("head_")]
        h0, h1 = heads[0]["ts"], heads[-1]["ts"] + heads[-1]["dur"]
        span = (ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / 1e3
        work = sum(k["dur"] for k in ks) / 1e3
        cpu = collections.Counter()
        dep = collections.Counter()
        skipped = 0.0
        ends_sorted = [(c["ts"] + c["dur"], c) for c in copies if lo - 50_000 <= c["ts"] < end]
        for prev, nxt in zip(ks, ks[1:]):
            g0, g1 = prev["ts"] + prev["dur"], nxt["ts"]
            gap = g1 - g0
            if gap < args.min_us:
                continue
            if gap > args.skip_gap_ms * 1e3:
                skipped += gap
                continue
            phase = "forward" if g1 <= h0 else "head" if g1 <= h1 else "backward"
            submitted = runtime.get(nxt["args"].get("correlation"), (None, None))[1]
            cpu_part = 0.0 if submitted is None else min(gap, max(0.0, submitted - g0))
            cpu[phase] += cpu_part
            rest = gap - cpu_part
            if rest < args.min_us:
                continue
            w0 = g0 + cpu_part
            last = None
            for t, c in ends_sorted:
                if t > g1 + 2:
                    break
                if t >= w0 - 2:
                    last = c
            if last is None:
                why = ("launch latency (< 50 us, kernel just submitted)" if rest < 50
                       else "no copy ended in the wait (another stream's kernel / an event)")
                dep[(phase, why)] += rest
                continue
            kind = copy_kind(last)
            issued = runtime.get(last["args"].get("correlation"), (None, None))[1]
            before = prev_same.get(id(last))
            if issued is not None and last["ts"] - issued > 50 and before is not None and \
                    last["ts"] - (before["ts"] + before["dur"]) < 20:
                why = f"engine queue behind {copy_kind(before).split(' ', 1)[1]}"
            elif issued is not None and w0 - issued < last["dur"]:
                why = "issued late"
            elif issued is not None and last["ts"] - issued > 50:
                why = "waited for its slot / an event"
            else:
                why = "transfer"
            dep[(phase, f"{kind}: {why}")] += rest
        nsteps += 1
        idle = span - work
        print(f"step {nsteps} (trace {lo / 1e3:.0f}-{end / 1e3:.0f} ms): span {span:.1f} ms, kernels {work:.1f}, "
              f"idle {idle:.1f} (CPU {sum(cpu.values()) / 1e3:.1f}, dependency {sum(dep.values()) / 1e3:.1f}"
              f"{f', skipped {skipped / 1e3:.0f}' if skipped else ''})")
        for phase in ("forward", "head", "backward"):
            print(f"    {phase:8s} CPU {cpu[phase] / 1e3:6.2f} ms")
            total[(phase, "CPU had not submitted the next kernel")] += cpu[phase]
        for key, us in sorted(dep.items(), key=lambda kv: -kv[1]):
            if us >= 100:
                print(f"    {key[0]:8s} {us / 1e3:6.2f} ms  {key[1]}")
            total[key] += us
    if nsteps:
        print(f"\nmean over {nsteps} steps, ms per step:")
        for key, us in sorted(total.items(), key=lambda kv: -kv[1]):
            if us / nsteps >= 50:
                print(f"  {us / nsteps / 1e3:7.2f}  {key[0]:8s} {key[1]}")


if __name__ == "__main__":
    main()
