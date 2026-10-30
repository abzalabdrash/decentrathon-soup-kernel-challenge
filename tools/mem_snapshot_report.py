"""What holds the GPU memory at the peak of a step, from a torch allocator snapshot.

    python tools/mem_snapshot_report.py runs_diag/solution_1/diag_memory_step6.pickle [--top 40] [--frames 6]

solution/diag.py records the allocator's history during one training step
(`torch.cuda.memory._record_memory_history`) and dumps `_dump_snapshot` at the step's end. The
snapshot has two parts: `segments` (the blocks alive at dump time) and `device_traces` (every
alloc / free since recording started, each with the Python stack that made it). Blocks that were
allocated before the recording began (weights, the embedding, optimizer state, pinned ring slots)
have no alloc event: they show in `segments` without frames.

This script replays the trace to find the moment of the highest total allocated, then lists what
was alive at that moment: the pre-recording blocks as one line (the static memory), and the blocks
allocated during the step grouped by the stack frame that made them, biggest groups first.
"""

import argparse
import collections
import pickle
import sys

MIB = 2 ** 20
NOISE = ("torch/_tensor.py", "torch/autograd/", "torch/_ops.py", "torch/nn/functional.py",
         "torch/nn/modules/module.py", "torch/utils/_device.py", "torch/_dynamo", "torch/overrides.py")


def frame_text(frame):
    return f"{frame.get('filename', '?')}:{frame.get('line', 0)} {frame.get('name', '?')}"


def interesting_frames(frames, count):
    """The first `count` Python frames that are not torch plumbing, innermost first (the snapshot
    also holds C++ frames, `??:0 torch::...`, which say nothing about the caller)."""
    python = [f for f in frames if str(f.get("filename", "")).endswith(".py")]
    chosen = [f for f in python if not any(n in f.get("filename", "") for n in NOISE)]
    return [frame_text(f) for f in (chosen or python or frames)[:count]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("snapshot")
    parser.add_argument("--top", type=int, default=40, help="groups to print")
    parser.add_argument("--frames", type=int, default=5, help="stack frames per group")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    with open(args.snapshot, "rb") as handle:
        snap = pickle.load(handle)

    traces = snap.get("device_traces", [])
    trace = traces[args.device] if args.device < len(traces) else []
    segments = [s for s in snap.get("segments", []) if s.get("device", 0) == args.device]

    # blocks alive at dump time, by address
    final_live = {}
    for seg in segments:
        addr = seg["address"]
        for block in seg.get("blocks", []):
            if block.get("state") == "active_allocated":
                final_live[addr] = (block["size"], block.get("frames") or [])
            addr += block["size"]

    allocated_in_trace = {e["addr"] for e in trace if e.get("action") == "alloc"}
    # blocks alive when the recording began: alive at the end but never allocated in the trace,
    # plus blocks freed in the trace that the trace never saw allocated
    pre = {addr: size for addr, (size, _) in final_live.items() if addr not in allocated_in_trace}
    seen_alloc = set()
    for e in trace:
        if e.get("action") == "alloc":
            seen_alloc.add(e["addr"])
        elif e.get("action") == "free_completed" and e["addr"] not in seen_alloc and e["addr"] not in pre:
            pre[e["addr"]] = e["size"]
    static_bytes = sum(pre.values())

    live = dict(pre)  # addr -> size
    frames_of = {}
    total = static_bytes
    peak, peak_live, peak_index = total, dict(live), -1
    for i, e in enumerate(trace):
        action = e.get("action")
        if action == "alloc":
            live[e["addr"]] = e["size"]
            frames_of[e["addr"]] = e.get("frames") or []
            total += e["size"]
            if total > peak:
                peak, peak_live, peak_index = total, dict(live), i
        elif action == "free_completed":
            size = live.pop(e["addr"], None)
            if size is not None:
                total -= size
    print(f"trace events: {len(trace)}; blocks alive at dump: {len(final_live)}, "
          f"{sum(s for s, _ in final_live.values()) / MIB:.0f} MiB")
    print(f"allocated before the recording (static): {len(pre)} blocks, {static_bytes / MIB:.0f} MiB")
    print(f"peak during the recorded step: {peak / MIB:.0f} MiB at event {peak_index}, "
          f"{len(peak_live)} blocks alive\n")

    # the static blocks by size class
    sizes = collections.Counter()
    for size in pre.values():
        sizes[size] += 1
    print("static blocks (size x count), largest first:")
    for size, count in sorted(sizes.items(), key=lambda kv: -kv[0] * kv[1])[:15]:
        print(f"  {size / MIB:9.1f} MiB x {count:4d} = {size * count / MIB:8.0f} MiB")

    # the step's blocks alive at the peak, grouped by their stack
    groups = collections.defaultdict(lambda: [0, 0])
    for addr, size in peak_live.items():
        if addr in pre:
            continue
        key = tuple(interesting_frames(frames_of.get(addr, []), args.frames))
        groups[key][0] += size
        groups[key][1] += 1
    step_bytes = sum(size for addr, size in peak_live.items() if addr not in pre)
    print(f"\nallocated during the step and alive at the peak: {step_bytes / MIB:.0f} MiB in "
          f"{sum(c for _, c in groups.values())} blocks; by allocation site:")
    for key, (size, count) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f"\n  {size / MIB:8.1f} MiB in {count:4d} blocks")
        for line in key or ("(no frames)",):
            print(f"      {line}")


if __name__ == "__main__":
    sys.exit(main())
