"""Where the end of a `soup train` process goes: base against our solution, measured on the run itself.

    python -m solution.bench_exit [--rounds 2] [--variant LABEL:base|solution[:K=V/K=V]] ...

Each run is the harness's own command (`harness/_train.py --config ... [--solution]`, the config from
run_task.make_config, the same environment as run_task.train: offline, empty Triton and Inductor
caches) started through solution/exit_probe.py, which timestamps the exit stages inside the process
(Trainer.train end, the adapter save, main() end, every atexit callback, the finalization's garbage
collections, the end of the module teardown) on CLOCK_MONOTONIC. This process stamps its Popen, every
output line as it arrives, the pipe's EOF and wait(), and polls /proc/<pid> every millisecond after
the child's main() is over: the moment its threads drop to one (the C-level exit() runs with every
thread alive; exit_group kills them), after which the kernel tears the process down (the address
space, the driver's file descriptors: the CUDA context and pinned pages if the runtime did not) until
wait() returns. The leader's own zombie state and empty statm come early in a threaded process (the
last thread does the teardown), so they are recorded but not used. So

    wall = popen -> ... -> train_end -> main_end -> atexit -> finalize -> C exit -> kernel exit -> wait

is split into stages, for the base (no pinned memory) and for ours. --rounds R runs every variant R
times, ABBA order, after one unmeasured run of the first variant (the first run of a session pays a
cold disk cache). Results: runs_bench/exit/<label>_<round>/ (train.log, probe.json), a table at the end.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "harness"


def _proc_state(pid):
    """(state letter, threads, resident pages) of a process from /proc, or None when it is reaped."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read()
        with open(f"/proc/{pid}/statm", "rb") as handle:
            statm = handle.read().split()
    except OSError:
        return None
    fields = stat[stat.rindex(b")") + 2:].split()
    return fields[0].decode(), int(fields[17]), int(statm[1]) if len(statm) > 1 else 0


class _Watch:
    """The child's output lines with their arrival times, and its exit as seen in /proc."""

    def __init__(self, proc, log):
        self.proc, self.log = proc, log
        self.lines, self.eof = [], None
        self.main_seen = threading.Event()
        self.marks = {}
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.poller = threading.Thread(target=self._poll, daemon=True)
        self.reader.start()
        self.poller.start()

    def _read(self):
        for raw in self.proc.stdout:
            t = time.monotonic()
            line = raw.decode("utf-8", "replace")
            self.lines.append((t, line))
            self.log.write(line)
            if "[exit-probe-main-end]" in line or "[solution] phase exit" in line:
                self.main_seen.set()
        self.eof = time.monotonic()
        self.main_seen.set()

    def _poll(self):
        pid, threads_max = self.proc.pid, 0
        while not self.main_seen.wait(0.05):
            st = _proc_state(pid)
            if st is None:
                return
            threads_max = max(threads_max, st[1])
        last_full = None
        while True:
            t = time.monotonic()
            st = _proc_state(pid)
            if st is None:
                self.marks.setdefault("reaped_seen", t)
                return
            state, threads, rss = st
            if state != "Z" and threads > 1:
                last_full = t
                self.marks["threads_last"] = threads
            elif "threads_drop" not in self.marks:
                self.marks["threads_drop"] = t
                self.marks["threads_before_drop"] = last_full
            if rss == 0 and "mm_gone" not in self.marks:
                self.marks["mm_gone"] = t
            if state == "Z" and "zombie" not in self.marks:
                self.marks["zombie"] = t
                return
            time.sleep(0.0005)


def _parse_probe(text):
    out, atexit_calls, gcs = {}, [], []
    for token in text.split():
        name, _, value = token.partition("=")
        if ":" in name:
            kind, _, label = name.partition(":")
            t0, _, dt = value.partition("+")
            (gcs if kind == "gc" else atexit_calls).append((kind, label, float(t0), float(dt)))
        else:
            out[name] = float(value)
    return out, atexit_calls, gcs


def run_one(name, side, env_extra, args, out):
    sys.path.insert(0, str(HARNESS))
    from common import OFFLINE_ENV  # noqa: E402
    from run_task import make_config  # noqa: E402

    run_dir = out / name
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    config = make_config(run_dir, Path(args.train_file).resolve(), args.seed)
    cmd = [sys.executable, str(REPO / "solution" / "exit_probe.py"), str(HARNESS / "_train.py"),
           "--config", str(config), "--metrics", str(run_dir / "metrics.json")]
    if side == "solution":
        cmd.append("--solution")
    env = dict(OFFLINE_ENV, SOUP_DB_PATH=str(run_dir / "experiments.db"),
               TRITON_CACHE_DIR=str(run_dir / "triton_cache"),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / "inductor_cache"), **env_extra)
    print(f"[bench_exit] {name}: {side} {env_extra or ''}", flush=True)
    with open(run_dir / "train.log", "w", encoding="utf-8") as log:
        t_popen = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        watch = _Watch(proc, log)
        code = proc.wait(timeout=args.timeout)
        t_wait = time.monotonic()
        watch.reader.join(5)
        watch.poller.join(5)
    probe_line = next((line for _, line in watch.lines if line.startswith("[exit-probe] ")), None)
    if probe_line is None:  # no line from the module teardown: the atexit part at least
        probe_line = next((line for _, line in watch.lines if line.startswith("[exit-probe-atexit] ")), "")
    probe, calls, gcs = _parse_probe(probe_line.split("] ", 1)[-1])
    phases = {}
    for t, line in watch.lines:
        m = re.match(r"\[solution\] phase (.+?) \+", line)
        if m:
            phases.setdefault(m.group(1), t)
    rel = lambda t: None if t is None else round(t - t_popen, 4)  # noqa: E731
    result = {"name": name, "side": side, "env": env_extra, "exit_code": code,
              "wall": rel(t_wait), "eof": rel(watch.eof),
              "probe": {k: rel(v) for k, v in probe.items()},
              "atexit": [(kind, label, rel(t0), round(dt, 4)) for kind, label, t0, dt in calls],
              "gc": [(label, rel(t0), round(dt, 4)) for _, label, t0, dt in gcs],
              "proc": {k: (rel(v) if isinstance(v, float) else v) for k, v in watch.marks.items()},
              "phases": {k: rel(v) for k, v in phases.items()}}
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        result["train_loop_s"] = metrics.get("train_loop_s")
        result["median_step_s"] = metrics.get("median_step_s")
    except (OSError, ValueError):
        pass
    (run_dir / "probe.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    _print_run(result)
    return result


def _stages(r):
    """Durations of the stages after the training loop, seconds."""
    p, proc = r["probe"], r["proc"]
    atexit_total = sum(dt for kind, _, _, dt in r["atexit"] if kind == "atexit")
    first_atexit = min((t0 for kind, _, t0, _ in r["atexit"] if kind == "atexit"), default=None)
    gc_final = sum(dt for _, t0, dt in r["gc"] if p.get("atexit_end") is not None and t0 >= p["atexit_end"])

    def d(a, b):
        return None if a is None or b is None else round(b - a, 4)

    return {
        "train_end->main_end": d(p.get("train_end"), p.get("main_end")),
        "main_end->atexit_end": d(p.get("main_end"), p.get("atexit_end")),
        "  atexit callbacks": round(atexit_total, 4),
        "  threads join": round(sum(dt for kind, _, _, dt in r["atexit"] if kind == "threads"), 4),
        "  main_end->first atexit": d(p.get("main_end"), first_atexit),
        "atexit_end->finalize_late": d(p.get("atexit_end"), p.get("finalize_late")),
        "  of it gc": round(gc_final, 4),
        "  modules_cleared": d(p.get("atexit_end"), p.get("modules_cleared")),
        "finalize_late->threads_drop": d(p.get("finalize_late"), proc.get("threads_drop")),
        "threads_drop->wait (kernel)": d(proc.get("threads_drop"), r["wall"]),
        "finalize_late->wait": d(p.get("finalize_late"), r["wall"]),
        "train_end->wait": d(p.get("train_end"), r["wall"]),
        "wall": r["wall"],
    }


def _print_run(r):
    print(f"[bench_exit] {r['name']}: exit {r['exit_code']}, wall {r['wall']} s, loop {r.get('train_loop_s')} s, "
          f"median step {r.get('median_step_s')}", flush=True)
    for key, value in _stages(r).items():
        print(f"    {key:30s} {value}")
    slow = sorted((c for c in r["atexit"] if c[3] >= 0.002), key=lambda c: -c[3])[:12]
    for kind, label, t0, dt in slow:
        print(f"    {kind:13s} {dt * 1000:8.1f} ms  {label}")
    for label, t0, dt in r["gc"]:
        if dt >= 0.002:
            print(f"    gc gen {label} at {t0} s: {dt * 1000:.1f} ms")
    print(f"    proc marks: {r['proc']}", flush=True)


def summarize(results):
    by = {}
    for r in results:
        if not r["name"].startswith("warm"):
            by.setdefault(r["name"].rsplit("_", 1)[0], []).append(r)
    print("\n[bench_exit] means per variant (s)")
    keys = list(_stages(results[0]).keys())
    labels = list(by)
    print(f"{'stage':32s}" + "".join(f"{label:>14s}" for label in labels))
    for key in keys:
        row = []
        for label in labels:
            vals = [v for v in (_stages(r)[key] for r in by[label]) if v is not None]
            row.append(f"{statistics.mean(vals):14.4f}" if vals else f"{'-':>14s}")
        print(f"{key:32s}" + "".join(row))
    for label in labels:
        calls = {}
        for r in by[label]:
            for kind, name, _, dt in r["atexit"]:
                calls.setdefault(f"{kind} {name}", []).append(dt)
        top = sorted(calls.items(), key=lambda kv: -statistics.mean(kv[1]))[:8]
        print(f"[bench_exit] {label}: slowest exit callbacks (mean ms): "
              + "; ".join(f"{name} {statistics.mean(v) * 1000:.1f}" for name, v in top))


def parse_variant(text):
    parts = text.split(":", 2)
    label, side = parts[0], parts[1] if len(parts) > 1 else "solution"
    if side not in ("base", "solution"):
        raise SystemExit(f"--variant {text}: side must be base or solution")
    env = dict(pair.split("=", 1) for pair in (re.split(r"[,/]", parts[2]) if len(parts) > 2 and parts[2] else []))
    return label, side, env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--train-file", default=str(REPO / "data" / "train.jsonl"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", default=str(REPO / "runs_bench" / "exit"))
    args = parser.parse_args()
    variants = [parse_variant(v) for v in (args.variant or ["base:base", "sol:solution"])]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(HARNESS))
    from run_task import load_task, warm_up  # noqa: E402

    if args.seed is None:
        args.seed = load_task()["training"]["seed"]
    warm_up(out)
    runs = [] if args.no_warm else [("warm_0",) + variants[0][1:]]
    for r in range(1, args.rounds + 1):
        order = variants if r % 2 else list(reversed(variants))
        runs += [(f"{label}_{r}", side, env) for label, side, env in order]
    results = [run_one(name, side, env, args, out) for name, side, env in runs]
    summarize(results)
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
