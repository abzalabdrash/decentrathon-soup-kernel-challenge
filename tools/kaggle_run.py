"""Run the challenge notebook on Kaggle from the command line and fetch the results.

    python tools/kaggle_run.py --user <kaggle_user> --repo https://github.com/<team>/<repo>.git \
        [--commit <sha>] [--mode full|solution|profile] [--slug soup-kernel-run] [--public] [--no-wait]

What it does:
1. builds a notebook from templates/kaggle_notebook.ipynb with REPO_URL / COMMIT filled in and the
   measurement cell set for the chosen mode;
2. writes kernel-metadata.json (GPU T4 x2, internet on, private by default) and pushes it with the
   Kaggle CLI (`kaggle kernels push --accelerator NvidiaTeslaT4`);
3. polls `kaggle kernels status` until the run ends;
4. downloads the log and the json results into runs_kaggle/<timestamp>/ and prints the summary.

Needs the Kaggle CLI (`pip install kaggle`) and a token: KAGGLE_API_TOKEN in the environment or
~/.kaggle/kaggle.json. A private GitHub repository needs a GitHub token on Kaggle. Kaggle secrets
added in the notebook editor are not attached to notebooks pushed through the API, so the token
goes into a private Kaggle dataset of the same account instead:

    python tools/kaggle_run.py --user <kaggle_user> --setup-secrets --github-token-file <file>

creates (or updates) the private dataset <kaggle_user>/soup-secrets with github_token.txt; every
later push attaches it and the notebook reads the token from /kaggle/input/soup-secrets/ when the
GITHUB_TOKEN secret is missing. Pass --no-secrets-dataset for a public repository.

Modes:
  full      harness/run_task.py (base + solution), then check_rules.py and score.py   ~10-12 min
  solution  harness/run_task.py --only solution --out runs_try (quick: checks 1,5-8 will fail)
  profile   harness/run_task.py --profile (full run plus a profiled solution run)
"""

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "templates" / "kaggle_notebook.ipynb"

RUN_CELLS = {
    "full": ["!cd /kaggle/working/repo && python harness/run_task.py\n"],
    "solution": ["!cd /kaggle/working/repo && python harness/run_task.py --only solution --out runs_try\n"],
    "profile": ["!cd /kaggle/working/repo && python harness/run_task.py --profile\n"],
}
CHECK_CELLS = {
    "full": ["!cd /kaggle/working/repo && python harness/check_rules.py\n",
             "!cd /kaggle/working/repo && python harness/score.py runs/result.json\n"],
    "solution": ["!cd /kaggle/working/repo && python harness/check_rules.py --result runs_try/result.json --skip-tests\n"],
    "profile": ["!cd /kaggle/working/repo && python harness/check_rules.py\n",
                "!cd /kaggle/working/repo && python harness/score.py runs/result.json\n"],
}


SECRET_MARKER = '    print("No GITHUB_TOKEN secret, cloning without it:", type(exc).__name__)\n'
SECRET_FALLBACK = [
    'if not token:\n',
    '    import glob\n',
    '    # private dataset soup-secrets of this account (secrets are not attached to API-pushed notebooks)\n',
    '    found = glob.glob("/kaggle/input/**/github_token.txt", recursive=True)\n',
    '    if found:\n',
    '        token = open(found[0], encoding="utf-8").read().strip()\n',
    '        print("GitHub token read from", found[0])\n',
    '    else:\n',
    '        print("no github_token.txt under /kaggle/input:", sorted(glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*")))\n',
]


def inject_token_fallback(cell):
    lines = cell["source"]
    for i, line in enumerate(lines):
        if line == SECRET_MARKER:
            cell["source"] = lines[: i + 1] + SECRET_FALLBACK + lines[i + 1:]
            return True
    return False


def build_notebook(repo_url, commit, mode, secrets_dataset=None):
    nb = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    cells = nb["cells"]
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    # cell 1: REPO_URL / COMMIT
    code_cells[0]["source"] = [f'REPO_URL = "{repo_url}"  # your repository\n',
                               f'COMMIT = "{commit}"  # the commit you submit; empty = the default branch\n']
    # the measurement cell is the one that calls run_task.py; the check cell calls check_rules.py
    for cell in code_cells:
        text = "".join(cell["source"])
        if "harness/run_task.py" in text:
            cell["source"] = RUN_CELLS[mode]
        elif "harness/check_rules.py" in text:
            cell["source"] = CHECK_CELLS[mode]
        elif secrets_dataset and "GITHUB_TOKEN" in text and not inject_token_fallback(cell):
            sys.exit("the clone cell of the template changed: cannot inject the token fallback")
    for cell in cells:
        if cell["cell_type"] == "code":  # markdown cells must not carry outputs (nbformat rejects it)
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def kaggle(*args, check=True):
    cmd = ["kaggle", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and proc.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout + proc.stderr


def wait_for(kernel, poll_s):
    started = time.time()
    last = ""
    while True:
        out = kaggle("kernels", "status", kernel, check=False).strip()
        if out != last:
            print(f"[{int(time.time() - started):5d} s] {out}", flush=True)
            last = out
        low = out.lower()
        if "complete" in low:
            return "complete"
        if "error" in low or "fail" in low:
            return "error"
        if "cancel" in low:
            return "cancelled"
        time.sleep(poll_s)


def summarize(out_dir, mode):
    sub = "runs_try" if mode == "solution" else "runs"
    result = None
    for path in out_dir.rglob("result.json"):
        if sub in path.parts:
            result = json.loads(path.read_text(encoding="utf-8"))
            break
    if result is None:
        print("no result.json in the output (the run failed before writing it); see the log")
        return
    summary = result.get("summary", {})
    for side in ("base", "solution"):
        info = summary.get(side)
        if info and info.get("ok"):
            print(f"{side:9s} time {info['wall_s']:8.1f} s   peak {info['gpu_peak_mib']:6d} MiB   "
                  f"train loop {info.get('train_loop_s')} s   eval loss {info.get('eval_loss')}")
        elif info:
            print(f"{side:9s} FAILED: {info.get('failed')}")
    ratios = summary.get("ratios")
    if ratios:
        print(f"speedup {ratios['speedup']:.3f}x   memory base/solution {ratios['memory_ratio']:.3f}x   "
              f"eval loss vs base {ratios.get('quality_delta_pct')} %")
    for run in result.get("runs", {}).get("solution", []):
        m = run.get("metrics", {})
        own = m.get("own_kernel_ms", {})
        loop = m.get("timed_loop_s") or 0
        share = sum(own.values()) / (1000 * loop) if loop else 0
        print(f"{run['run']}: own kernels {100 * share:.1f}% of timed steps; "
              f"median step {m.get('median_step_s')} s; apply {m.get('apply_s')} s; "
              f"torch peak reserved {m.get('torch_max_reserved_mib')} MiB")
        if own:
            top = sorted(own.items(), key=lambda kv: -kv[1])[:8]
            print("   " + ", ".join(f"{k} {v:.0f} ms" for k, v in top))
    for path in out_dir.rglob("checks.json"):
        checks = json.loads(path.read_text(encoding="utf-8"))
        for c in checks.get("checks", []):
            print(f"{'PASS' if c['ok'] else 'FAIL'}  {c['check']}  {c['detail'][:160]}")
        for note in checks.get("notes", []):
            print(f"NOTE  {note}")
        break


def setup_secrets(user, token_file, out):
    """Create or update the private dataset <user>/soup-secrets holding github_token.txt."""
    token = Path(token_file).read_text(encoding="utf-8").strip()
    if not token:
        sys.exit(f"{token_file} is empty")
    folder = Path(out) / f"secrets-{user}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "github_token.txt").write_text(token + "\n", encoding="utf-8")
    meta = json.dumps({"title": "soup-secrets", "id": f"{user}/soup-secrets",
                       "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    for name in ("dataset-metadata.json", "datasets-metadata.json"):  # the CLI's help names both
        (folder / name).write_text(meta, encoding="utf-8")
    out_text = kaggle("datasets", "create", "-p", str(folder), "-q", check=False)
    low = out_text.lower()
    if "already" in low or "exists" in low or "409" in low or "duplicate" in low:
        out_text = kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-q", check=False)
        low = out_text.lower()
    print(out_text.strip())
    if not any(word in low for word in ("success", "created", "updated", "uploaded", "complete")):
        sys.exit(f"could not create or update {user}/soup-secrets, see the output above")
    print(f"private dataset {user}/soup-secrets ready (processing can take a minute)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"), help="Kaggle username")
    parser.add_argument("--repo", help="git URL of the team repository")
    parser.add_argument("--setup-secrets", action="store_true",
                        help="create/update the private dataset <user>/soup-secrets and exit")
    parser.add_argument("--github-token-file", help="file with the GitHub token (with --setup-secrets)")
    parser.add_argument("--no-secrets-dataset", action="store_true",
                        help="do not attach <user>/soup-secrets (public repository)")
    parser.add_argument("--commit", default="", help="commit sha to check out; empty = default branch")
    parser.add_argument("--mode", choices=sorted(RUN_CELLS), default="full")
    parser.add_argument("--slug", default="soup-kernel-run", help="kernel slug (one per purpose)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--public", action="store_true", help="make the kernel public (for the submission)")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--timeout", type=int, default=5400, help="max run time in seconds")
    parser.add_argument("--poll", type=int, default=30)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--out", default=str(REPO / "runs_kaggle"))
    args = parser.parse_args()
    if not args.user:
        sys.exit("--user or KAGGLE_USERNAME is required")
    if args.setup_secrets:
        if not args.github_token_file:
            sys.exit("--setup-secrets needs --github-token-file")
        setup_secrets(args.user, args.github_token_file, args.out)
        return
    if not args.repo:
        sys.exit("--repo is required")
    secrets_dataset = None if args.no_secrets_dataset else f"{args.user}/soup-secrets"
    if args.commit == "HEAD":
        args.commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True, check=True).stdout.strip()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    push_dir = Path(args.out) / f"{stamp}-{args.mode}-{args.slug}"
    push_dir.mkdir(parents=True, exist_ok=True)
    nb_name = f"{args.slug}.ipynb"
    (push_dir / nb_name).write_text(
        json.dumps(build_notebook(args.repo, args.commit, args.mode, secrets_dataset),
                   ensure_ascii=False, indent=1), encoding="utf-8")
    kernel = f"{args.user}/{args.slug}"
    meta = {"id": kernel, "title": args.title or args.slug, "code_file": nb_name,
            "language": "python", "kernel_type": "notebook",
            "is_private": "false" if args.public else "true",
            "enable_gpu": "true", "enable_internet": "true", "machine_shape": args.accelerator,
            "dataset_sources": [secrets_dataset] if secrets_dataset else [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (push_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"pushing {kernel} ({args.mode}, commit {args.commit or 'default branch'}) ...", flush=True)
    print(kaggle("kernels", "push", "-p", str(push_dir), "--accelerator", args.accelerator,
                 "--timeout", str(args.timeout)).strip())
    print(f"https://www.kaggle.com/code/{kernel}")
    if args.no_wait:
        return
    status = wait_for(kernel, args.poll)
    out_dir = push_dir / "output"
    out_dir.mkdir(exist_ok=True)
    print(kaggle("kernels", "output", kernel, "-p", str(out_dir), "-o", "-q",
                 "--file-pattern", r"(result|checks|metrics|profile)\.json$|\.log$|trainer_state\.json$",
                 check=False).strip())
    print(f"\n=== {status}: outputs in {out_dir} ===")
    summarize(out_dir, args.mode)
    for log in out_dir.glob("*.log"):
        print(f"\n--- tail of {log.name} ---\n{log_tail(log)}")


def log_tail(path, lines=60):
    """Kaggle writes the notebook log as a JSON list of {stream_name, time, data}; show the text."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        entries = json.loads(text)
        out = "".join(str(e.get("data", "")) for e in entries if isinstance(e, dict))
    except ValueError:
        out = text
    return "\n".join(out.splitlines()[-lines:])


if __name__ == "__main__":
    main()
