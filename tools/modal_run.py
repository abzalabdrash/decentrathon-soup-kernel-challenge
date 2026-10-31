"""Run the challenge harness on a Modal.com T4 from the command line and fetch the results.

    python tools/modal_run.py --profile <modal_profile> [--commit <sha>|HEAD] [--mode full] [--slug soup-kernel-run]

The same modes, switches and output layout as tools/kaggle_run.py, so a session switches by
changing the tool name: the steps are the Kaggle notebook's shell lines, built by kaggle_run's
build_notebook (same template, same --variant / --repeats / --abba / --warm-run / --jury-* /
--bench / --skip-pytest / --env handling), and run in one Modal container with one T4:

1. header: nvidia-smi (a T4 or abort), CPU count and a single-thread loop timing, package versions;
2. the Hugging Face cache (Qwen2.5-1.5B-Instruct and the dataset) unpacked from the account's
   Volume `soup-hf-cache` to local disk (filled on the first run of an account: downloaded once,
   before the measured runs, and saved as one tar);
3. site-packages read once (the image's files load lazily; Kaggle's pip cell leaves them warm),
   unless --no-prewarm;
4. the team repo cloned into /kaggle/working/repo at --commit (the GitHub token travels as an
   ephemeral Modal Secret read from --github-token-file, goes into one git header, and is removed
   from the environment before any step runs);
5. the notebook's steps: prepare_data.py, pytest solution (unless --skip-pytest), run_task.py,
   check_rules.py, score.py, each streamed to this terminal and to output/<slug>.log;
6. the repo's runs*/ trees (logs, result/metrics/checks/profile json, trainer_state, adapter
   config and weights, pickles, json.gz; no compiler caches, no checkpoint optimizer state)
   returned as a tar and unpacked into runs_modal/<stamp>-<mode>-<slug>/output/repo/..., then the
   same summary kaggle_run prints.

Host: most Modal T4s sit on a PCIe x8 link (~6 GB/s host to GPU), Kaggle's on x16 (~12 GB/s), and
our solution streams weights over it. By default (--host x16) the tool starts --wave containers
at once; each reads its link from nvidia-smi, an x8 one returns in seconds, the first x16 one
claims the run (Modal Dict soup-host-claims) and runs it; after --waves waves without one it runs
on any host with a warning. --host any takes the first container.

Every call gets its own container and GPU (no slots); the ephemeral app keeps running when this
terminal goes away (detached), so `--attach --slug X --mode Y` fetches a run started earlier, and
`--no-wait` starts one and returns. Stop a run: `modal app stop <app id>` with the same profile.

Profiles: --profile, else MODAL_PROFILE, else DEFAULT_PROFILE (acc6). A workspace that refuses the
run (paused, over its spend limit, bad token) ends the tool with one ERROR line and exit code 2; no
container getting a T4 within --start-timeout (the account's GPU limit reached, or no free T4)
cancels the calls and exits 4; a run that dies in Modal exits 3; a run whose steps raised exits 1.
The image (Python 3.13,
torch 2.11.0+cu128, triton 3.6.0, the pinned Soup stack, build-essential) is built once per
account and cached by Modal; the first run of an account also fills the Volume. See
docs/notes/m1-modal.md.
"""

import argparse
import ast
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEAM_REPO = "https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers.git"
TOKEN_FILE = os.environ.get("SOUP_GITHUB_TOKEN_FILE", r"C:\Users\abdra\OneDrive\Desktop\github.token")
MODES = ("bench", "diag", "full", "jury", "profile", "solution", "variants")  # kaggle_run's RUN_CELLS
DEFAULT_PROFILE = "acc6"  # workspace enotrocket2009, credit left on 31 Oct (acc4, the active profile, is paused)

# T4 $0.000164/s, CPU $0.0000131 per core and second, memory $0.00000222 per GiB and second
PRICE_GPU_H, PRICE_CPU_H, PRICE_GIB_H = 0.5904, 0.0472, 0.0080

WORK = "/kaggle/working"  # the Kaggle paths, so result.json and the logs read the same
REPO_DIR = f"{WORK}/repo"
HF_HOME = "/root/.cache/huggingface"
VOL_DIR = "/hfvol"
HF_TAR = f"{VOL_DIR}/hf_cache.tar"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
KEEP = re.compile(r"(result|checks|metrics|profile|bench)\.json$|\.log$|trainer_state\.json$|\.pickle$"
                  r"|\.json\.gz$|(^|/)(adapter_config\.json|adapter_model\.safetensors|rewind\.jsonl|task\.yaml)$")
SKIP_DIRS = {"triton_cache", "inductor_cache", "__pycache__"}
MAX_FILE = 300 << 20


def early_options():
    """--profile, --slug, --mode and the container size, read before `import modal`: the profile
    is fixed when modal is imported and the function's resources when its decorator runs."""
    if __name__ != "__main__":
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile")
    parser.add_argument("--slug", default="soup-kernel-run")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--cpu", type=float, default=4.0)
    parser.add_argument("--memory", type=int, default=16384)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--github-token-file")
    known, _ = parser.parse_known_args()
    if known.profile:
        os.environ["MODAL_PROFILE"] = known.profile
    elif not os.environ.get("MODAL_PROFILE"):
        os.environ["MODAL_PROFILE"] = DEFAULT_PROFILE
    if known.github_token_file:
        os.environ["SOUP_GITHUB_TOKEN_FILE"] = known.github_token_file
    os.environ["SOUP_MODAL_APP"] = f"soup-{known.mode}-{known.slug}"
    os.environ["SOUP_MODAL_CPU"] = str(known.cpu)
    os.environ["SOUP_MODAL_MEMORY"] = str(known.memory)
    os.environ["SOUP_MODAL_TIMEOUT"] = str(known.timeout)


def local_token():
    """The GitHub read token: the file here, the injected secret in the container."""
    if __name__ != "__main__":
        return os.environ.get("SOUP_GH_TOKEN")
    try:
        return Path(os.environ.get("SOUP_GITHUB_TOKEN_FILE", TOKEN_FILE)).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


early_options()
import modal  # noqa: E402

CPU = float(os.environ.get("SOUP_MODAL_CPU", "4"))
MEMORY = int(os.environ.get("SOUP_MODAL_MEMORY", "16384"))
TIMEOUT = int(os.environ.get("SOUP_MODAL_TIMEOUT", "5400"))

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("build-essential", "git", "procps")
    # Kaggle's torch build (cu128) and its torchvision (Kaggle has it, transformers imports it)
    .pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    # Kaggle's pip freeze as constraints: every dependency at Kaggle's version (datasets 4.8.5,
    # numpy 2.1.3, huggingface_hub, tokenizers, ...); torchao too, which Kaggle has and peft may import
    .add_local_file(Path(__file__).resolve().parent / "kaggle_freeze.txt", "/root/kaggle_freeze.txt", copy=True)
    .pip_install("soup-cli[train]==0.77.0", "transformers==5.16.1", "peft==0.20.0", "trl==0.29.1",
                 "bitsandbytes==0.50.2", "accelerate==1.14.0", "pytest", "torch==2.11.0", "triton==3.6.0",
                 "torchao==0.10.0", extra_options="-c /root/kaggle_freeze.txt")
    .env({"HF_HOME": HF_HOME, "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
)
app = modal.App(os.environ.get("SOUP_MODAL_APP", "soup-harness"))
hf_volume = modal.Volume.from_name("soup-hf-cache", create_if_missing=True)
claims = modal.Dict.from_name("soup-host-claims", create_if_missing=True)  # run key -> the call on an x16 host


@app.function(image=image, gpu="T4", cpu=CPU, memory=MEMORY, timeout=TIMEOUT,
              volumes={VOL_DIR: hf_volume}, single_use_containers=True,
              secrets=[modal.Secret.from_dict({"SOUP_GH_TOKEN": local_token()})])
def run_remote(spec):
    """The notebook's steps in this container; returns a tar of the outputs and the log, or, on a
    host whose T4 is not on the PCIe link the run asks for, a short dict at once."""
    started = time.time()
    claims.put(f"{spec['claim']}:started", True)  # the launcher's sign that a container got a GPU
    link = subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.gen.max,pcie.link.width.current,"
                           "pcie.link.width.max", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.strip()
    try:
        width = int(link.split(",")[1])
    except (IndexError, ValueError):
        width = 0
    if spec["host"] == "x16":
        if width < 16:
            print(f"[modal_run] T4 on PCIe x{width} ({link}): not this host", flush=True)
            return {"host": f"x{width}"}
        if not claims.put(spec["claim"], modal.current_function_call_id(), skip_if_exists=True):
            print("[modal_run] PCIe x16, but another container already took the run", flush=True)
            return {"host": "x16, not first"}
    token = os.environ.pop("SOUP_GH_TOKEN", "") or ""
    Path(WORK).mkdir(parents=True, exist_ok=True)
    log_path = Path(WORK) / f"{spec['slug']}.log"
    log = open(log_path, "w", encoding="utf-8")
    status = {"status": "error", "steps": [], "spec": spec}

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    def sh(cmd, env=None, quiet=False):
        """bash -c `cmd`, output streamed line by line; killed at the function's deadline."""
        t0 = time.time()
        proc = subprocess.Popen(["bash", "-c", cmd], cwd=WORK, env=env or os.environ.copy(),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True)
        for line in proc.stdout:
            if not quiet:
                say(line.rstrip("\n"))
            if time.time() - started > spec["timeout"] - 180:
                os.killpg(proc.pid, 9)
                say(f"[modal_run] killed: {spec['timeout'] - 180} s of the function's time used")
                break
        code = proc.wait()
        return code, round(time.time() - t0, 1)

    def capture(cmd):
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout.strip()

    try:
        gpus = capture("nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,"
                       "power.limit,memory.used,memory.total --format=csv,noheader")
        say(f"[modal_run] gpu: {gpus}; PCIe gen max, width, width max: {link}")
        status["pcie_width"] = width
        if "T4" not in gpus or len(gpus.splitlines()) != 1:
            raise SystemExit("not exactly one Tesla T4")
        bench = capture("python -c \"import time; t = time.perf_counter(); s = 0\n"
                        "for i in range(10**7): s += i\nprint(round(time.perf_counter() - t, 3))\"")
        say(f"[modal_run] cpu: nproc {capture('nproc')}, os.cpu_count {os.cpu_count()}, "
            f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}, "
            f"{capture('grep -m1 MHz /proc/cpuinfo').replace(chr(9), '')}, "
            f"python loop of 1e7 additions {bench} s; "
            f"{capture('grep MemTotal /proc/meminfo')}; python {sys.version.split()[0]}; "
            f"request cpu {spec['cpu']} memory {spec['memory']} MiB")
        say("[modal_run] " + capture(
            "python -c \"from importlib import metadata as m; print(', '.join(f'{p} {m.version(p)}' for p in "
            "('torch', 'triton', 'torchvision', 'torchao', 'soup-cli', 'transformers', 'peft', 'trl', 'bitsandbytes', "
            "'accelerate', 'datasets', 'numpy', 'huggingface_hub', 'tokenizers', 'safetensors')))\""))

        t0 = time.time()
        Path(HF_HOME).mkdir(parents=True, exist_ok=True)
        restored = Path(HF_TAR).exists()
        if restored:
            code, _ = sh(f"tar -xf {HF_TAR} -C {HF_HOME}")
            restored = code == 0
        say(f"[modal_run] hugging face cache {'restored from the volume' if restored else 'empty: filled on this run'}"
            f" in {time.time() - t0:.1f} s")
        if spec["prewarm"]:
            t0 = time.time()
            sh("for d in $(python -c \"import sysconfig; print(sysconfig.get_paths()['purelib'])\") "
               "$(python -c \"import sysconfig; print(sysconfig.get_paths()['stdlib'])\"); do "
               "find \"$d\" -type f -print0 | xargs -0 -n 256 -P 8 cat > /dev/null; done", quiet=True)
            say(f"[modal_run] site-packages read in {time.time() - t0:.1f} s")

        auth = []
        if token:  # the token goes in a header for this one command: never in a URL, a file or the output
            basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
            auth = ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
        shutil.rmtree(REPO_DIR, ignore_errors=True)
        clone = subprocess.run(["git", *auth, "clone", "-q", spec["repo"], REPO_DIR],
                               capture_output=True, text=True)
        del auth, token
        if clone.returncode:
            raise SystemExit("git clone failed: " + clone.stderr.strip())
        if spec["commit"]:
            subprocess.run(["git", "-C", REPO_DIR, "checkout", "-q", spec["commit"]], check=True)
        say("[modal_run] repo at " + capture(f"git -C {REPO_DIR} log -1 --format='%H %s'"))

        def fill_volume():
            t0 = time.time()
            code, _ = sh(f"python -c \"from huggingface_hub import snapshot_download; "
                         f"print(snapshot_download('{MODEL}'))\"")
            parts = " ".join(p for p in ("hub", "datasets") if (Path(HF_HOME) / p).is_dir())
            tmp = f"{HF_TAR}.tmp{os.getpid()}"
            if code == 0 and parts and sh(f"tar -cf {tmp} -C {HF_HOME} {parts}")[0] == 0:
                os.replace(tmp, HF_TAR)
                hf_volume.commit()
                say(f"[modal_run] volume soup-hf-cache filled ({Path(HF_TAR).stat().st_size >> 20} MiB) "
                    f"in {time.time() - t0:.1f} s")
            else:
                say("[modal_run] could not fill the volume; the next run downloads again")

        for step in spec["steps"]:
            say(f"$ {step['cmd']}")
            code, seconds = sh(step["cmd"], dict(os.environ, **step["env"]))
            status["steps"].append({"cmd": step["cmd"], "exit": code, "s": seconds})
            say(f"[modal_run] exit {code} after {seconds} s")
            if not restored and "prepare_data.py" in step["cmd"]:
                fill_volume()
                restored = True
            if time.time() - started > spec["timeout"] - 180:
                say("[modal_run] out of time: the remaining steps are skipped")
                break
        if not restored:
            fill_volume()
        status["status"] = "complete"
    except BaseException as exc:  # noqa: BLE001 - always return what there is
        import traceback

        say(f"[modal_run] {type(exc).__name__}: {exc}")
        say(traceback.format_exc()[-3000:])
    status["elapsed_s"] = round(time.time() - started, 1)
    say(f"[modal_run] {status['status']} after {status['elapsed_s']} s")
    log.close()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=3) as tar:
        tar.add(log_path, arcname=log_path.name)
        data = json.dumps(status, indent=1).encode()
        info = tarfile.TarInfo("modal_status.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        repo = Path(REPO_DIR)
        for top in sorted(repo.glob("runs*")) if repo.is_dir() else []:
            for root, dirs, files in os.walk(top):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                for name in files:
                    path = Path(root) / name
                    rel = path.relative_to(WORK).as_posix()
                    if not KEEP.search(rel) or (name.startswith("adapter_") and "/checkpoint-" in rel):
                        continue
                    if path.stat().st_size > MAX_FILE:
                        print(f"[modal_run] not returned (over {MAX_FILE >> 20} MiB): {rel}")
                        continue
                    tar.add(path, arcname=rel)
    return buf.getvalue()


def build_steps(args, variants):
    """The Kaggle notebook's shell lines for these arguments, in order, each with the environment
    the notebook's os.environ lines set before it (the --env switches, from the run_task cell on).
    The install, clone and GPU cells are done by the container itself."""
    sys.path.insert(0, str(REPO / "tools"))
    from kaggle_run import build_notebook

    nb = build_notebook(args.repo, args.commit, args.mode, None, args.repeats,
                        [tuple(e.split("=", 1)) for e in args.env], variants, args.bench,
                        args.skip_pytest, args.abba, args.warm_run, (args.jury_data, args.jury_seed),
                        args.variant_data)
    steps, env = [], {}
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] != "code" or "REPO_URL =" in text or "pip install" in text \
                or "GITHUB_TOKEN" in text:
            continue
        for line in cell["source"]:
            line = line.rstrip("\n")
            found = re.fullmatch(r"os\.environ\[(.+?)\] = (.+)", line)
            if found:
                env[ast.literal_eval(found.group(1))] = ast.literal_eval(found.group(2))
            elif line.startswith("!"):
                steps.append({"cmd": line[1:], "env": dict(env)})
            elif line.strip() and line.strip() != "import os":
                sys.exit(f"the notebook has a line this tool does not run: {line!r}")
    return steps


def extract(data, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if not member.isfile() or name.startswith("/") or ".." in Path(name).parts:
                continue
            target = out_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def finish(data, run_dir, args):
    from kaggle_run import log_tail, summarize

    out_dir = run_dir / "output"
    extract(data, out_dir)
    status = json.loads((out_dir / "modal_status.json").read_text(encoding="utf-8"))
    hours = (status.get("elapsed_s") or 0) / 3600
    spec = status.get("spec", {})
    cost = hours * (PRICE_GPU_H + spec.get("cpu", CPU) * PRICE_CPU_H + spec.get("memory", MEMORY) / 1024 * PRICE_GIB_H)
    print(f"\n=== {status['status']}: outputs in {out_dir} ===")
    print(f"container time {status.get('elapsed_s')} s, about ${cost:.2f} on the requested resources; "
          f"T4 on PCIe x{status.get('pcie_width')}")
    summarize(out_dir, args.mode)
    for log in out_dir.glob("*.log"):
        print(f"\n--- tail of {log.name} ---\n{log_tail(log)}")
    return 0 if status["status"] == "complete" else 1


def fail(code, text):
    print(f"\nmodal_run: ERROR: {text}", flush=True)
    sys.exit(code)


def cancel(calls):
    for call in calls:
        try:
            call.cancel()
        except Exception:  # noqa: BLE001 - it may have finished on its own
            pass


def no_start(spec, args, calls, launched):
    """Cancel and exit when no container of this run got a GPU within --start-timeout: Modal
    queues calls over the workspace's GPU limit (or while no T4 is free) instead of refusing them."""
    if claims.get(f"{spec['claim']}:started") or time.time() - launched < args.start_timeout:
        return
    cancel(calls)
    fail(4, f"no container got a T4 within {args.start_timeout} s on profile {os.environ.get('MODAL_PROFILE')}: "
            "the workspace's GPU concurrency limit is reached (other runs of this account) or Modal has "
            "no free T4; the calls are cancelled. Try another --profile.")


def launch(spec, args):
    """Spawn the run. With --host x16 (the default): waves of --wave containers at once; each
    looks at its T4's PCIe link, an x8 one returns at once, the first x16 one claims the run in
    the Dict soup-host-claims and goes on. Returns the call that runs the steps."""
    launched = time.time()
    if spec["host"] == "any":
        call = run_remote.spawn(spec)
        while not claims.get(f"{spec['claim']}:started"):
            no_start(spec, args, [call], launched)
            time.sleep(3)
        return call
    seen = []
    for wave in range(1, args.waves + 1):
        calls = {}
        for _ in range(args.wave):
            call = run_remote.spawn(spec)
            calls[call.object_id] = call
        pending, deadline = set(calls), time.time() + 300
        while pending and time.time() < deadline:
            no_start(spec, args, [calls[c] for c in pending], launched)
            winner = claims.get(spec["claim"])
            if winner:
                cancel(calls[other] for other in pending - {winner})
                print(f"[modal_run] PCIe x16 host found in wave {wave} (losers: {', '.join(seen) or 'none'})",
                      flush=True)
                return calls.get(winner) or modal.FunctionCall.from_id(winner)
            for call_id in list(pending):
                try:
                    seen.append(calls[call_id].get(timeout=0)["host"])
                    pending.discard(call_id)
                except (TimeoutError, modal.exception.TimeoutError):
                    pass
                except Exception as exc:  # noqa: BLE001
                    seen.append(type(exc).__name__)
                    pending.discard(call_id)
            time.sleep(2)
        cancel(calls[c] for c in pending)
        print(f"[modal_run] wave {wave}: no x16 host ({', '.join(seen)})", flush=True)
    print(f"[modal_run] WARNING: no PCIe x16 T4 in {args.waves} waves; running on any host "
          "(host-to-GPU copies at about half of Kaggle's rate)", flush=True)
    return launch(dict(spec, host="any"), args)


def remote_has(commit):
    if not commit:
        return True
    out = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=REPO, capture_output=True,
                         text=True).stdout
    return bool(out.strip())


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", help=f"Modal profile of ~/.modal.toml (default: MODAL_PROFILE, else {DEFAULT_PROFILE})")
    parser.add_argument("--repo", default=TEAM_REPO, help="git URL of the team repository")
    parser.add_argument("--github-token-file", default=TOKEN_FILE,
                        help="file with the GitHub read token (or SOUP_GITHUB_TOKEN_FILE)")
    parser.add_argument("--commit", default="", help="commit sha to check out (pushed); HEAD = this checkout's")
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument("--slug", default="soup-kernel-run", help="run name (the app is soup-<mode>-<slug>)")
    parser.add_argument("--timeout", type=int, default=5400, help="max container time in seconds")
    parser.add_argument("--cpu", type=float, default=4.0, help="CPU cores requested (Kaggle's T4 x2 has 4)")
    parser.add_argument("--memory", type=int, default=16384, help="memory requested, MiB")
    parser.add_argument("--no-prewarm", action="store_true", help="do not read site-packages before the steps")
    parser.add_argument("--host", choices=("x16", "any"), default="x16",
                        help="x16: only a T4 on a PCIe x16 link like Kaggle's (~12 GB/s; most Modal T4s are "
                             "on x8, ~6 GB/s); any: the first container")
    parser.add_argument("--wave", type=int, default=5, help="--host x16: containers tried at once")
    parser.add_argument("--waves", type=int, default=6, help="--host x16: waves before running on any host")
    parser.add_argument("--start-timeout", type=int, default=300,
                        help="exit (code 4) when no container got a T4 within this many seconds")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="environment variable for the measured runs (repeatable), e.g. SOUP_SOLUTION_UNPAD=0")
    parser.add_argument("--jury-data", default="--train-rows 10000:10256 --eval-rows 50000:50256",
                        help="--mode jury: arguments of tools/jury_data.py (which rows train and eval)")
    parser.add_argument("--jury-seed", type=int, default=7, help="--mode jury: run_task.py --seed")
    parser.add_argument("--variant-data", default="", metavar="ARGS",
                        help="--mode variants: train every run on tools/jury_data.py ARGS (pass with '=') "
                             "with --jury-seed, as kaggle_run.py")
    parser.add_argument("--variant", action="append", default=[], metavar="LABEL:K=V,K=V",
                        help="--mode variants: one solution-only run per variant and round (repeatable)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="run_task.py --repeats N; with --mode variants, the number of rounds")
    parser.add_argument("--bench", default="", metavar="MODULE [ARGS]",
                        help="--mode bench: the module to run with python -m, and its arguments")
    parser.add_argument("--abba", action="store_true", help="--mode variants: even rounds in reverse order")
    parser.add_argument("--warm-run", action="store_true",
                        help="--mode variants: one unmeasured run of the first variant first")
    parser.add_argument("--skip-pytest", action="store_true", help="drop the pytest solution step")
    parser.add_argument("--no-wait", action="store_true", help="start the run and return (fetch it with --attach)")
    parser.add_argument("--attach", action="store_true",
                        help="wait for the latest run <stamp>-<mode>-<slug> started earlier, then fetch its outputs")
    parser.add_argument("--out", default=str(REPO / "runs_modal"))
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "tools"))
    from kaggle_run import parse_variant

    profile = os.environ.get("MODAL_PROFILE")
    if args.attach:
        runs = sorted(Path(args.out).glob(f"*-{args.mode}-{args.slug}/call_id.txt"))
        if not runs:
            sys.exit(f"no {args.out}/*-{args.mode}-{args.slug}/call_id.txt")
        run_dir = runs[-1].parent
        call_id = runs[-1].read_text(encoding="utf-8").strip()
        print(f"waiting for {call_id} ({run_dir.name}, profile {profile}) ...", flush=True)
        try:
            data = modal.FunctionCall.from_id(call_id).get()
        except Exception as exc:  # noqa: BLE001
            explain(exc, profile, "run")
        sys.exit(finish(data, run_dir, args))
    variants = [parse_variant(v) for v in args.variant]
    if (args.mode == "variants") != bool(variants):
        sys.exit("--mode variants needs --variant, and --variant needs --mode variants")
    if (args.mode == "bench") != bool(args.bench):
        sys.exit("--mode bench needs --bench, and --bench needs --mode bench")
    if args.bench and not re.fullmatch(r"[A-Za-z0-9_. =:/-]+", args.bench):
        sys.exit(f"--bench {args.bench!r}: a module name and plain arguments only")
    if args.commit == "HEAD":
        args.commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                                     text=True, check=True).stdout.strip()
    if not remote_has(args.commit):
        print(f"warning: {args.commit} is on no remote branch this checkout knows; the clone needs it pushed")
    if not local_token():
        print(f"warning: no GitHub token in {args.github_token_file}: cloning without one")

    steps = build_steps(args, variants)
    spec = {"repo": args.repo, "commit": args.commit, "mode": args.mode, "slug": args.slug,
            "steps": steps, "timeout": args.timeout, "cpu": args.cpu, "memory": args.memory,
            "prewarm": not args.no_prewarm, "host": args.host}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    spec["claim"] = f"{stamp}-{args.mode}-{args.slug}-{os.getpid()}-{time.time_ns() % 10**6}"
    run_dir = Path(args.out) / f"{stamp}-{args.mode}-{args.slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"modal {app.name} on {profile} ({args.mode}, commit {args.commit or 'default branch'}, "
          f"{len(steps)} steps) ...", flush=True)
    stage = "start"
    try:
        with modal.enable_output():
            with app.run(detach=True):
                stage = "launch"
                call = launch(spec, args)
                (run_dir / "call_id.txt").write_text(call.object_id, encoding="utf-8")
                print(f"call {call.object_id}; outputs go to {run_dir}", flush=True)
                if args.no_wait:
                    print(f"detached: python tools/modal_run.py --profile {profile} "
                          f"--attach --mode {args.mode} --slug {args.slug}")
                    return
                stage = "run"
                data = call.get()
    except Exception as exc:  # noqa: BLE001 - Modal's refusals come as several exception types
        explain(exc, profile, stage)
    sys.exit(finish(data, run_dir, args))


def explain(exc, profile, stage):
    """A refusal or failure from Modal as one clear line and a non-zero exit, not a traceback."""
    text = " ".join(str(exc).split())
    low = text.lower()
    if "spend limit" in low or "billing" in low or "credit" in low:
        why = "the workspace is over its spend limit (its credit is used up): use another --profile"
    elif "paused" in low:
        why = "the workspace is paused (billing or verification in the Modal dashboard): use another --profile"
    elif "token" in low or "auth" in low or "profile" in low:
        why = "the profile's token is refused or the profile is unknown (python -m modal profile list)"
    elif stage == "run":
        why = "the run failed in Modal (container stopped: out of credit, preempted or timed out); no outputs"
    else:
        why = "Modal refused the run"
    fail(3 if stage == "run" else 2,
         f"profile {profile}, {stage}: {why}. Modal said: {type(exc).__name__}: {text[:400]}")


if __name__ == "__main__":
    main()
