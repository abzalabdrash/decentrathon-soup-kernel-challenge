# M1: the harness on a Modal T4 (`tools/modal_run.py`)

Kaggle gives two GPU notebooks per account and a 15-25 min queue per measurement. Modal gives each
call its own container with one T4, so every session can measure at once. `tools/modal_run.py`
mirrors `tools/kaggle_run.py`: the same modes and switches, the same steps (it builds the Kaggle
notebook with kaggle_run's own `build_notebook` and runs its shell lines in order), the same output
layout and the same summary lines (kaggle_run's `summarize`).

## Switching from Kaggle

```bash
# Kaggle
wsl -d Ubuntu -- bash tools/kaggle_wsl.sh 1 --mode variants --slug soup-t15 --commit <sha> --variant a: --variant b:K=1 --repeats 2 --skip-pytest
# Modal: Git Bash on Windows with the Windows Python (3.10, modal 1.4.1 installed); no WSL
python tools/modal_run.py --profile acc3 --mode variants --slug soup-t15 --commit <sha> --variant a: --variant b:K=1 --repeats 2 --skip-pytest
```

Results go to `runs_modal/<stamp>-<mode>-<slug>/output/` (gitignored by `runs_*/`): `repo/runs*/...`
as on Kaggle (`train.log`, `metrics.json`, `result.json`, `checks.json`, `trainer_state.json`,
`eval_*.log`, `warmup.log`, plus `adapter_config.json` and `adapter_model.safetensors`), the step
log in `output/<slug>.log`, and `output/modal_status.json` (exit code and seconds of each step,
container time, PCIe width). `spec.json` and `call_id.txt` sit next to `output/`.

Every flag of kaggle_run works the same: `--mode full|solution|variants|jury|bench|diag|profile`,
`--commit <sha>|HEAD` (pushed: the container clones the team repo; the tool warns when no remote
branch has it), `--env K=V`, `--variant`, `--repeats`, `--abba`, `--warm-run`, `--jury-seed`,
`--jury-data=...`, `--bench`, `--skip-pytest`, `--no-wait`, `--attach` (with `--slug` and `--mode`:
waits for the latest run of that name and fetches it). Not needed here: `--user`, `--setup-secrets`,
`--public`, `--poll`. New: `--profile`, `--host x16|any`, `--wave`, `--waves`, `--cpu` (default 4,
Kaggle's T4 x2 has 4 vCPUs), `--memory` (MiB, default 16384), `--timeout` (container seconds,
default 5400), `--no-prewarm`, `--github-token-file`.

Runs are detached: if the terminal or the Claude session dies, the container keeps going until it
ends; fetch it with `--attach`. To stop one: `MODAL_PROFILE=<p> python -m modal app list`, then
`python -m modal app stop <app id>`. The app is named `soup-<mode>-<slug>`. On this laptop set
`PYTHONIOENCODING=utf-8` for `python -m modal ...` (the console is cp1251); the tool reconfigures
its own stdout.

## Pick an x16 host (the default)

Most Modal T4s sit on a PCIe **x8** link: 5.8 GB/s host to GPU with pinned memory, against
11.5-12.3 GB/s on Kaggle and on Modal's x16 hosts (probe of 10 containers: 9 x8, 1 x16; the host
search below saw about 1 in 6 x16). Our solution streams NF4 weights and the head over PCIe, so on
x8 its step is 0.56 s instead of 0.43 and the head waits 49 ms per step instead of 5.6: x8 numbers
do not transfer to Kaggle, and they rank PCIe-heavy variants wrongly.

So by default (`--host x16`) the tool starts `--wave` (5) containers at once; each reads
`pcie.link.width.current` from nvidia-smi before doing anything, an x8 one returns in seconds, the
first x16 one claims the run in the Modal Dict `soup-host-claims` and runs it, and the rest are
cancelled. The x16 hosts also have Kaggle's CPU clock (2.0 GHz, 14 CPUs visible, 110 GB). After
`--waves` (6) waves without an x16 host it runs on any host and says so in a WARNING line. In
practice: found in wave 1 or 2, under 2 minutes, about $0.01-0.02 of losing containers.
`--host any` takes the first container (correctness checks, quick smoke runs).

## Profiles (accounts)

`--profile <name>` or `MODAL_PROFILE`; without either, the active profile of `~/.modal.toml`
(`acc4`, which is paused: always pass one). Probe of 31 Oct 09:20 (a T4 function per workspace):

| profile | workspace | T4 | spent in October |
|---|---|---|---|
| glofcast-prod | abzalabdrash-bs | ok | $0.02 |
| acc3 | aldiarzanybek055 | ok | $8.57 (+ under $1 of M1 runs) |
| tripleacodex | tripleacodex | ok | $12.99 |
| abdrashabzal-bs | abdrashabzal-bs | ok | $19.94 |
| acc7 | abdygalimova-sh | ok | $32.61 |
| sholpanabd79 | sholpanabd79 | ok | $42.47 |
| acc6 | enotrocket2009 | ok | $44.76 |
| acc4, acc8 | | workspace paused | |
| acc5, acc9, acc10 = acc11, acc12 = nurlan-abdr2006 | | spend limit exceeded | |

The CLI shows spend (`python -m modal billing report --for "this month" --json`, sum of `Cost`),
not the credit left; a workspace out of credit answers "exceeded its spend limit" at launch, then
move to another profile. The image (built once per workspace, ~5 min, cached by Modal) and the
Volume `soup-hf-cache` (model and dataset as one 3 GB tar, filled by the first run of a workspace,
75 s) are ready on acc3, tripleacodex and glofcast-prod (image only); the other workspaces pay
both on their first run. Suggested: one profile per session, so a session that burns its
workspace's credit does not stop the others.

## Cost and concurrency

T4 $0.59/h, CPU $0.047 per core-hour, memory $0.008 per GiB-hour: about $0.85/h with the defaults.
The tool prints an estimate per run. Measured: full mode 460-550 s of container time, $0.12-0.14
(with `pytest solution`, 225 s of it); variants with two runs and `--skip-pytest` 260-300 s, $0.06-0.07;
diag 137 s.

Each call is its own ephemeral app and container: no slot files, no queue on our side. Modal
limits the GPUs running at once per workspace (the plan's limit; we have not hit it with 6
containers on one workspace); beyond it calls wait in Modal's queue. A host search uses 5
containers for a few seconds each.

## What differs from Kaggle

- **A/B comparisons stay within one container**: `--mode variants` (with `--abba`, `--repeats`,
  `--warm-run`). Never compare numbers of two Modal calls, or a Modal run with a Kaggle run: hosts
  differ in PCIe, CPU and GPU clocks, like Kaggle sessions do.
- PCIe: see above; with `--host any` most runs land on x8.
- GPU: the same Tesla T4 (70 W cap, 1590 MHz max SM clock), driver 610.57 (Kaggle 580.178), one
  GPU visible (Kaggle has two; the harness hides the second one). Under a matmul load the clock
  falls to 700-1300 MHz at the 70 W cap, as on Kaggle. The base Soup run, GPU-bound, comes out
  faster than on Kaggle (79.7 s on x16, 69 s on x8 hosts, against 83.9-92.3 on Kaggle), so `s` of a
  Modal full run is not the jury's `s`.
- Check 9 holds: nvidia-smi is there and polled the same way, idle start (3-6 MiB), peaks agree,
  the allocator env is the harness's.
- CPU: gVisor hides the model. x16 hosts: 2.0 GHz; x8 hosts: 3.1 GHz; `nproc` and
  `OMP_NUM_THREADS` follow `--cpu`. Each run log starts with nproc, MHz and a single-thread loop
  timing (`python loop of 1e7 additions`, 1.3-1.5 s) so hosts can be compared.
- Software: Python 3.13.3 (Kaggle 3.13.15), torch 2.11.0+cu128, triton 3.6.0, torchvision 0.26.0,
  torchao 0.10.0 and every dependency at Kaggle's version: `tools/kaggle_freeze.txt` is Kaggle's
  `pip freeze` (taken with `kaggle_run.py --mode bench --bench "pip freeze"`) and goes to pip as
  constraints (without it pip took datasets 5.1.0 and numpy 2.5 instead of 4.8.5 and 2.1.3). Kaggle
  has ~900 more packages (sklearn, scipy, wandb, ...); the solution's diag line shows only
  torchvision of those gets loaded on Kaggle, and it is installed here too.
- Disk: the image's files load lazily on first read, so the tool reads site-packages once before
  the steps (11 s; Kaggle's pip cell leaves them warm); `--no-prewarm` turns that off. The model is
  unpacked to local disk from the Volume (7 s) before the steps; on Kaggle the harness's warm-up
  downloads it, also before the measured runs. Kaggle's first measured run pays a cold disk cache
  (~8 s); here the first run of a container still pays about 1.3 s more than the second
  (first step and start), so use `--warm-run` for variants as on Kaggle.

## Validation (31 Oct, main 5be1da5 = a6c3fe3 plus docs)

| run | host | base s / MiB | ours s / MiB | s | median step | own kernels | checks |
|---|---|---|---|---|---|---|---|
| Kaggle full, namedesign (STATUS) | x16 | 85.0 / 14 345 | 47.0 / 401 | 1.81 | 0.451 | 13.4% | 10/10 |
| Modal full m1-x16, acc3 | x16, 2.0 GHz | 79.7 / 14 347 | **46.5 / 403** | 1.71 | 0.428 | 14.7% | **10/10** |
| Modal full m1-check2, acc3 | x8, 3.1 GHz | 69.2 / 14 345 | 49.8 / 401 | 1.39 | 0.561 | 10.0% | 10/10 |
| Modal full m1-check (datasets 5.1.0, before the constraints) | x8 | 69.5 / 14 346 | 49.3 / 402 | 1.41 | 0.559 | 10.0% | 10/10 |

Noise within one container (`--mode variants --variant base: --variant base2: --repeats 1
--skip-pytest`, the same code twice): on x16 (tripleacodex, m1-x16noise) 48.35 and 47.04 s, steps
3+ 28.52 and 28.59 s, medians 0.426 and 0.427, 404 MiB both; on x8 (tripleacodex, m1-noise)
50.44 and 50.31 s, steps 3+ 35.33 and 35.28, medians 0.561 both, 401 MiB both. Steps agree within
0.3%; the first run's start costs ~1.3 s on x16, so `--warm-run`.

Diag on x8 (m1-diag): `diag pcie pinned 64 MB x3: host->GPU 6.22 GB/s, GPU->host 6.60 GB/s`
against 12.0-12.25 / 12.8-13.1 GB/s in Kaggle's diag runs.
