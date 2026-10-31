# S1: apply begin -> first training step (branch claude/s1-startup-f929a7)

Numbers are Kaggle T4, one session per comparison, means of 2 ABBA rounds after a warm run, seconds
from `apply begin` unless noted.

## Where the time went (main a6c3fe3: soup-s1-diag0 startup profile, soup-s1-knobs)

- apply 3.3 s: imports 2.8-3.0 s (torch._dynamo 0.89 via bitsandbytes' torch.compile wrappers;
  transformers.masking_utils imports it anyway, so it cannot be skipped; datasets 0.85, peft 0.45,
  transformers.models 0.34 through squad), our modules and patches ~0.3 s; next to them triton's
  cache-key thread hashes libtriton.so (415 MB, ~1.4 CPU-s).
- Soup CLI 0.55 s, tokenizer 0.59 s, model load 3.5 s (tensor loop 2.25 s), trainer build 1.85 s,
  train() -> step 0.17 s, step 0 0.53 s (median step 0.40-0.44).
- Without the warm-up (SOUP_SOLUTION_WARMUP=0) the first step comes at 7.67 s against 9.39 s; the
  model load is 2.26 s instead of 3.58 s. The warm-up (5.5 s from apply end) ended just before train.
- The T4 box behaves CPU-bound (4 vCPUs): work moved into "idle" CPUs slows the main thread. Moving
  the 12 launcher builds into apply() made apply 0.64 s longer; holding the GIL in the row
  tokenization made the trainer build no faster and the warm-up 0.9 s longer. What helps is less
  CPU work in total, or less on the main thread.

## Switches (all off by default until measured)

- SOUP_SOLUTION_EMBED_ASYNC (load_stream.py, head_stream.py): the embedding's job returns its pinned
  buffer and a thread casts the slice into it (`copy_`, the kernel `.to()` runs); joined before the
  load returns. Bit-equal (tiny Qwen2 with and without the pinned copies, failing thread, real model).
- SOUP_SOLUTION_LOAD_PREAD (load_pinned.py, load_stream.py): the pinned staging reads each tensor's
  bytes with `preadv` from the file (byte ranges from the safetensors header, files recorded from
  `safe_open` in Soup's setup) instead of faulting in the mmap; the embedding fill too (32 MB chunks
  into a scratch, cast into the buffer). Locally the real load's staging 1.1-2.3 s -> 0.33 s, the
  tensor loop 1.6-3.0 s -> 0.77 s. Bit-equal (tiny and real model).
- SOUP_SOLUTION_TRITON_KEY=light (warmup.py): triton's cache key with libtriton.so named by size,
  mtime and inode instead of a sha256 of 415 MB (the harness gives every run an empty
  TRITON_CACHE_DIR; the key only names its entries); 0.20 s -> 0.002 s locally.
- SOUP_SOLUTION_ENTRY_POINTS (startup.py): `importlib.metadata.entry_points()` read once from apply
  begin (bitsandbytes, fsspec, Soup's plugins asked three times, ~65 ms each on Kaggle) and answered
  with `select()` while sys.path is unchanged (as the allowed packages_distributions cache).
- SOUP_SOLUTION_LOAD_CODE_TABLE (load_stream.py): bitsandbytes' NF4 table built once per load and
  cloned on the GPU for every weight (same values, own tensor per quant state).
- SOUP_SOLUTION_EARLY_PROBES=p2p (trainer_build.py): only accelerate's nvidia-smi check in the thread.
- Measuring only: SOUP_SOLUTION_HELD_ENCODE (trainer_build.py), SOUP_SOLUTION_LAUNCHERS_EARLY
  (warmup.py), SOUP_SOLUTION_WARMUP_START=setup (warmup.py).

## T4 sessions

soup-s1-v3 (094a774, first step / wall / model setup / trainer build / warm-up end):
- base 9.64 / 44.03 / 3.50 / 1.85 / 9.14
- embed 9.48 / 44.11 / 3.25 / 1.85 / 9.28 (the warm-up becomes the critical path)
- code 9.74 / 44.69 / 3.49; held 10.18 / 44.86 (waits 0.34 s for a warm-up 0.9 s longer)
- launch 10.12 / 45.05 (apply 4.24 against 3.60; warm-up 2 s shorter, not on the critical path)
- all5 9.86 / 45.03; all5s (warm-up from Soup's setup) 9.85 / 45.19 (Soup CLI 0.38 s against 0.68)

## Measured and rejected

- GIL switch interval 0.2 / 0.1 ms during the warm-up (knobs): first step 9.43 / 9.71 s against 9.39 s;
  one worker 10.27 s (waits 1.8 s for the warm-up).
- GCC precompiled header for the launchers: two header builds of ~0.25 s eat the 0.04 s per launcher.
- torch._dynamo: `torch.compile` itself imports torch._inductor -> torch._dynamo, which
  masking_utils needs anyway.
- filelock's audit hook: 39k audit events in apply at ~0.27 us = 11 ms.
- Meta-device init of the model skeleton: 0.02 s locally. `get_device` in the loop: 6 ms locally.
- TRITON_DISABLE_LINE_INFO: ~10 % of ptxas only.

## State at the hand-off (31 Oct ~11:05 Astana, branch head below)

All S1 switches are still OFF by default; nothing is flipped yet. Decide from the two sessions in
flight, then flip with the scratchpad recipe (the `_flag(..., "1")` defaults in load_stream.py,
startup.py, warmup.py `install_light_triton_key`) and run `--mode full`.

In flight on Kaggle (re-join with `wsl -d Ubuntu -- bash tools/kaggle_wsl.sh <acct> --attach --slug <slug> --mode variants`):
- peagverr (2) `soup-s1-v4`, commit 836ec72: base / embed (EMBED_ASYNC) / tk (TRITON_KEY=light) /
  ep (ENTRY_POINTS) / etk (the three) / etkp (etk + EARLY_PROBES=p2p + LOAD_CODE_TABLE) / etks (etk +
  WARMUP_SKIP=step_ops). Shows each switch alone and the combination.
- namedesign (1) `soup-s1-v5`, commit dfe0d65: base / etk / pread (LOAD_PREAD) / etkr (etk + pread) /
  etkrc (+ code table + p2p) / etkrs (+ skip step_ops). Shows LOAD_PREAD alone and on top of etk.
- Parse: per-run phases from `apply begin` in runs_var/*/solution_1/train.log (apply end,
  setup_transformers begin/end, train begin, first step begin, "warmup done in", "waited", "tensor
  loop", "pinned load copies ... read from the file with preadv").

Expected / to decide: EMBED_ASYNC is a measured win (v3: first step -0.16 s, model setup -0.25 s);
TRITON_KEY=light should cut apply (1.4 CPU-s less next to the imports); ENTRY_POINTS ~-0.1 s;
LOAD_PREAD should shorten the tensor loop (local: 1.6-3.0 s -> 0.77 s) and remove the loop's
staging wait. With EMBED_ASYNC the warm-up became the critical path (v3: warm-up end 9.28 s vs
train 9.24 s): after the load and apply cuts, WARMUP_START=setup (Soup CLI 0.68 -> 0.38 s in v3)
may become a win again: measure etkr vs etkr + SOUP_SOLUTION_WARMUP_START=setup. Keep HELD_ENCODE,
LAUNCHERS_EARLY off (losses in v3); LOAD_CODE_TABLE neutral in v3.

Also in this commit: warmup.py step_ops warm-up creates its tensors with explicit fp32 (B1's
traces: under an fp16 default dtype it failed with "Attempting to unscale FP16 gradients" and the
optimizer kernels stayed cold for step 0). A test under `torch.set_default_dtype(torch.float16)` is
still to add.

Left: decide defaults from v4/v5; one ABBA session of new defaults vs old; `--mode full` with the
10 checks; merge main (B1's step boundary defaults change the first-step duration); README lines for
the switches that stay on (embedding cast in a thread, preadv staging, light triton key, shared
entry points); tests for these switches pass locally (targeted), the full `pytest solution` goes to
Kaggle/Modal (Modal runner: tools/modal_run.py on main, profile tripleacodex for this task).

## soup-s1-v4 result (peagverr, 836ec72; ABBA x2; first step from apply begin / wall / apply / model setup)

- base 9.30 / 44.73 / 3.36 / 3.48
- embed 9.21 / 45.04 / 3.34 / 3.07 (waits 0.15 s for the warm-up: it is now the critical path)
- tk 9.23 / 45.17 / 3.30 / 3.46; ep 9.30 / 45.20
- etk 8.88 / 44.63 / 3.18 / 3.19; **etkp (etk + EARLY_PROBES=p2p + LOAD_CODE_TABLE) 8.77 / 44.64** / 3.22 / 3.11
- etks (etk + skip step_ops) 8.80, but step 0 0.70 s against 0.53: no.
So: first step -0.53 s with EMBED_ASYNC + TRITON_KEY=light + ENTRY_POINTS + EARLY_PROBES=p2p +
LOAD_CODE_TABLE, wall within noise (median step differs by 5 ms between variants). v5 (LOAD_PREAD
on top) still to fetch: `wsl -d Ubuntu -- bash tools/kaggle_wsl.sh 1 --attach --slug soup-s1-v5 --mode variants`.
Recommended defaults (pending v5): the etkp set, plus LOAD_PREAD if v5 shows it.
