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

## Continuation (branch claude/s1-startup-2, from d7d5278, main ce27a54 merged in 4aa84d4)

### soup-s1-v4 and soup-s1-v5, phases from `apply begin` (means of 2 ABBA rounds after a warm run)

Columns: apply end / Soup setup begin / setup end / train begin / first step begin / warm-up length /
waited for the warm-up / tensor loop / staging / loop waited for staging / step 0 / wall.

soup-s1-v4 (peagverr, 836ec72):
- base 3.36 / 3.87 / 7.35 / 9.15 / 9.30 / 5.33 / 0.00 / 2.18 / 0.99 / 0.17 / 0.53 / 44.73
- embed 3.34 / 3.86 / 6.93 / 9.05 / 9.21 / 5.74 / 0.15 / 1.84 / 1.08 / 0.27 / 0.54 / 45.04
- tk 3.30 / 3.74 / 7.20 / 9.07 / 9.23 / 5.54 / 0.00 / 2.19 / 1.12 / 0.18 / 0.53 / 45.17
- ep 3.30 / 3.83 / 7.37 / 9.20 / 9.30 / 5.25 / 0.00 / 2.30 / 1.02 / 0.15 / 0.55 / 45.20
- etk 3.18 / 3.67 / 6.86 / 8.79 / 8.88 / 5.48 / 0.07 / 1.89 / 1.17 / 0.23 / 0.54 / 44.63
- etkp (+ p2p probes + code table) 3.22 / 3.73 / 6.84 / 8.68 / 8.77 / 5.42 / 0.00 / 1.74 / 1.16 / 0.56 / 0.53 / 44.64
- etks (+ WARMUP_SKIP=step_ops) 3.25 / 3.75 / 6.88 / 8.69 / 8.80 / 5.24 / 0.00 / 1.92 / 1.10 / 0.23 / 0.70 / 44.88

soup-s1-v5 (namedesign, dfe0d65):
- base 3.34 / 3.85 / 7.36 / 9.12 / 9.28 / 5.28 / 0.00 / 2.16 / 0.97 / 0.17 / 0.53 / 46.48
- pread 3.35 / 3.85 / 7.04 / 8.95 / 9.11 / 5.54 / 0.00 / 2.00 / 0.71 / 0.05 / 0.54 / 46.78
- etk 3.19 / 3.71 / 6.77 / 8.80 / 8.90 / 5.65 / 0.11 / 1.88 / 1.14 / 0.31 / 0.54 / 46.54
- etkr (+ pread) 3.17 / 3.70 / 6.85 / 8.78 / 8.88 / 5.56 / 0.04 / 1.81 / 0.89 / 0.17 / 0.53 / 46.55
- etkrc (+ code table + p2p) 3.24 / 3.76 / 6.72 / 8.86 / 8.96 / 5.67 / 0.25 / 1.69 / 0.98 / 0.44 / 0.56 / 46.80
- etkrs (+ skip step_ops) 3.21 / 3.69 / 6.79 / 8.71 / 8.82 / 5.36 / 0.04 / 1.92 / 0.93 / 0.20 / 0.68 / 46.72

Reading: the three switches together (etk) bring the first step 0.42 s (v4) and 0.38 s (v5) earlier,
more than the sum of each alone (embed -0.09, tk -0.07, ep 0.00): on the 4-vCPU box the CPU the
light key frees during apply and the embedding's cast off the main thread add up. The peak is
401 MiB in every run; walls are within the loop's noise (train_loop_s varies by +-0.3 s between
runs of one variant). LOAD_PREAD alone -0.17 s, on top of etk -0.02 s (its embedding fill from the
file takes 0.74 s against 0.33 s in the thread, and the loop now waits on the warm-up, not on
staging). The code table with p2p probes: loop -0.12 s but the load waits longer for staging and
the train waits 0.25 s for the warm-up: no gain. Skipping step_ops: first step -0.06..-0.08 s,
step 0 +0.15..+0.17 s: a loss. With etk the warm-up (apply end + 5.5-5.7 s) ends together with
train begin (waits 0.07-0.11 s): from here the warm-up and the main thread are both on the
critical path.

Registration times of the warm-up on the T4 (26 runs of v4 and v5, seconds, two workers):
head_stream_kernels 1.68, rms_norm_rows_fwd 1.36, rope_qk_kernel 1.01, fused_proj_nf4_dequant3
0.82, nf4_dequant_kernel 0.79, rms_norm_rows_bwd 0.75, fused_proj_x16_rebuild 0.73,
rms_norm_rows_bwd_split 0.62, example_swiglu_fwd_kernel 0.52, step_ops 0.51,
example_swiglu_bwd_kernel 0.42, fused_proj_masked_pair_sum 0.40, cublas 0.22 (sum 9.8 s). A
two-worker list schedule of these costs gives 5.04 s in registration order and 5.06 s longest first
(lower bound 4.9 s): WARMUP_ORDER=longest no longer helps, and splitting registrations would not
either. What is left: fewer or cheaper compiles, an earlier start, or more workers.

Decided (0b1bba7): EMBED_ASYNC, TRITON_KEY=light, ENTRY_POINTS on by default; EARLY_PROBES,
LOAD_CODE_TABLE, WARMUP_SKIP off.

### soup-s1-v6 (namedesign, 0b1bba7 = new defaults on main ce27a54, means of 2 ABBA rounds)

Same columns as above (the session is ~0.1 s slower at the start than v4/v5; peak 371 MiB for all):
- new (defaults: etk) 3.36 / 3.89 / 7.23 / 9.30 / 9.41 / 5.81 / 0.12 / 1.96 / 1.19 / 0.25 / 0.51 / 44.28
- old (EMBED_ASYNC=0, TRITON_KEY=full, ENTRY_POINTS=0) 3.62 / 4.08 / 7.76 / 9.70 / 9.86 / 5.75 / 0.00 / 2.29 / 1.06 / 0.17 / 0.50 / 44.41
- w3 (3 workers) 3.37 / 3.95 / 7.62 / 9.37 / 9.46 / 4.63 / 0.00 / 2.23 / 1.27 / 0.39 / 0.49 / 43.95
- wb (WARMUP_START=apply_begin) 4.67 / 5.35 / 7.92 / 9.64 / 9.75 / 7.33 / 0.00 / 1.44 / 0.89 / 0.24 / 0.49 / 44.37
- pr (LOAD_PREAD) 3.46 / 4.01 / 7.39 / 9.49 / 9.60 / 6.04 / 0.13 / 2.01 / 0.94 / 0.15 / 0.50 / 44.31

The new defaults against the old ones: first step 0.45 s earlier (third session in a row). Three
workers end the warm-up at +8.0 s (1.3 s before train begin) but slow the main thread by about as
much as they save (model setup +0.33 s, first step +0.05 s). The warm-up from apply begin makes
apply 1.3 s longer: a loss of 0.34 s. LOAD_PREAD: +0.19 s.

Audit (aud runs, SOUP_SOLUTION_WARMUP_AUDIT=1): every one of the 16 variants the warm-up compiles is
launched in training (from 64 launches of head_finish_kernel and rms_norm_rows_bwd without the
add up to 10,688 of nf4_dequant_kernel), and nothing compiles outside the warm-up: there is no
unreachable specialization to drop (review 1, item 6: nothing to gain here).

Startup profile (aud runs, SOUP_SOLUTION_STARTUP_PROFILE=1): the main thread has a CPU 56 % of the
model setup (1.97 of 3.54 s) and 24 % of the row tokenization (0.40 of 1.69 s). The tokenization's
time is in one line, transformers' `_encode_plus` -> the Rust `encode_batch` (1.48 s of the 1.90 s
trainer build): each of the 512 calls releases the GIL and waits to get it back from the warm-up
workers (HELD_ENCODE's case). In the model setup bitsandbytes' `get_4bit_type` (functional.py:854-855,
the NF4 table per weight with a host wait) is 0.53 s of the main thread (LOAD_CODE_TABLE's case),
the wait for staging 0.30 s. The warm-up threads spend 3.1 of their 7.1 thread-seconds of the model
setup waiting for subprocesses: 2.2 s in the 12 launcher builds (cc -O0), the rest ptxas. With two
workers the warm-up is on the critical path, so the switches that shorten the main thread (HELD_ENCODE,
LOAD_CODE_TABLE) lost in v3-v5; with three workers it has 1.3 s of slack: soup-s1-v8 measures them on
top of three workers, and LAUNCHERS_EARLY again (apply now has idle CPUs since the light triton key).
Fixed in 2928db3: the launchers-ahead probe built nf4_dequant3's launcher while fused_proj.py (C1's
diet) registers nf4_dequant4.

### Lines for README (startup, switches on by default after this branch)

- Эмбеддинг при загрузке приводится к fp16 в pinned-буфер в отдельном потоке (`copy_` из того же
  среза safetensors, ядро приведения то же, что у `.to()`), главный поток в это время грузит слои;
  значения читаются только после join в конце загрузки. Побитово (тесты на маленькой и на реальной
  модели). `SOUP_SOLUTION_EMBED_ASYNC=0` возвращает исходный путь.
- Ключ кэша Triton называет libtriton.so размером, временем изменения и inode вместо sha256 от
  415 МБ (1,4 с CPU рядом с импортами на 4 vCPU). Кэш каждого прогона харнесс создаёт пустым,
  ключ только именует его записи; скомпилированные ядра те же. `SOUP_SOLUTION_TRITON_KEY=full`
  возвращает ключ Triton.
- `importlib.metadata.entry_points()` читается один раз за прогон и отвечает `select()` на тот же
  неизменяемый набор, пока не изменился `sys.path` (bitsandbytes, fsspec и плагины Soup спрашивали
  три раза по ~65 мс). Ничего не устанавливается во время прогона, ответы те же; аналог
  разрешённого кэша `packages_distributions()`. `SOUP_SOLUTION_ENTRY_POINTS=0` выключает.
- Эффект трёх вместе на T4: первый шаг на 0,38–0,45 с раньше от начала `apply()` (три сессии:
  soup-s1-v4, soup-s1-v5, soup-s1-v6), пик памяти тот же.

### soup-s1-v8 (namedesign, 2928db3, means of 2 ABBA rounds)

- new (defaults) 3.40 / 3.92 / 7.04 / 9.04 / 9.15 / 5.65 / 0.01 / 1.85 / 1.06 / 0.32 / 0.50 / 43.59
- lahc (LAUNCHERS_EARLY + HELD_ENCODE + LOAD_CODE_TABLE) 3.83 / 4.53 / 7.51 / 9.15 / 9.25 / 3.60 / 0.00 / 1.69 / 1.07 / 0.52 / 0.50 / 43.98
- la (LAUNCHERS_EARLY) 3.87 / 4.46 / 7.59 / 9.34 / 9.43 / 3.95 / 0.00 / 1.86 / 1.07 / 0.36 / 0.50 / 44.06
- w3hc (3 workers + HELD_ENCODE + LOAD_CODE_TABLE) 3.37 / 4.01 / 7.55 / 9.35 / 9.46 / 5.93 / 0.00 / 2.27 / 1.33 / 0.62 / 0.50 / 44.14
- dl (DRIVER_EARLY=libs) 3.71 / 4.51 / 7.64 / 9.46 / 9.56 / 5.29 / 0.00 / 1.88 / 1.17 / 0.27 / 0.50 / 44.27
- w3 3.40 / 4.04 / 7.71 / 9.51 / 9.62 / 4.99 / 0.00 / 2.32 / 1.34 / 0.40 / 0.50 / 44.53
- w3h (3 workers + HELD_ENCODE) 3.41 / 4.11 / 7.78 / 9.54 / 9.63 / 4.83 / 0.00 / 2.23 / 1.33 / 0.39 / 0.50 / 44.47
- hc (HELD_ENCODE + LOAD_CODE_TABLE, 2 workers) 3.34 / 3.87 / 6.96 / 9.73 / 9.85 / 6.45 / 0.70 / 1.81 / 1.10 / 0.61 / 0.50 / 44.44

Nothing beats the defaults. The GIL and the 4 vCPUs are shared: what one side gains the other
loses. hc shortens the main thread (setup end -0.08 s, the trainer build is faster) and the warm-up
grows 0.8 s, so train waits 0.70 s for it. A third worker slows the model load by 0.4-0.6 s (tensor
loop 2.3 s against 1.85). The launchers built during the imports make apply 0.43 s longer and Soup's
CLI 0.18 s longer; the warm-up then ends 1.7 s before train begin, and HELD_ENCODE with the code
table take 0.5 s off the main thread: 9.25 s, still 0.1 s behind. DRIVER_EARLY=libs makes apply
0.3 s longer and saves the warm-up only what it no longer has to wait for.

### soup-s1-v9 (namedesign, 2928db3, means of 2 ABBA rounds)

- new (defaults) 3.28 / 3.81 / 7.05 / 8.93 / 9.04 / 5.61 / 0.00 / 1.92 / 1.17 / 0.27 / 0.49
- lahc 3.63 / 4.35 / 7.35 / 8.95 / 9.04 / 3.54 / 0.00 / 1.75 / 1.09 / 0.67 / 0.49
- lahcs (+ WARMUP_START=setup) 3.70 / 4.06 / 7.14 / 8.96 / 9.06 / 4.75 / 0.00 / 1.88 / 1.19 / 0.56 / 0.49
- lahcn (+ LAUNCHER_NICE=10) 3.80 / 4.55 / 7.51 / 9.17 / 9.26 / 3.56 / 0.00 / 1.73 / 1.12 / 0.62 / 0.49
- lahcsn (+ both) 3.69 / 4.06 / 7.13 / 8.91 / 9.01 / 4.69 / 0.00 / 1.83 / 1.19 / 0.59 / 0.49

All within 0.05 s of the defaults (lahcn 0.22 s behind): the launchers ahead buy the warm-up slack
with apply time, and HELD_ENCODE with the code table spend that slack on the main thread, for no
net gain. The criterion (>= 0.1 s to the first step, no loss in the wall, bit-equal) is met only by
the three switches turned on in 0b1bba7.

### Decision (final defaults of this branch)

On: SOUP_SOLUTION_EMBED_ASYNC, SOUP_SOLUTION_TRITON_KEY=light, SOUP_SOLUTION_ENTRY_POINTS.
Off (measured on the T4, no gain): LOAD_PREAD, LOAD_CODE_TABLE, HELD_ENCODE, LAUNCHERS_EARLY,
LAUNCHER_NICE, WARMUP_START=setup|apply_begin, WARMUP_WORKERS=3, WARMUP_ORDER=longest,
WARMUP_DRIVER_EARLY=1|libs, WARMUP_SKIP=step_ops; EARLY_PROBES stays off (team decision).
Not done: the load pipeline (review 1, item 4): the tensor loop is 1.85-1.96 s with the new
defaults (under the 2 s threshold of the brief), its waits for staging 0.25-0.32 s, and the main
thread is not what bounds the first step alone.
Diagnostics added: SOUP_SOLUTION_WARMUP_AUDIT=1 (launches per compiled variant).

### Final measurements (namedesign)

- soup-s1-v10 (117f1a0, final code, new defaults against the old ones, 3 ABBA rounds after a warm
  run): first step from apply begin 9.40 against 9.85 s (-0.45 s; per run 9.41 / 9.30 / 9.49 against
  9.82 / 9.95 / 9.77), apply end 3.40 against 3.58, Soup setup end 7.05 against 7.81, tensor loop
  1.90 against 2.31 s; wall 44.11 against 45.25 s (the loop also differs by 0.75 s, partly noise);
  peak 371 MiB in both; eval loss 2.04936-2.04941 in both.
- soup-s1-full1 (1238618): 10 of 10 checks, 41.7 s / 371 MiB against the base 79.1 s / 14 345 MiB
  (s 1.895, m 38.67), eval -0.015 %, own kernels 13.9 %; first step +9.10 s after apply begin.
  pytest on the T4: 418 passed, 1 failed (test_trainer_build::test_early_probes_serve_same_answers:
  ENTRY_POINTS, now on by default, had wrapped entry_points in the shared pytest process; fixed in
  117f1a0).
- soup-s1-full2 (117f1a0): 10 of 10 checks, 46.1 s / 371 MiB against the base 87.5 s (a hot
  session: train loop 29.0 s; s 1.899, m 38.67), eval -0.014 %, own kernels 12.7 %, pytest 419
  passed, 5 skipped; first step +9.75 s after apply begin.

### soup-s1-v11 (117f1a0, 3 ABBA rounds): first step new 9.36, DRIVER_EARLY=1 9.47, WARMUP_SWITCH_MS=5 9.32 s: both off.

## State at the stop (31 Oct ~13:50, captain's order; no runs in flight)

- In main already (0b1bba7 merged as d62e731): EMBED_ASYNC, TRITON_KEY=light, ENTRY_POINTS on.
- In this branch after that, all off by default: WARMUP_AUDIT (diagnostics), WARMUP_DRIVER_EARLY=1|libs,
  the launchers-ahead probe fix (nf4_dequant4), the early-probes tests unwrap ENTRY_POINTS (fixes the
  red test on the T4: full2 pytest 419 passed), notes.
- Rulings 31 and 13: this branch still has `sys.setswitchinterval` in warmup.py `_run` (inherited) and
  CHEAP_FINGERPRINT/DATASET_HASH defaults as before. NOT merged with main 53349c8: when merging,
  take main's `_run` (no switch interval) and main's trainer_build.py defaults; conflicts expected in
  warmup.py (`_run` finally block: my DRIVER_EARLY join next to main's removed setswitchinterval;
  `start_early`: my `_early_key_and_driver` thread vs A1's hash-thread registration; `_shutdown`
  thread list) and test_warmup.py (my two tests appended at the end). The WARMUP_SWITCH_MS knob
  must go with main's change.
- Left: the torch cold-kernel warm-up (R3 item 5) not started; test_boundary::test_layer_slices_keep_the_lists
  order dependence not investigated.
