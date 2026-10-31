# L1: apply begin -> first training step (branch claude/l1-startup)

Kaggle T4 (account slerfas), one session per comparison, `--mode variants --abba --warm-run`,
means of 2 ABBA rounds; seconds from `[solution] phase apply begin` unless noted. Phases parsed
from each run's train.log (apply end, setup_transformers begin/end, train begin, first step begin,
"warmup done in", "waited ... for the warmup", "tensor loop", staging).

## Where the time went on main 8387938 (soup-l1-p0, startup sampler `SOUP_SOLUTION_STARTUP_PROFILE=1`)

- apply 3.1-3.4 s (imports 3.0 s at 96 % of a CPU), Soup CLI 0.45 s, model setup 3.3 s, trainer
  build 3.7 s, train() -> first step 0.3 s: first step at +10.9 s. The warm-up (7.2 s from apply end)
  ended 0.1-0.3 s before train begin.
- Without the warm-up (`SOUP_SOLUTION_WARMUP=0`): model setup 2.0 s instead of 3.3, trainer build
  3.5 s: the trainer build is slow by itself.
- Trainer build (3.85 s sampled): 3.34 s in datasets' fingerprint of Soup's tokenization map
  (`Hasher.hash(checked_format_row)`), of which 2.87 s is dill's pure-Python pickle of the tokenizer
  (151,643-entry `_vocab`, 151,387 `_merges` tuples: ~760 k save() calls), 0.30 s dill's `dis` text
  scan for globals, 0.41 s filelock's Python audit hook (`_audit_fork_safety`, installed at filelock's
  import; pickle calls `id()` 1.2 M times and each is a `builtins.id` audit event). The tokenizer is
  pickled exactly once per run (the second map, Soup's assistant masks, and TRL's truncation hash in
  milliseconds), so a memo of its hash saves nothing. This is what the answer-13 compliance cost
  (CHEAP_FINGERPRINT and DATASET_HASH off: trainer build 1.8 s in m6 -> 3.8-4.3 s in m7/m8).
- Model setup (3.27 s, main thread at 55 % of a CPU): tokenizer load 0.53 s, the skeleton's
  `nn.Linear.__init__` on meta 0.37 s (kaiming_uniform_ through the device context: 0.28 s), tensor
  loop 1.89 s (bitsandbytes' `get_4bit_type` per weight 0.55 s, waits for staging 0.36 s), peft 0.27 s.
  The warm-up threads spend 2.1 s of that window waiting for the C compiler (one `cc` per launcher).
- The GIL switch interval is the interpreter's 5 ms (the organizers forbade changing it): locally, on 4
  CPUs (`taskset -c 0-3`), the whole warm-up takes 3.9 s with the main thread idle, 4.0 s with it
  issuing small torch ops, 14 s with it running pure Python. Pure-Python phases of the main thread
  (dill's pickle) starve the warm-up; the warm-up's GIL handoffs slow the main thread's GIL retakes.

## Switches of this branch (all off by default)

- `SOUP_SOLUTION_LAUNCHERS_ONE` (launcher_one.py, started from warmup.py `_run`): the probe runs the
  queued registrations of the compile-only kernel modules (an explicit list) once with triton's
  `jit_cache_hook` answering "skip", their `make_launcher` sources go into one translation unit (each
  source's file-scope names renamed by `#define`, a module init that keeps each launcher's own module),
  one `cc` run, served to triton's `compile_module_from_src` by source text. Kernels (cubins) and launch
  code are triton's. Locally one launcher 0.22 s, 13 in one TU 0.34 s; whole warm-up on 4 CPUs
  3.3 -> 1.6 s. Tests launch real kernels through the combined launchers (bit-exact results), unknown
  sources and failed builds take triton's own build. T4: warm-up 6.95 -> 4.91 s (soup-l1-v2).
- `SOUP_SOLUTION_LOAD_CODE_TABLE` (existing, S1): bitsandbytes' NF4 table built once per load and cloned
  per weight. With the warm-up shorter it pays now: first step -0.27 s, wall -0.33 s (soup-l1-v2).
- `SOUP_SOLUTION_META_INIT` (load_stream.py, scoped to Soup's `_setup_transformers`): torch's init
  functions return a meta tensor untouched (the skeleton's reset_parameters on meta). Tests: generator
  states, all LoRA A/B tensors and every loaded weight bit-equal after the whole setup without
  reseeding; a real tensor gets the same values and leaves the generator in the same state.
- `SOUP_SOLUTION_LOAD_EARLY` (load_early.py, started first in `startup.prepare_imports`; answer 48):
  the checkpoint's data sections read with preadv into host memory during the imports, registered with
  the driver per chunk (chunks end on tensor boundaries, read in the loader's natural key order: the
  file holds the layers alphabetically); load_pinned.py stages a tensor as a view of its chunk,
  load_stream.py's embedding cast reads from it; only for files transformers opens with the same real
  path, size and mtime; buffers dropped when the load uses none of them, after the load's final
  synchronize, and at the end of Soup's setup. Which files: the `base` of the `--config` Soup runs with
  (a hint of what to read, not a mode switch; without `--config` nothing is read), its local directory
  or HF cache snapshot. Tests: tiny load bit-equal (with and without the async embedding cast), failed
  read, other files dropped.
- `SOUP_SOLUTION_EARLY_ROWS` (existing, night): Soup's own build_format_row and `_map_text_sft_rows`
  (the same Dataset.map, datasets' own fingerprint: now checked in test_trainer_build.py) in a thread
  during the model load, so the 2.9 s of dill hashing leaves the main thread's path (answer 15).
- `SOUP_SOLUTION_TRACKER_THREAD` (tracker_async.py, hooked from startup.patch_soup_first_step_mark;
  answer 16): Soup's per-log INSERT + commit (fsync per row) by a writer thread with its own connection
  to the same database, the timestamp taken at the call, rows in call order; every other tracker use
  waits for the queue through `_get_conn`; exit waits; writer errors raised by the next call.

Deleted after the organizers' answers 8, 9 (and 13, 31): TOKENIZER_PICKLE (a byte-identical faster
pickle of the tokenizer: soup-l1-v1 first step -0.91 s, wall -0.70 s; "no" to question 8),
FAST_FINGERPRINT, DATASET_HASH with `install_fast_tokenizer_reducer`, CHEAP_FINGERPRINT,
`startup.freeze_imported_objects`, the warm-up's switch-interval code, EARLY_PROBES.

FAST_ROWS (on by default until 2a7f33e) changes datasets' fingerprint of Soup's map: on Soup's own
`_map_text_sft_rows`, the real Qwen tokenizer and 32 task rows, the rows are identical and the
fingerprint is 81be3add1e3b65ff without, d84e8ceaa6a009e2 with. dill gathers the globals to pickle
with `dill.detect.globalvars(recurse=True)`, which walks every function reachable through global names
(Soup's loss_mask helpers among them) and adds the builtins their code uses; the replacements use
`map`, `reversed`, `set`, `tuple`, `str`, `all`, `Exception`, which the originals do not. Named like the
originals: no change. As callable objects (which dill does not walk): the originals' own names
(`enumerate`, `hasattr`) disappear instead. An identical fingerprint would need the replacement to use
exactly the originals' global names. Its gain locally: 0.176 -> 0.076 s for the 256 rows.

Measured, not kept: `SOUP_SOLUTION_HELD_ENCODE` (+0.45 s), `TRITON_STORE_BINARY_ONLY=1` (-0.1 s,
within noise; it makes triton write only the cubin and the metadata JSON to the run's own cache
directory instead of every IR stage, nothing reused between runs).

## T4 sessions

soup-l1-v1 (3688838; first step / train begin / warm-up end / wall / waited for the warm-up):
- base 10.93 / 10.62 / 10.50 / 48.42 / 0.00
- tp (TOKENIZER_PICKLE, deleted) 10.02 / 9.71 / 9.71 / 47.72 / 0.21
- tpf (+ FAST_FINGERPRINT, deleted) 10.03 / 9.71 / 9.71 / 47.80 / 0.29
- tpfh (+ HELD_ENCODE) 10.47 / 10.15 / 10.16 / 48.24 / 0.61
- tpfs (+ TRITON_STORE_BINARY_ONLY) 9.93 / 9.62 / 9.53 / 47.75 / 0.18
Peak 395 MiB in all, own kernels 16.2-16.4 %, check 9 clean.

soup-l1-v2 (f73bcec; first step / train begin / warm-up length / warm-up end / wall):
- base 11.15 / 10.84 / 6.95 / 10.19 / 47.25
- lo (LAUNCHERS_ONE) 10.77 / 10.46 / 4.91 / 8.20 / 47.00
- tpf (deleted) 10.27 / 9.96 / 6.62 / 9.96 / 46.30
- tpflo 9.88 / 9.56 / 5.42 / 8.76 / 46.23
- tpfloc (+ LOAD_CODE_TABLE) 9.61 / 9.28 / 4.62 / 7.95 / 45.90
Peak 395 MiB, own kernels 16.9 %.

soup-l1-v3 (d1f102f; first step / train begin / setup end / warm-up length / wall):
- base 11.34 / 11.02 / 6.90 / 7.37 / 46.29
- lo (LAUNCHERS_ONE) 10.82 / 10.50 / 6.76 / 4.99 / 46.11
- lofr (lo + FAST_ROWS=1, since deleted) 10.68 / 10.36 / 6.86 / - / 45.86
- lor (lo + EARLY_ROWS) 11.38 / 11.06 / 9.63 / 5.98 / 46.55: no-go, the thread's pure-Python dill hash
  (2.9 s) holds the GIL for whole 5 ms intervals and the main thread's model load (thousands of torch
  ops, each releasing and retaking the GIL) grows by 2.9 s
- lorc / lorcm / lorcme 11.35 / 11.26 / 11.22 (on top of EARLY_ROWS); LOAD_EARLY read 2944 MiB in
  2.13 s from apply begin, all 338 tensors staged from it, the loop's staging wait 0
Peak 395 MiB, own kernels 17.1-17.3 %.

Not done, and why: the speculative NF4 quantization in a thread before from_pretrained (answer 13) runs
into the same GIL handoffs: each torch op of the thread waits up to 5 ms while the main thread imports
or runs Soup's CLI (pure Python), about 10-15 ops per weight for 196 weights. filelock's audit hook
(0.41 s of the trainer build, every `builtins.id` event of dill's pickle) is installed before apply()
and audit hooks cannot be removed. Imports in a thread (answer 14): the main thread is the importer and
both are GIL-bound.

soup-l1-v4 (7661609; each variant on top of the previous; first step / setup end / tensor loop / warm-up
length / wall):
- base 11.42 / 6.94 / 1.90 / 7.44 / 46.53
- lo (LAUNCHERS_ONE) 10.92 / 6.84 / 1.88 / 4.82 / 45.91
- loc (+ LOAD_CODE_TABLE) 10.85 / 6.75 / 1.85 / 5.19 / 45.83
- locm (+ META_INIT) 10.82 / 6.70 / 1.97 / 5.12 / 45.86
- locme (+ LOAD_EARLY) 10.76 / 6.55 / 1.35 / 5.02 / 45.83 (staging wait 0; apply +0.11 s from the reading
  thread next to the imports)
- locmet (+ TRACKER_THREAD) 10.83 / 6.66 / 1.46 / 5.07 / 45.70
Peak 395 MiB, own kernels 17.2-17.4 %. The int8 kernels (I1) still compile outside the warm-up at steps
1-3 (their own thread); their launchers now come from the one build.

soup-l1-full1 (7661609, all five switches by `--env`): all 10 checks; 47.1 s / 395 MiB against the base
84.3 s / 14 345 MiB (speedup 1.79, memory 36.3), eval -0.001 %, own kernels 16.7 %, Soup 79 passed.
pytest solution 440 passed, 4 failed: test_jury_warmup_exit x2 (called `warmup._run` with the removed
switch-interval argument: fixed in d43c313), test_recompute[lengths0] (memory bound; fails in P1's full
run too), test_boundary::test_layer_slices_keep_the_lists (passes alone with and without the switches;
in the whole process boundary.py's watched set keeps ids of collected modules and a new ModuleList can
reuse one: B1's area).
soup-l1-jury1 (d43c313, seed 151, train rows 38000:38256, eval rows 48000:48256, all five switches):
all 10 checks; 47.8 s / 415 MiB against 83.2 s / 14 353 MiB (speedup 1.74, memory 34.6), eval +0.021 %,
own kernels 17.0 %; pytest 442 passed, 2 failed (the recompute and boundary tests above).

Decision (1e91696): LAUNCHERS_ONE and LOAD_CODE_TABLE on by default (together first step -0.57 s, wall
-0.70 s in v4). Off, with their numbers: META_INIT (-0.03 s), LOAD_EARLY (-0.06 s, 2.9 GB pinned host
memory during the load), TRACKER_THREAD (wall -0.13 s, one session), EARLY_ROWS (+0.5 s).

soup-l1-full2 (1e91696, the new defaults, no `--env`): all 10 checks; 46.4 s / 395 MiB against the base
80.2 s / 14 345 MiB (speedup 1.73 in a fast base session, memory 36.3), eval +0.004 %, own kernels 17.0 %,
Soup 79 passed; pytest solution 443 passed, 1 failed (test_recompute[lengths0], as in P1's runs; the
boundary test passed this time).

## Lines for README (switches on by default)

- Лаунчеры ядер Triton собираются одним запуском компилятора C: поток прогрева один раз прогоняет
  регистрации прогрева с `knobs.runtime.jit_cache_hook` Triton, отвечающим «пропустить» (ничего не
  компилируется и не запускается, видны только сигнатуры), берёт исходники лаунчеров у `make_launcher`
  Triton и собирает их в одну единицу трансляции (имена каждого исходника переименованы через
  `#define`, каждый лаунчер сохраняет свой модуль). `compile_module_from_src` Triton получает готовый
  модуль по тексту исходника; любой другой исходник и неудачная сборка идут обычным путём Triton.
  Кубины и код запуска те же. На T4 прогрев 7,0–7,4 → 4,8–5,0 с. `SOUP_SOLUTION_LAUNCHERS_ONE=0`
  выключает.
- Таблица NF4 bitsandbytes строится один раз за загрузку и копируется для каждого веса (те же 16
  значений, свой тензор у каждого quant state); без этого `get_4bit_type` на каждый из 196 весов
  создавал её заново с ожиданием GPU. `SOUP_SOLUTION_LOAD_CODE_TABLE=0` выключает.
