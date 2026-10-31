# L2: speculative load and the start's GIL budget (branch claude/l2-spec-load)

Kaggle T4 (account slerfas), `--mode variants --abba --warm-run`, means of 2 ABBA rounds, seconds from
`[solution] phase apply begin`. Base = main 0b26618 (LAUNCHERS_ONE and LOAD_CODE_TABLE on).

## Where the start stands (soup-l2-p0, startup sampler)

apply 3.2-3.5 s (imports 3.2 s at 96 % of a CPU), Soup CLI 0.5 s, model setup 3.3 s (the main thread on a
CPU 47 % of it; the warm-up threads 6.1 CPU-s meanwhile), trainer build 3.9 s (main 73 %; datasets' dill
hash of the tokenizer 2.9 s), train() -> first step 0.3 s: first step at +10.6..10.9 s. The warm-up (4.6-5.0
s from apply end with LAUNCHERS_ONE) ends ~2.5 s before train begin.

Model setup on the main thread (sampled, incl. GIL waits): Condition.wait 0.78 s (staging 0.55: LOAD_EARLY
removes it; the pinner's embedding buffer 0.21: LOAD_PREPIN), Linear.__init__ of the skeleton 0.49 s
(META_INIT), quantize_4bit 0.56 s (LOAD_SPEC), tokenizer 0.53 s, replace_with_bnb_linear 0.25 s.

The process is GIL-saturated from apply begin to the first step: the main thread's CPU work (~8.3 s:
imports 3.1, model setup 1.55, trainer build 2.7, the rest ~1) plus the warm-up's GIL work add up to the
first step. With the switch interval at the interpreter's 5 ms (the organizers forbade changing it), every
torch op of the main thread's load (each releases and retakes the GIL) waits behind a warm-up thread that
holds it. Cutting the main thread's own work therefore pays less than 1:1 while the warm-up wants the
GIL, and moving the warm-up only moves the cost:
- one warm-up worker (`SOUP_SOLUTION_WARMUP_WORKERS=1`, soup-l2-v2): model setup 0.7-1.0 s shorter
  (setup end 7.25 -> 6.26-6.52), but the warm-up takes 7.6 s, ends at train begin and train waits up to
  0.8 s: first step -0.1..-0.26 s;
- the warm-up from apply begin (two workers, soup-l2-p0): apply +1.0 s, first step +0.32 s.

## Switches of this branch (off by default)

- `SOUP_SOLUTION_LOAD_SPEC` (load_spec.py; answer 13; needs LOAD_EARLY): once apply()'s imports are done a
  thread quantizes the layers' 196 linear weights in the loop's order from LOAD_EARLY's pinned bytes: the
  same raw copy and in-place cast as load_pinned.py's GPU cast, then bitsandbytes 0.50.2's
  `quantize_4bit` steps (functional.py:917-971) with the NF4 table made once under float32 (bnb's
  `get_4bit_type` builds it with `torch.tensor(list)` under the process default dtype, which transformers
  sets to fp16 on the main thread while it builds the skeleton: a call from the thread in that window made
  fp16 tables, caught by the bit-equality test), one weight at a time (the double quantization's
  statistics are per weight), at most 96 MiB of finished weights waiting on the GPU. load_pinned.py's job
  returns a two-byte stride-0 stand-in after checking the file (through load_early's real-path check:
  transformers opens the cache's snapshot symlinks), key, shape, dtypes, the module (bnb Linear4bit), the
  Params4bit attributes and the device; `Params4bit._quantize` on the stand-in sets what `_quantize` sets
  from the thread's result. Any mismatch drops the speculation; the thread is joined and the device
  synchronized before the pinned chunks are released. Tests: tiny load bit-equal with every weight handed
  over (also through symlinks), mismatch fallback, the steps equal bnb's quantize_4bit under an fp16
  default dtype.
- `SOUP_SOLUTION_LOAD_ONE_SYNC` (load_stream.py, weight_stream.py `sync=`): one synchronize when the load
  returns instead of one per offloaded layer (the GPU tensors are freed in stream order; nothing reads the
  host copies before the load returns). Test: same host copies and step.
- `SOUP_SOLUTION_LOAD_PREPIN` (load_early.py): the embedding's pinned buffer (445 MiB) made by the reading
  thread during the imports; load_stream.py's pinner takes it for a request of exactly that size;
  leftovers unpinned at release. Test: taken, load bit-equal.
- LOAD_EARLY: each opened path resolved once (realpath + stat per tensor before).

## T4 sessions

soup-l2-p0 (0b26618; first step / setup end / wall): base 10.92 / 6.82 / 44.05; early 10.78 / 6.64 /
44.40; wb (warm-up from apply begin) 11.24 / 7.26 / 44.95.

soup-l2-v1 (f91a66d; first step / setup end / wall):
- base 10.65 / 6.60 / 44.90
- e (LOAD_EARLY) 10.57 / 6.43 / 44.62
- es (+ LOAD_SPEC: handed over 0, the symlink path bug) 10.53 / 6.55 / 44.49
- eso (+ LOAD_ONE_SYNC) 10.47 / 6.48 / 44.19
- esomt (+ META_INIT + TRACKER_THREAD) 10.48 / 6.37 / 44.04
Peak 395 MiB everywhere.

soup-l2-v2 (f91a66d; first step / setup end / warm-up length / waited / wall):
- base 11.63 / 7.25 / 5.42 / 0.00 / 46.33
- w1 (one worker) 11.37 / 6.52 / 7.64 / 0.14 / 46.24
- ew1 11.91 / 6.76 / 7.95 / 0.32 / 47.17
- esw1 11.52 / 6.26 / 7.57 / 0.81 / 46.39
- esomw1 11.65 / 6.25 / 7.68 / 0.86 / 46.54

soup-l2-v3 (93574b3, a slow session; first step / setup end / tensor loop / wall):
- base 12.01 / 7.45 / 1.97 / 46.17
- eomt (LOAD_EARLY + LOAD_ONE_SYNC + META_INIT + TRACKER_THREAD) 12.02 / 7.22 / 1.29 / 46.13
- seomt (+ LOAD_SPEC: 196 of 196 handed over, the loop waited 0.00 s) 11.71 / 7.01 / 1.15 / 45.89
- pseomt (+ LOAD_PREPIN) 11.40 / 6.68 / 0.95 / 45.53: first step -0.61 s, wall -0.64 s
- wpseomt (+ one warm-up worker) 11.65 / 6.23 / 0.77 / 45.64 (train waits 0.9 s for the warm-up)
Peak 395 MiB, per-run checks 3, 4, 9 pass.

## The warm-up's variants (do_not_specialize)

The WARMUP_AUDIT (L1, soup-l1-p0) lists 16 compiled variants of 12 kernels, each launched in training.
The four kernels with two variants differ only in constexpr flags or pointer dtypes (rms_norm_rows_fwd
ADD and an fp32/fp16 add input, rope_qk_kernel BACKWARD, rms_norm_rows_bwd ADD_GRAD/SECOND and dtype,
head_chunk_kernel FIRST); every int argument is in one divisible-by-16 class. No variant comes from
specialization on token counts or strides, so `do_not_specialize` would merge nothing; merging the
constexpr variants would mean rewriting those kernels. Costliest compiles under contention:
rms_norm_rows_rebuild 1.50 s, nf4_dequant4 1.18, head_finish 1.02, masked_pair_sum 0.57.

`SOUP_SOLUTION_WARMUP_NICE=<n>` (warmup.py `lower_thread_priority`; allowed by the organizers): the warm-up's
worker threads and the one-build thread raise their own Linux nice value (`os.setpriority` on the thread
id; the C compiler and ptxas they start inherit it); the main thread and the process keep theirs.
