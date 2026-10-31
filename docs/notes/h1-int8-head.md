# H1: the streamed head in int8 (branch claude/h1-int8-head-b4a21c)

Organizers' answer 29 allows int8 for the lm_head (forward and backward by the input). `solution/head_int8.py`,
switch `SOUP_SOLUTION_HEAD_INT8` (default: on in the `soup train` process only, as int8_bwd.py; pytest keeps
the fp16 head), wired through `head_stream.py` (`HeadStreamer.start_int8`, `prefetch`, `head`, `restore`,
`finish_setup`). Tests `solution/test_head_int8.py` (21), bench `solution/bench_head_int8.py`.

## What it does

- A second host copy of the tied weight: Q = round(W*c / s_v) int8 with one fp32 scale per vocabulary row
  (223 MiB pinned, next to the fp16 copy, which stays for the exact embedding lookup and for eval/generation).
  Built once per run in a thread started at the end of Soup's setup (pinning, GPU quantization chunk by
  chunk, then the kernels' compilation): 0.3 s (bench, kernels cached) to 1.5 s (cold) on the T4.
- c = |gamma| of the decoder's final RMSNorm weight (`SOUP_SOLUTION_HEAD_INT8_SMOOTH=gamma`, alpha 1): the
  head quantizes h / c per token and the copy holds W * c, which takes the final hidden states' outlier
  channels (absmax/rms ~19 per token, channels 1301, 553, 503, ...) into the weight. The products are the
  same; dL/dh is divided by c at the end.
- Per chunk of 4,096 vocabulary rows (ring of 2 slots, 6.0 MiB each: the int8 rows then their scales):
  `_int_mm(h_q, Q_c^T)` -> int32 logits (the forward in blocks of 512 hidden rows, `SOUP_SOLUTION_HEAD_INT8_FWD_ROWS`,
  so the int32 buffer stays the size of the fp16 head's logits); `head_i8_chunk_kernel` (own): online
  max/sum/entropy/argmax/target as head_stream.py, P = exp(x - m) * s_v quantized per row and chunk, the
  target's row of W_q; `int8_transpose_kernel` (own) into the dead logits buffer (IMMA needs both operands
  contiguous along the contraction, the vocabulary in the backward); `_int_mm(P_q, Q_c)`; `head_i8_acc_kernel`
  (own): acc = acc * exp(m_old - m_new) + G * scale_P; `head_finish_kernel` from head_stream.py.
- Heads 1-2 of the run take the fp16 path (`SOUP_SOLUTION_HEAD_INT8_START=2`); head 3 waits for the build
  thread if it is not done, so which steps run int8 never depends on timing. The first step's loss is the
  fp16 head's.

## Precision (local, real final hidden states of 24-32 public rows, fp16 model, against the fp16 head)

| variant | loss | entropy | accuracy (rel.) | dL/dh rel. error |
|---|---:|---:|---:|---:|
| W8A16 (weight only) | +0.05..+0.07 % | +0.17..+0.19 % | -0.1..-0.3 % | 2.0 % |
| W8A8, no smoothing | +0.48..+0.61 % | +0.23..+0.28 % | -0.8..-1.3 % | 6.7 % |
| W8A8, c = gamma^0.5 | +0.40 % | +0.17 % | -0.09 % | |
| **W8A8, c = gamma (default)** | **+0.07 %** | -0.18 % | 0.00 % | |
| W8A8, calibrated SmoothQuant 0.65 | +0.04 % | -0.11 % | +0.32 % | |

P quantized per row and chunk adds nothing measurable to dL/dh on real data (2.05 % vs 2.03 % for fp16 P
with the same int8 weight); on peaked synthetic rows 0.7 %.

## T4 (bench, real weight, random hidden; head per step, ms)

soup-h1-bench1 (4bc1c25), ring of 2 prefetched right before the head (the worst case for the copies):

| N / labelled | fp16 | int8 | delta | head memory fp16 -> int8 (512-row forward blocks) |
|---|---:|---:|---:|---|
| 512 / 158 | 42.8 | 20.3 | -22.5 | 29.4 -> 24.6 MiB |
| 512 / 308 | 43.0 | 21.7 | -21.3 | 30.7 -> 26.6 |
| 830 / 250 | 43.2 | 30.4 | -12.9 | 32.7 -> 26.1 |
| 830 / 504 | 44.0 | 40.2 | -3.8 | 35.5 -> 31.5 |
| 1440 / 434 | 63.7 | 52.5 | -11.2 | 39.1 -> 33.6 |
| 1440 / 861 | 82.8 | 63.3 | -19.5 | 43.5 -> 39.0 |
| 2048 / 600 | 86.7 | 67.6 | -19.0 | 46.1 -> 42.6 |
| 2048 / 1238 | 110.0 | 80.8 | -29.3 | 51.0 -> 46.4 |

Per kernel (soup-h1-prof1, c9f3232, resident chunks, N 830 / 504): fp16 GEMMs 20.4 ms + head_chunk_kernel
3.2 = 23.6; int8 GEMMs 13.7 + head_i8_chunk_kernel (8 warps) 7.8 + transpose 2.4 + accumulate 1.4 = 25.4.
With the ring the same kernels run 20-60 % longer (fp16 forward GEMM 19.1 vs 12.1 ms): the copies slow the
compute, and the fp16 head moves twice the bytes. NN layout for the backward GEMM: 2x slower than TN + the
transpose, so the transpose stays. Chunk kernel sweep (soup-h1-kern1): 16 warps 5.8-7.4 ms/head against
8.5-12.6 for 8; R rows per program (2-8) and the reduction-free P scale (PBOUND) not better; transpose
128 x 128 with 8 warps 2.47 vs 2.77 ms.

After the kernel changes (2e5e2ac: 16-warp chunk kernel, 128 x 128 transpose; hidden rows shaped like the
final norm's output, gamma * randn, so the smoothing sees what it sees in training), soup-h1-bench3, ring of 2:

| N / labelled | fp16 | int8 | delta | head memory fp16 -> int8 | int8 vs fp16: loss / entropy |
|---|---:|---:|---:|---|---|
| 512 / 158 | 41.3 | 20.3 | -21.0 | 29.8 -> 23.8 MiB | +0.036 % / -0.050 % |
| 512 / 308 | 41.3 | 20.3 | -21.0 | 30.7 -> 27.6 | +0.031 % / -0.050 % |
| 830 / 250 | 41.7 | 23.7 | -17.9 | 32.7 -> 26.7 | -0.007 % / -0.035 % |
| 830 / 504 | 41.9 | 30.4 | -11.6 | 35.4 -> 31.9 | +0.007 % / -0.035 % |
| 1440 / 434 | 50.8 | 43.1 | -7.7 | 39.5 -> 34.4 | -0.014 % / +0.008 % |
| 1440 / 861 | 62.3 | 52.1 | -10.2 | 43.6 -> 39.5 | -0.001 % / +0.008 % |
| 2048 / 600 | 67.0 | 55.7 | -11.3 | 45.3 -> 42.6 | -0.010 % / +0.009 % |
| 2048 / 1238 | 83.9 | 65.2 | -18.8 | 51.7 -> 46.3 | -0.007 % / +0.009 % |

Compute alone (every chunk resident) the int8 head is +1..+2.8 ms slower up to 830 tokens and 0..-5.8 ms faster
above: the gain is the halved PCIe (and the copies' smaller drag on the GEMMs), not the IMMA GEMMs, whose saving the
chunk kernel (int32 logits and the scales: 1.5x the fp16 kernel), the transpose (2.5 ms) and the accumulation
(0.4-2.5 ms) take back. Build in the thread: pinning 0.09 s, quantization 0.21 s, compilation of the three kernels
1.22 s (bench process; 5.6 s inside a training run with five kernels in soup-h1-abba1, hence three kernels and the
switch at head 5).

The side-stream transpose (removed): no faster, and it raced at a head's start (its first transpose waited only for
the copy, not for the compute work still using the reused buffer: a wrong loss at 2048 tokens in soup-h1-bench2).

## In training

soup-h1-abba1 (e50b06b: 8-warp chunk kernel, 64 x 64 transpose, five kernels compiled in the thread at setup end,
switch at head 3; peagverr, ABBA 2 rounds + warm): fp16 47.10 s / steps 3+ 26.32 / step 0 0.503; int8 47.97 /
26.52 / 0.632; peak 395 MiB both; own kernels 17.1 -> 20.2 %. The thread took 5.58 s in training and finished at
step 2 (head 3 waited for it); setup end -> train begin 3.89 -> 4.23 s. No gain with those kernels.

soup-h1-abba2 (2e5e2ac: 16-warp kernel, 128 x 128 transpose, three kernels, switch at head 5; the build with
torch ops on the default stream; ABBA 2 rounds + warm, peagverr):

| variant | wall | steps 3+ | step 0 | setup end -> train begin | train begin -> end | peak | own kernels |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 48.75 | 27.93 | 0.500 | 4.05 / 3.94 | 30.39 | 395 | 16.6 % |
| int8, build at setup end | 48.60 (-0.15) | 27.21 (-0.71) | 0.575 | 4.24 / 4.44 | 29.80 (-0.59) | 395 | 19.0 % |
| int8, build at the first head | 48.47 (-0.28) | 27.22 (-0.70) | 0.599 | 4.05 / 4.08 | 29.87 (-0.52) | 395 | 19.0 % |

The steps gain 0.7 s; the build ate it back: at setup end its quantization (about 1,500 small torch ops, each
waiting for the GIL behind the trainer build, and synchronous copies on the default stream) took 3.4-3.6 s and
delayed train begin by 0.35 s; at the first head it took 2.1 s (ready after 4-5 heads) and cost step 0 +0.1 s.
fe361ac: the quantization is one own kernel (int8_rows_kernel, also used for the hidden rows) per 1,024 rows on a
stream of its own, all asynchronous with one synchronize (bench process: compilation 0.32 s, pinning 0.03,
quantization 0.07 locally), the build at the first head by default.

soup-h1-abba3 (ce85dc8, the async build with int8_rows_kernel; a hot session; ABBA 2 rounds + warm, peagverr):

| variant | wall | steps 3+ | step 0 | train begin -> end | peak | own kernels |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 49.50 | 28.83 | 0.502 | 31.35 | 395 | 16.3 % |
| **int8 (default: build at the first head, kernels compiled in its thread)** | **48.95 (-0.55)** | 28.07 (-0.76) | 0.626 | 30.72 (-0.63) | 395 | 18.6 % |
| int8, kernels compiled in warmup.py's threads (SOUP_SOLUTION_HEAD_INT8_WARM=warmup) | 49.49 (-0.01) | 27.95 (-0.88) | 0.549 | 30.48 | 395 | 18.6 % |

Per round the default is -0.94 and -0.16 s of wall. The build in training: compilation 1.67-1.82 s, pinning 0.12,
quantization 0.13-0.18 (was 0.64-0.70), ready after 4 heads (head 5 never waited); train begin as with fp16.
Compiling in the warm-up moves train begin 0.5-0.9 s later (the warm-up is the startup's critical path): rejected.

Full and jury runs (2e5e2ac with SOUP_SOLUTION_HEAD_INT8_AT=head, i.e. the int8 head as in fe361ac/ce85dc8 apart
from the build's quantization; peagverr):

- soup-h1-full1 (public rows): **all 10 checks**; base 81.2 s / 14,345 MiB, ours 45.8 s / 395 MiB (s 1.771 against a
  fast base, m 36.3), eval +0.020 %, loss first step 0.01 %, mean of all steps 0.24 %; own kernels 19.7 %; pytest on
  the T4 458 passed, 5 skipped; Soup's 79 pass. Mean per-step relative gaps to the base (check 5's measure, limit
  3 %): loss 0.24 %, entropy 0.30 %, accuracy 0.89 % (main 8387938 in soup-m8-full: 0.06 / 0.18 / 0.28 %).
- soup-h1-jury1 (seed 139, rows 34000:34256 / 46000:46256, base/solution/solution/base): **all 10 checks**; base
  90.0 s, ours 50.3 s (s 1.787), peak 435 MiB (reserved 314), eval +0.027 %; per-step gaps of the two solution
  runs: loss 0.23 / 0.22 %, entropy 0.32 / 0.30 %, accuracy 0.68 / 0.70 % (largest single step 2.3 %; base_2 vs
  base_1: 0.01 / 0.01 / 0.10 %). The fp16 head on the same rows and seed (soup-h1-jury1c, ce85dc8 with
  SOUP_SOLUTION_HEAD_INT8=0, another session): all 10 checks, 51.8 s, **peak 435 MiB (reserved 314) as with int8**,
  eval -0.002 %, loss curve 0.05 %: the 435 MiB is these rows', not the int8 head's.

## Accuracy first (orchestrator, 31 Oct ~17:00): rescoring and the loss correction (7b67a32)

Forward-only on real final hidden states (`solution/bench_head_int8_real.py`, 6 steps of 4 public rows, the int8
head against the fp16 streamed head on the same rows; mean |gap| per step, check 5's measure):

| | loss | entropy | accuracy | rewind rows |
|---|---:|---:|---:|---:|
| int8 head as in the runs above | 0.145 % | 0.189 % | 0.923 % (5-16 argmax flips per step, 1-2 % of labelled rows) | 0.410 % |
| + rescoring of near-ties (DELTA 1.0) and the first-order loss correction | **0.045 %** | 0.189 % | **0.041 %** (one flip in 4,287 rows) | **0.092 %** |

So the accuracy gap of the training runs (0.68-0.89 %) was the int8 forward's argmax flips, not the trajectory.
`head_i8_rescore_kernel` (own): the chunk kernel now also keeps each labelled row's int8 runner-up (one more
reduction, labelled rows only); a labelled row whose top two (or top and target) are within DELTA logits takes the
exact fp16 logits of those tokens (fp16 hidden row, fp16 weight rows read by the kernel from the pinned host copy:
cudaHostRegister'ed memory is mapped under UVA; fp32 dot rounded to fp16 as the base's GEMM output; the first id
among equal values as torch.argmax). Thresholds (flips left / accuracy gap): off 0.92 %, 0.25: 0.34 %, 0.5 and
above: 0.04 %. Every labelled row's loss is corrected to first order with the exact logits of its top two and its
target: loss -= sum_k p_k (x8_k - x_k) - (x8_t - x_t) (p from the int8 pass). Entropy and dL/dh untouched. No sync,
no CPU work; switches SOUP_SOLUTION_HEAD_INT8_DELTA (0: off) and SOUP_SOLUTION_HEAD_INT8_LOSS_FIX.

dL/dh: the split the orchestrator asked for (P's positive part quantized, s_v folded in, accumulated with the
online rescaling, W_q[y] of the same int8 weight subtracted after the normalization) is what the head did from
the start; against a plain P - Y quantization emulated with the same int8 logits the relative error to the fp16
head's dL/dh is the same on real hidden states (0.0930 vs 0.0931, 0.0995 vs 0.0996, ...): the 7-10 % error comes
from P computed from int8 logits (and W_q: 2 % with fp16 P), not from the target's -1.

Rules: main bc18d4b's "same behaviour when measured and tested": HEAD_INT8 is on by default in every process (no
argv check); `solution/conftest.py` pins the fp16 head for solution's own tests, test_head_int8.py builds the int8
copy itself. The first five heads (steps 0-4) are the fp16 head while the copy is built (START_HEADS 5, as the
fifth kernel adds compile time): the first step's loss is the fp16 head's. Organizers' answer 12 (wave 2): an int8
copy of lm_head quantized at load and kept in pinned memory next to the fp16 copy is allowed.

Cost (Modal T4 on PCIe x8, h1-prof2, 7b67a32, per head): head_i8_rescore_kernel 0.77 ms at 830 / 504 labelled
and 1.31 ms at 1440 / 861; the chunk kernel 5.6 ms (with the runner-up). soup-h1-abba4 (7b67a32, Kaggle, ABBA 2
rounds): fp16 47.20 s / steps 3+ 25.84, int8 47.30 / 25.71: no wall gain; per step (fp16 / int8, s) 0: 0.505 /
0.601, 1: 0.312 / 0.426, 2: 0.412 / 0.497, 3: 0.475 / 0.607, 4: 0.546 / 0.581, then int8 ahead: steps 0-5 +0.46 s,
steps 6+ -0.38 s (-6.6 ms per step; soup-h1-abba3: +0.28 / -0.91 s, -15.6 ms per step). Steps 0-4 run the fp16
head in both: their loss is the build thread's (compilation of five kernels, 2.2 s, takes the GIL from the main
thread's launches). 73796c4 drops int8_rows_kernel (torch ops on the build's own stream are cheap enough).

Tests: test_head_int8.py 21 (the int8 pass against fp64 int8 math with rescoring and correction off; rescoring
against the fp16 head: equal correctness, loss error halved at least; real hidden states: 3 steps, loss < 0.2 %,
entropy < 0.5 %, accuracy < 0.3 %, rewind < 0.3 %, skipped without the cached checkpoint).

## T4 runs after the accuracy work

Speed (in training):
- Modal h1-abba5 (7b67a32, PCIe x8 host, ABBA 2 rounds): fp16 55.08 s / steps 3+ 37.37; int8 53.73 / 35.47 (-1.35 s
  wall); without the loss correction 53.39 / 35.32; without rescoring and correction 53.43 / 35.30: the two cost
  ~0.15 s of steps on an x8 host (zero-copy reads), less on x16.
- Modal h1-abba6 (73796c4, PCIe x16, ABBA 3 rounds): fp16 48.58 / 27.34; int8 48.17 (-0.41) / 26.95 (-0.39); peak
  417 against 395 MiB: quant_rows' temporaries on the build stream (its own pool, reserved 296 vs 274). Fixed in
  693af66 (preallocated buffers, test_build_allocates_nothing_on_its_stream); not measured on the T4 yet.
- soup-h1-abba4 (7b67a32, Kaggle x16): +0.10 s wall, steps 6+ -0.38 s, steps 0-5 +0.46 s (above).

Accuracy criteria (|delta eval| <= 0.2 %, mean curves <= 0.5 %, first steps on the fp16 head), jury runs with the
rescoring and the loss correction (base/solution/solution/base; curves = mean per-step gap of each solution run to
base_1; rewind = check 5's measure from result.json):

| run | commit | checks | check 6: first step, mean of all steps | check 7: eval | loss | entropy | accuracy | rewind | peak |
|---|---|---|---|---|---:|---:|---:|---:|---|
| soup-h1-jury3, seed 149, rows 26000/44000 | 7b67a32 | 10/10 | 0.00 / 0.08 %, 0.00 / 0.09 % | +0.020 % | 0.084 / 0.087 % | 0.189 / 0.183 % | 0.382 / 0.444 % | 0.273 % | 415 MiB |
| soup-h1-jury4, seed 139, rows 34000/46000 | 73796c4 | 10/10 | 0.01 / 0.08 %, 0.01 / 0.08 % | +0.016 % | 0.077 / 0.079 % | 0.312 / 0.339 % | 0.441 / 0.497 % | 0.328 % | 457 MiB (the bug above; fp16 head on these rows: 435) |
| base_2 vs base_1 (noise) | | | 0.00 / 0.01 % | | 0.008 % | 0.007-0.008 % | 0.070-0.091 % | | |

Before the rescoring and the correction (2e5e2ac): accuracy 0.68-0.89 %, rewind 0.48-0.61 %, loss 0.22-0.24 %.
What remains of the accuracy gap is the trajectory (dL/dh 7-10 % from the fp16 head's, through P of int8 logits),
not flips: at 0.44-0.50 % it sits at the 0.5 % line on seed 139.

## Final runs on d3d3f65 (b024044 only fixes a test)

- soup-h1-full5 (Kaggle): **10/10**; base 86.8 s / 14,345 MiB, ours 48.6 s / 395 MiB (reserved 274: the build-stream
  fix holds), check 6: first step 0.01 %, mean of all steps 0.09 %; check 7: eval +0.022 %; curves loss 0.092 %,
  entropy 0.328 %, accuracy 0.308 %, rewind 0.305 %. pytest on the T4: 456 passed, 3 failed:
  test_weight_stream::test_stream_memory (fails on main m9 too), test_head_int8::test_build_allocates_nothing_on_its_stream
  (too strict in the full suite: fixed in b024044), test_boundary::test_layer_slices_keep_the_lists (also in
  soup-h1-full3; not on main: boundary's cache keys on module ids, and a new ModuleList that reuses a dead module's
  id bumps the generation, so it depends on the allocation history of the tests before it, which this branch's
  conftest fixture changes; to fix in test_boundary.py, not this branch's file).
- Modal h1-jury6 (seed 149, rows 26000/44000): **10/10**; s 1.744, eval +0.014 %, check 6: 0.00 / 0.09 %, curves loss
  0.088 / 0.085 %, entropy 0.186 / 0.163 %, accuracy 0.419 / 0.462 %, rewind 0.307 / 0.286 %; reserved 292-294
  (as 7b67a32 on Kaggle).
- Modal h1-abba7 (PCIe x16, ABBA 3 rounds): fp16 49.41 s / steps 3+ 27.46 / step 0 0.696; **int8 49.97 (+0.56) /
  27.54 / 0.889**; peak 396 both. Steps 0-5 +0.52 s (the build thread), steps 6+ -0.19 s (-3.3 ms per step; ce85dc8
  before the accuracy work: -15.6 ms per step in soup-h1-abba3).

## Where the gain went (orchestrator, ~18:45: diagnose on Modal x16 before any full/jury)

- Head kernels (Modal x16 h1-prof3 / x8 h1-prof4, per head, resident chunks, 830 / 504 and 1440 / 861 labelled):
  fp16 22.5 / 46.3 ms; int8 without the rescoring 22.0 / 38.8; with the rescoring (runner-up in every chunk)
  23.6 / 41.1: the runner-up's second reduction 1.3 / 2.3 ms, the rescore kernel 0.3-0.7 ms, the loss correction
  ~0. 39bba80: the runner-up from each chunk's max plus the second of the chunk that takes over the running top
  (accuracy gap 0.041 % as before; chunk kernel +0.85 / +1.4 ms instead of +1.3 / +2.3).
- In training (Modal x16, ABBA, per step after the switch): h1-abba8 (75558d9) int8 -5.4 ms, without the loss
  correction -3.7, plain (no rescoring, no runner-up) -5.4; h1-abba9 (39bba80) int8 -2.9, plain -1.7. So the
  accuracy machinery was not what ate the gain: the plain int8 head itself gained only 2-5 ms per step in training
  against -17 ms per head in the bench. Steps 0-5: +0.28..+0.39 s (the build thread: compilation 2.1 s in training:
  chunk 0.7, acc 0.3, transpose 0.5, rescore 0.5-0.6; quantization 0.3; pinning 0.03).
- The host side: enqueueing one head took 10.5 ms on this laptop's CPU against 3.7 for the fp16 head (8 launches per
  chunk: two forward GEMMs and two chunk kernels for the 512-row blocks, the transpose, the backward GEMM, the
  accumulation; fused_proj.py's C1 notes: ~50 us of CPU per Triton launch on the training path, the harness's timing
  events included). ad4cf33: the previous chunk's accumulation inside the next chunk kernel (one head_i8_acc_kernel
  after the loop), the forward in one block up to 1024 rows, the views once per head: 5.8 ms at 830 tokens (fp16
  4.0), 7.7 ms at 1440 (fp16 5.5).

- After the launch cut (ad4cf33), Modal ABBAs, per step against fp16:
  h1-abba11 (x16, 3 rounds): wall 48.06 -> 47.80 (-0.26), steps 0-5 +0.41 s, steps 6+ -0.56 s (-9.7 ms per step);
  h1-abba10 (x8 host, copy-bound, not Kaggle-like): wall -1.45, steps 6+ -35 ms per step;
  h1-abba12 (x16, 3 rounds): fp16 46.69; int8 46.98 (+0.29: steps 0-5 +0.41, steps 6+ -6.8 ms per step); build at
  the end of Soup's setup (SOUP_SOLUTION_HEAD_INT8_AT=setup) 46.86 (+0.17: steps 0-5 -0.06, steps 6+ -6.9 ms per
  step, but setup end -> train begin 4.03 against 3.60 s, +0.45).
- The build costs 0.4-0.45 s wherever it goes: compiling the four kernels is ~2.1-2.3 s in training (cold, locally:
  chunk 1.13 s + 0.45 s launcher build, acc 0.04 + 0.34, transpose 0.09 + 0.33, rescore 0.12 + 0.39; the launcher
  builds are gcc subprocesses, the compile proper holds the GIL) and the quantization 0.25-0.35 s; the switch then
  gains 7-10 ms per step on x16 hosts. Net 0..-0.26 s of wall: below the 0.4-0.5 s the orchestrator asked for.
- Host side per head now (laptop CPU, cProfile): `_int_mm` 1.4 ms (76 calls), Triton launches 2.4 ms (79, mostly
  argument specialization of ~40 arguments), copies 0.6 ms; more cuts (fewer arguments, the transpose inside the
  chunk kernel with a 6 MiB buffer of its own) are worth ~0.1 s per run.

## Final ABBA (dcf2d19: this branch + main 33a7023, which holds 834e5ad; Modal, 3 rounds + warm, peak 395-397 MiB all)

Variants: the fp16 streamed head; int8 without the corrections (SOUP_SOLUTION_HEAD_INT8_DELTA=0,
SOUP_SOLUTION_HEAD_INT8_LOSS_FIX=0: no rescoring, no loss correction, no runner-up); int8 with them (the code's
settings). Per step against fp16 (steps 0-5 hold the build; the switch is at head 6).

| host | variant | wall | steps 0-5 | steps 6+ | build in training |
|---|---|---:|---:|---:|---|
| x16 (h1-final-abba2) | fp16 | 49.59 | | | |
| | int8 without corrections | 49.74 (+0.15) | +0.39 | -0.88 s (-15.2 ms per step) | 2.0-2.2 s (compilation 1.6-1.75) |
| | int8 with corrections | 50.72 (+1.13) | +0.53 | -0.03 s (-0.5 ms per step) | 2.8-3.0 s (compilation 2.3-2.5) |
| x8, copy-bound (h1-final-abba, no x16 host in 6 waves) | fp16 | 49.10 | | | |
| | int8 without corrections | 47.09 (-2.01) | +0.25 | -2.06 s (-35.5 ms per step) | |
| | int8 with corrections | 47.38 (-1.72) | +0.35 | -2.08 s (-35.9 ms per step) | |

On the x16 host (Kaggle's T4s are x16) there is no wall gain: the build costs 0.4-0.5 s of steps 0-5 and the
corrected variant's steps did not gain in this session (its run-to-run spread, 29.4-30.5 s of training, is wider than
the effect). On x8 hosts the head is copy-bound and the int8 copy's halved bytes win ~2 s. Verdict unchanged: the
int8 head stays off by default (SOUP_SOLUTION_HEAD_INT8=0); with the orchestrator's and Astra's reading (int32
logits, P's quantization, the transpose and the scaling take back the IMMA saving on short and medium shapes; the
corrections polish the shown metrics at the same dL/dh) it is not worth enabling. The peak fix (build buffers
from the training stream's pool, 693af66) holds: 395-397 MiB in every run.

## Summary (31 Oct ~19:35)

- Code: 75558d9..ad4cf33 on the branch, int8 head off by default (SOUP_SOLUTION_HEAD_INT8=1 turns it on).
- Accuracy: met (full + two juries 10/10 on d3d3f65, curves <= 0.5 %, eval +0.008..+0.022 %; real hidden states:
  accuracy 0.04 %, loss 0.05 %, rewind 0.09 %, entropy 0.19 %).
- Speed: not met. Best ABBA on an x16 host -0.26 s wall (h1-abba11), +0.17..+0.29 s in h1-abba12; the build's
  ~0.4 s (mostly compiling the chunk kernel) against 7-10 ms per step. No-go as the default.

## Summary (31 Oct ~18:40)

- Accuracy criteria met on the final code: full + two juries 10/10, |delta eval| 0.014-0.022 %, every curve
  <= 0.5 % (accuracy 0.31-0.50 %, the closest), first steps on the fp16 head (first step 0.00-0.01 %).
- Speed criterion not met: with the rescoring, the runner-up and the loss correction the head gains 3-7 ms per
  step after the switch, and the build thread (compilation of four kernels, 1.7-2.2 s, sharing the GIL with steps
  0-4) costs +0.3..+0.5 s: wall +0.10 s (Kaggle soup-h1-abba4, 7b67a32) and +0.56 s (Modal x16 h1-abba7, d3d3f65).
  Verdict: no-go as the default; the default is "0" on the branch since its last commit (the code is ready behind
  SOUP_SOLUTION_HEAD_INT8=1; all runs above had it on).
- What would turn it: the build's compilations off the steps (they cost 0.1 s per step for five steps), and the
  runner-up / rescore cost per step (measured 0.8-1.3 ms + the extra reduction; the rest of the drop from -15.6 to
  -3.3 ms per step is unexplained and session-dependent).
