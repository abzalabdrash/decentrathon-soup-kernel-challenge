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

## In training

(pending: soup-h1-abba1, soup-h1-bench2)
