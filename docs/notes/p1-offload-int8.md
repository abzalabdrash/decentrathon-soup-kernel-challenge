# P1: the MLP's (gate|up) crosses PCIe as int8 (branch `claude/p1-offload-int8-8429f6`, from main 8387938)

Organizers' answer 43: a tensor saved only for the backward may be stored in int8 with a scale (lossy)
when the forward, the loss and the metrics do not change and all 10 checks pass.

## 1. Is the (gate|up) traffic on the critical path at all? (ceiling, T4, ABBA 2 rounds + warm run)

Variants of main 8387938 (`a66accd` only adds `--variant-data` to the runner), solution-only runs:
`base` (defaults), `nogu` = `SOUP_SOLUTION_OFFLOAD_MIN_NUMEL=100000000` (the (gate|up) stays on the GPU,
nothing of it crosses PCIe; the layer inputs still go), `pf2` = `SOUP_SOLUTION_OFFLOAD_PREFETCH=2` (the
backward asks for the previous layer's (gate|up) one tensor earlier, at the start of this layer's MLP
backward instead of its post-attention norm).

| rows | variant | wall s (2 runs) | steps 3+ s | nvidia-smi MiB | alloc peak MiB |
|---|---|---|---|---|---|
| public (soup-p1-ceil-pub) | base | 47.14 / 49.79 | 27.37 | 395 | 229.3 |
| | nogu | 49.70 / 48.89 | 27.73 | 1735 | 1546.0 |
| | pf2 | 50.36 / 50.16 | 28.17 | 495 | 278.4 |
| `--select longest` (2,048 tokens every step, seed 157; soup-p1-ceil-long) | base | 75.79 / 77.55 | 52.60 | 453 | 284.5 |
| | nogu | 73.39 / 74.40 | 50.06 | 2373 | 2156.9 |
| | pf2 | 74.59 / 75.15 | 51.17 | 593 | 354.3 |

On the public rows the (gate|up) traffic is hidden (nogu is no faster). On 2,048-token steps it is not
in that session: without it the run is 2.8 s shorter (steps 3+ -2.5 s), and starting the previous
layer's copy one tensor earlier alone gives -1.8 s (at +140 MiB). Session matters: in soup-p1-abba-long2
the card throttled (its first run 75.5 s, the later ones of the same variant 82 s) and the steps were
compute-bound; there the copies were not the bound (section 5).

## 2. What is in the branch (cebfd3e)

`solution/offload_int8.py` (switch `SOUP_SOLUTION_OFFLOAD_INT8`, on by default only in the `soup train`
process, like int8_bwd.py; =1 also under pytest, =0 off):

- The int8 form: per element (i, t) of the gate half one int16, the gate's int8 in the low byte and the
  up's in the high byte, plus one fp32 scale per token and group of 128 features of each half (absmax /
  127, round to nearest). 2 I M bytes + scales = 51.6 % of the half (gate|up) (14.7 MiB instead of 28.4
  at 832 tokens, 36.1 instead of 70.0 at 2,048).
- `gu_quant_kernel` (SWIGLU): computes the MLP's SwiGLU forward exactly as the example kernel does
  (`gate * sigmoid(gate) * up` in fp32, one rounding: hidden bit for bit, test) and in the same pass
  writes the int8 form in place over the gate half of (gate|up) (each element's two bytes, loaded before
  by the same program; no other program touches them). No extra read of (gate|up), no new buffer of its
  size; only the scales are allocated.
- fused_proj.py, MLP forward (a few lines): `offload.offload_early(gate_up)` at the gate|up GEMM returns
  "int8" after reserving the host slot; then `swiglu_quant` replaces the example SwiGLU launch and
  `offload_early(gate_up, wire=...)` starts the D2H of the int8 form (half the bytes) at once. In flight
  the offload holds (gate|up)'s storage, as the half offload does, for half as long.
- The backward brings back one uint8 buffer (half the bytes, half the GPU slot). With the int8 MLP
  backward and `SOUP_SOLUTION_OFFLOAD_INT8_FUSED=1` (default) fused_proj's backward unpacks inside
  `offload_int8.raw(True)` and gets that buffer: `int8_swiglu_bwd_kernel` (INT8_IN) reads the int16
  pairs and scales (rounded to half as the rebuild pass would, so bit for bit the same result, test) and
  writes d_gate / d_up interleaved into the two halves of its own int32 product (no new buffer);
  `quant_t_int8_kernel` (IL) reads that layout. Otherwise (fp16 backward: the first four steps, or the
  last layer, still on the GPU) `gu_dequant_kernel` rebuilds the half tensor.
- The kernels compile in a thread started at the first MLP backward; the first 112 offloads (four steps)
  stay half, the next one waits for the thread: a fixed switch point. `SOUP_SOLUTION_OFFLOAD_INT8_GROUP`,
  `_TILE_M`, `_WARPS` are experiment knobs (defaults 128, 128, 8).

Tests: `solution/test_offload_int8.py` (kernels against torch, SwiGLU bit for bit, all three ways back
under GPU delays, the version guard, the switch point, no record_stream / CPU wait, the fused MLP forward
bit for bit and dL/dx within 2 %, the raw path bit for bit the rebuild path, the int8 backward's memory).
Locally (RTX 5060, WSL): test_offload_int8 + test_int8_bwd + test_offload + test_xoffload +
test_jury_offload_marks 46 passed, test_fused_proj 40 passed.

## 3. Numerics (solution/bench_offload_int8.py on the T4, soup-p1-bench, 047d9a4: the same arithmetic)

Relative error against the fp16 MLP backward with the half offload, one Qwen2.5-1.5B-shaped block
(fused attention with LoRA r=16 on q and v, then the fused MLP, both residual; NF4 weights):

| tokens | path | dL/dx | LoRA A_q, B_q, A_v, B_v |
|---|---|---|---|
| 826 | int8 MLP backward (main today), half offload | 0.0224 | 0.0226 0.0223 0.0239 0.0236 |
| 826 | fp16 backward, int8 offload, group 128 | 0.0049 | 0.0049 0.0049 0.0053 0.0052 |
| 826 | int8 backward + int8 offload, group 128 | 0.0230 | 0.0232 0.0229 0.0244 0.0238 |
| 2048 | int8 MLP backward, half offload | 0.0224 | 0.0226 0.0226 0.0235 0.0231 |
| 2048 | int8 backward + int8 offload, group 128 | 0.0229 | 0.0234 0.0232 0.0247 0.0235 |

Groups of 64 / 256 features: 0.45 / 0.53 % alone, the same 2.3 % combined. The int8 offload adds
~0.5 % in quadrature to the 2.2 % that the int8 MLP backward already puts on the gradients.

**Real layers, real activations** (`python -m solution.bench_offload_int8 --real 2 13 27 --data
data/train.jsonl`, 10da017, RTX 5060 in fp16): Qwen2.5-1.5B loaded in NF4 as Soup loads it, the
training rows through the plain transformers model for each layer's real input, then that layer (its own
NF4 weights and norms, LoRA r=16 on q and v around its own q_proj / v_proj, random A and B) through the
fused path; relative error against the fp16 MLP backward with the half offload:

| layer, rows (tokens) | path | dL/dx | LoRA A_q, B_q, A_v, B_v |
|---|---|---|---|
| 2, rows 0-3 (884) | fp16 backward + int8 offload | 0.0117 | 0.0147 0.0124 0.0160 0.0140 |
| | int8 backward + half offload (main) | 0.0708 | 0.0737 0.0695 0.1013 0.0811 |
| | int8 backward + int8 offload (P1) | 0.0719 | 0.0736 0.0715 0.1012 0.0822 |
| 2, 4 longest of rows 0-63 (2,048) | fp16 backward + int8 offload | 0.0117 | 0.0150 0.0146 0.0105 0.0123 |
| | main | 0.0712 | 0.0775 0.0713 0.0675 0.0724 |
| | P1 | 0.0723 | 0.0812 0.0759 0.0705 0.0769 |
| 13, rows 0-3 (884) | fp16 backward + int8 offload | 0.0056 | 0.0048 0.0057 0.0058 0.0061 |
| | main | 0.0255 | 0.0229 0.0238 0.0243 0.0279 |
| | P1 | 0.0261 | 0.0229 0.0233 0.0249 0.0280 |
| 13 (2,048) | fp16 backward + int8 offload | 0.0055 | 0.0073 0.0063 0.0056 0.0063 |
| | main | 0.0253 | 0.0327 0.0276 0.0244 0.0270 |
| | P1 | 0.0259 | 0.0341 0.0306 0.0240 0.0274 |
| 27, rows 0-3 (884) | fp16 backward + int8 offload | 0.0077 | 0.0074 0.0075 0.0057 0.0076 |
| | main | 0.0391 | 0.0385 0.0398 0.0302 0.0387 |
| | P1 | 0.0398 | 0.0386 0.0398 0.0282 0.0389 |
| 27 (2,048) | fp16 backward + int8 offload | 0.0077 | 0.0083 0.0087 0.0102 0.0079 |
| | main | 0.0394 | 0.0370 0.0425 0.0338 0.0389 |
| | P1 | 0.0402 | 0.0372 0.0427 0.0364 0.0393 |

On real activations the int8 offload alone moves the LoRA gradients by 0.5-1.6 % (layer 2 most: its
gate|up have outlier features), 2-3x the synthetic figure. Main's int8 MLP backward is the larger term,
2.3-10 % (7-10 % in layer 2, against 2.2 % on synthetic data); on top of it P1 changes the error against
fp16 by at most +0.5 points (sum in quadrature). P1 against main directly: 1.4-4.2 %, because the int8
backward's per-token quantization of dL/d(gate|up) rounds differently once its inputs move (each path is
~7 % from fp16 in layer 2, their difference is part of that). The checks' curves (section 6) are where
these errors would show; layer 2's 7-10 % is a property of main's int8 backward worth a look on its own.

## 4. Cost and gain per layer (T4, soup-p1-bench2, launch latency excluded)

Copies (pinned host, one layer): D2H 2.35 -> 1.19 ms at 832 tokens, 5.61 -> 2.91 ms at 2,048; H2D
2.54 -> 1.28 and 5.96 -> 3.10. A separate quantization pass (the first design) cost 0.28 ms at 832 and
0.55 ms at 2,048 per layer at its best tile (128 tokens x 4-8 warps; 64 x 4: 0.32 / 0.75), the rebuild
pass about the same: 28 layers x 64 steps of that is 0.5 s (public) to 1-2 s (2,048) of GPU per run,
which is what the long-row ABBA below paid. Folded into the SwiGLU (cebfd3e), the forward adds only the
int8 write: locally 0.10 ms at 832 and 0.17 ms at 2,048 over the example SwiGLU (T4: TODO).

## 5. ABBA (solution-only runs, same session, 2 rounds ABBA + warm run)

| commit, design | rows | variant | wall s | steps 3+ s | step 0 s | nvidia-smi | alloc |
|---|---|---|---|---|---|---|---|
| 8c9e37b: separate quant into a new buffer, rebuild pass (soup-p1-abba-pub) | public | off | 46.34 / 47.24 | 26.14 | 0.500 | 395 | 229.3 |
| | | i8 | 47.09 / 47.14 | 26.32 | 0.631 | 455 | 239.5 |
| d0d48a0: quant after the down GEMM into hidden's storage (soup-p1-abba-pub2) | public | off | 45.90 / 46.39 | 25.20 | 0.512 | 395 | 229.3 |
| | | i8 fused (fp16 dgate/dup buffer) | 45.29 / 45.60 | 24.25 | 0.651 | 435 | 254.7 |
| | | i8 rebuild pass | 46.19 / 46.14 | 24.80 | 0.628 | 455 | 229.3 |
| d0d48a0 (soup-p1-abba-long2, card throttling) | longest, 2,048 | off | 82.34 / 82.04 | 59.06 | 1.093 | 453 | 284.5 |
| | | i8 fused | 83.18 / 83.36 | 59.84 | 1.115 | 513 | 320.6 |
| | | i8 rebuild pass | 84.53 / 84.49 | 61.05 | 1.077 | 553 | 284.5 |

| 1f44bbc: + d_gate/d_up interleaved as half words over the int32 product (soup-p1-abba-jury) | jury-like (seed 157, train 42000:42256, eval 51800:52056) | off | 50.35 / 52.91 | 30.16 | 0.665 | 453 | 267.6 |
| | | i8 | 55.96 / 56.12 | 34.15 | 0.740 | 453 | 268.6 |
| 1f44bbc (soup-p1-abba-pub3) | public | off | 47.60 / 47.50 | 26.80 | 0.501 | 395 | 229.3 |
| | | i8 | 49.85 / 50.36 | 29.13 | 0.622 | 395 | 230.1 |
| cebfd3e: int8 form made in place by the SwiGLU forward (soup-p1-abba-jury2) | jury-like | off | 50.98 / 51.13 | 30.90 | 0.667 | 453 | 267.6 |
| | | i8 | 53.65 / 53.90 | 33.57 | 0.717 | 453 | 268.6 |

Bytes to the host over a long-row run: 143.5 GiB (off) -> 87.9 GiB (i8). Step 0 is +0.13 s with int8:
the compile thread competes with the first step (62c2094 starts it after the first step).

1f44bbc and cebfd3e lost 2.6-4.4 s to two slow kernels, not to the offload: with d_gate / d_up written as
interleaved half words (stride-2 stores) and per-element scale gathers, int8_swiglu_bwd_kernel took 0.94
ms instead of 0.23 at 832 tokens (2.16 / 0.62 at 2,048) and the transposing quantization 0.36 / 0.09
(own-kernel share 29-30 % against 16-17 %). 1e0eac4 stores one 32-bit word per element and loads one
scale vector per 32-row tile: 0.190 ms from the int8 form against 0.225 from the half one at 832 tokens,
0.545 / 0.625 at 2,048; quantization 0.080 / 0.093 and 0.299 / 0.303 (RTX 5060). The memory of
1f44bbc / cebfd3e is that of off (453 MiB on the jury rows, 395 public): the earlier +40 / +60 MiB are
gone.

**1e0eac4 on jury-like rows (soup-p1-abba-jury3, seed 157, train 42000:42256, eval 51800:52056):**

| variant | wall s | train phase s (begin -> end) | steps 3+ s | median step | step 0 | nvidia-smi | alloc | own |
|---|---|---|---|---|---|---|---|---|
| off | 49.61 / 50.22 | 31.49 / 32.02 | 29.13 | 0.457 | 0.649 | 453 | 267.6 | 16.9 % |
| i8 | 49.56 / 49.35 | 30.90 / 30.85 | 28.13 | 0.434 | 0.658 | 453 | 268.6 | 20.5 % |

The training phase is -0.88 s and steps 3+ -1.00 s (median step -5 %); the wall only -0.46 s because the
i8 runs reached `train begin` 0.4 s later (model load and trainer build: 17.26 / 17.11 s against 16.84 /
16.78; nothing before training reads the switch, so that is load jitter of this session). Inside the
loop, step 4 (the switch point, 113th offload) took 0.96 / 0.86 s against 0.50: it waited ~0.4 s for the
compile threads (mine and int8_bwd's, which compiled the INT8_IN variants too), and steps 1-3 were
+0.25 s from their contention. bbc1001 moves the switch to step 7 (196 offloads) and compiles the INT8_IN
variants in offload_int8's own thread (int8_bwd's thread compiles exactly main's kernels): without the
stall the loop gain is ~-1.3 s on these rows.

**1e0eac4 on public rows (soup-p1-abba-pub5):**

| variant | wall s | steps 3+ s | median step | step 0 | nvidia-smi | alloc | own |
|---|---|---|---|---|---|---|---|
| off | 46.34 / 46.64 | 25.44 | 0.400 | 0.504 | 395 | 229.3 | 17.5 % |
| i8 | 46.14 / 45.69 | 24.76 | 0.377 | 0.507 | 395 | 230.1 | 20.8 % |

Wall -0.58 s, steps 3+ -0.68 s, median step -5.8 %, the same peak. On the public rows the ceiling showed
no PCIe bound, so this gain is mostly the backward's SwiGLU and quantization kernels reading half the
bytes (int8 form) and the shorter D2H/H2D contention with the weight stream.

**bbc1001 on 2,048-token steps (soup-p1-abba-long3, `--select longest`, seed 157):**

| variant | wall s | steps 3+ s | median step | step 0 | nvidia-smi | alloc | own |
|---|---|---|---|---|---|---|---|
| off | 77.93 / 77.52 | 55.27 | 0.895 | 0.982 | 453 | 284.5 | 14.4 % |
| i8 | 76.78 / 76.77 | 54.40 | 0.883 | 1.001 | 453 | 284.5 | 17.4 % |

Wall -0.95 s, steps 3+ -0.88 s, the same peak: where the separate quantization pass lost 1.1 s (d0d48a0),
the SwiGLU-fused form and the int8-form backward gain about a second.

**Summary of the go/no-go (same-session ABBA, i8 against off):** jury-like rows -0.46 s wall / -0.88 s
training phase (1e0eac4, with the 0.4 s switch-step stall that bbc1001 removes), public rows -0.58 s
(1e0eac4), 2,048-token rows -0.95 s (bbc1001); peak equal everywhere (395 public, 453 jury-like and
long); 10 of 10 in the full run.

## 6. Full + jury

**Full, bbc1001** (soup-p1-full, peagverr; int8 on in `soup train`, as the measured process runs it):
**all 10 checks**; base 83.1 s / 14,345 MiB, ours **46.4 s / 395 MiB** (s = 1.79 in a slow session; the
same code without int8 is ~0.5 s slower by the ABBA above), eval +0.010 %, loss curve 0.01 % first step
/ 0.06 % mean, own kernels 20.3 % (gu_quant_kernel, gu_dequant_kernel, quant_t_int8_il_kernel and the
INT8_IN variant of int8_swiglu_bwd_kernel count). T4 `pytest solution`: 455 passed, 5 skipped, 1 failed:
`test_recompute::test_decoder_layer_recompute_bitwise_and_memory[lengths0]`, the order-dependent
memory test that also fails on main (m7, m9 in STATUS; locally on 8387938 too).

**Merged with main 0100e3a** (fd27547; clean) and aligned with bc18d4b's rule (same behaviour when
measured and tested): `SOUP_SOLUTION_OFFLOAD_INT8` now defaults on in every process; `solution/conftest.py`
pins it off together with the int8 backward for the bitwise tests of the half path (test_offload_int8
switches it on). a00105a: locally test_fused_proj, test_int8_bwd, test_jury_offload_marks, test_offload,
test_offload_int8, test_xoffload 86 passed; Soup's 79 with SOUP_APPLY_SOLUTION=1 passed.

**Jury, a00105a** (soup-p1-jury, slerfas; seed 157, train 42000:42256, eval 51800:52056; `--eval-rows
52000:52256` from the brief runs past the dataset's 52,201 rows): **all 10 checks**; base 88.73 / 89.94 s,
ours **49.94 / 49.45 s** (s = 1.80), peak **453 MiB** (m = 31.7; the int8-off variant of the same rows had
453 too in soup-p1-abba-jury3, so this is main's peak on these rows, not the change's), eval +0.016 %,
loss curve 0.00 % first / 0.06 % mean, entropy 0.16-0.20 %, accuracy 0.27-0.30 %, rewind 0.18 %, own
kernels 19.9 %.

**Accuracy round (orchestrator's tightened criteria: |delta eval| <= 0.2 %, mean curves <= 0.5 %, first-step
loss ~0.01 %), all on the merged head 9ded812:**

| run | delta eval | loss first | loss mean | entropy | accuracy | rewind row loss | wall / peak |
|---|---|---|---|---|---|---|---|
| full (soup-p1-full2) | -0.000 % | 0.01 % | 0.07 % | (not printed in full mode) | | | 45.4 s vs 80.1 / 395 MiB, 10 of 10 |
| jury seed 157 (soup-p1-jury, a00105a = same code), solution_1 | +0.016 % | 0.001 % | 0.059 % | 0.160 % | 0.266 % | 0.182 % | 49.94 s / 453 MiB, 10 of 10 |
| solution_2 | | 0.001 % | 0.057 % | 0.198 % | 0.297 % | 0.187 % | 49.45 s |
| base_2 against base_1 (the base's own spread) | | 0.000 % | 0.009 % | 0.006 % | 0.086 % | 0.026 % | |
| jury seed 163 (soup-p1-jury163; train 44000:44256, eval 51400:51656), solution_1 | +0.010 % | 0.001 % | 0.049 % | 0.145 % | 0.364 % | 0.169 % | 49.15 s / 413 MiB, 10 of 10 |
| solution_2 | | 0.001 % | 0.048 % | 0.172 % | 0.298 % | 0.162 % | 49.75 s |
| base_2 against base_1 | | 0.000 % | 0.008 % | 0.009 % | 0.120 % | 0.026 % | |

| jury seed 163, same rows, `SOUP_SOLUTION_OFFLOAD_INT8=0` (soup-p1-jury163off: main's path), solution_1 | -0.003 % | 0.001 % | 0.049 % | 0.177 % | 0.329 % | 0.165 % | 50.34 s / 415 MiB, 10 of 10 |
| solution_2 | | 0.001 % | 0.049 % | 0.161 % | 0.291 % | 0.174 % | 51.35 s |
| base_2 against base_1 | | 0.000 % | 0.008 % | 0.010 % | 0.111 % | 0.029 % | |

Every line is inside the tightened limits (largest: accuracy 0.364 % against 0.5 %), and on the same
rows and seed the int8 offload does not move them: off 0.049 / 0.049 % loss mean, 0.177 / 0.161 %
entropy, 0.329 / 0.291 % accuracy, 0.165 / 0.174 % rewind; on 0.049 / 0.048, 0.145 / 0.172, 0.364 /
0.298, 0.169 / 0.162. The accuracy and entropy gaps (~0.3 % and ~0.16 % against the base's own 0.1 % and
0.01 %) are the solution's level with or without P1.

**Allocation snapshot and waits (diag, public rows, step 10 snapshot, steps 10-11 traced, int8 from step
7; soup-p1-diag-on on slerfas, soup-p1-diag-off = `SOUP_SOLUTION_OFFLOAD_INT8=0` on peagverr; 10da017):**
- Allocations of step 10 with int8 on, by site: the forward's (gate|up) GEMM output (fused_proj.py:1535,
  25.2 MiB x 28: the int8 form is written into it, it is the copy's source); the backward's prefetches
  (offload.py:507) bring back the int8 form, ~13 MiB each, plus the layer inputs; the half rebuild
  (offload_int8.py:155) runs once per step, 25.2 MiB, for the last layer only (still on the GPU at the
  backward's start). The int8 backward writes dgate|dup over its int32 product: no half (gate|up)
  buffer in the backward otherwise.
- Dependency waits of the compute stream (gaps whose next kernel was launched before the gap began,
  tools/trace_waits.py): 71.8 ms off, 32.4 ms on over the two traced steps; before the MLP backward's
  first kernel after the unpack 45.4 -> 20.1 ms, before one of the fp16 GEMMs 15.7 -> 2.4 ms. Copies
  over the two steps: H2D 4,848 -> 4,240 MiB, D2H 1,525 -> 894 MiB. (The traced steps run under the
  profiler, CPU-bound: 5.1-5.2 s of CPU-side gaps, so the absolute waits are smaller than unprofiled.)
- Kernel times across the two sessions are not comparable (the cuBLAS GEMMs ran ~20 % slower on the
  'on' card), but per replaced kernel: int8_swiglu_bwd 17.8 ms (int8 form) against 19.8 (half),
  quant_t 9.6 (il) against 11.4, and the forward's SwiGLU 0.42 ms per call with the int8 form against
  0.15 for the example kernel: the fused kernel runs at ~120 GB/s at its 128 x 8 tile (chosen from the
  quantization-only sweep), so its tile is the next fix (soup-p1-bench3).
- soup-p1-bench3 (T4, 04d6ada), the forward's SwiGLU with the int8 form against the example SwiGLU, ms
  per layer at 704 / 1,440 / 2,048 tokens: example 0.160 / 0.318 / 0.451; tile 128 x 8 (default) 0.397 /
  0.661 / 0.707; 64 x 4 0.293 / 0.798 / 0.861; 64 x 8 0.298 / 0.801 / 0.878; 32 x 4 0.456 / 0.983 /
  1.448; 128 x 4 0.939 / 1.757 / 2.244 (spills). No tile wins everywhere; 128 x 8 stays (best on the long
  steps, where the copies bind). What it adds (+0.24 to +0.34 ms per layer, ~0.3-0.5 s per run) is
  inside the measured ABBA gains; a tile by token count (64 x 4 up to ~900 tokens) would take ~0.1 ms
  per layer off the short steps (values identical, only speed).
- A second compressed prefetch slot: not added. The jury-row peaks are 413-453 MiB already (main's own
  peak on those rows), so an extra 18 KB/token slot (up to 35 MiB at 2,048 tokens) would cross 450 on the
  longest rows; the waits it would hide are the 20 ms left in two steps above.

## 7. State and what is left

Branch `claude/p1-offload-int8-8429f6`, merged with main 0100e3a. Switches: `SOUP_SOLUTION_OFFLOAD_INT8`
(default 1, every process; conftest pins it off for the bitwise tests), `SOUP_SOLUTION_OFFLOAD_INT8_FUSED`
(default 1), experiment knobs `_GROUP` (128), `_TILE_M` (128), `_WARPS` (8), `_START_CALLS` (196),
`_COMPILE_AFTER` (28). solution/__init__.py does not list them yet (the orchestrator's file).

Not done: the T4 tile sweep of the SwiGLU-fused quantization kernel (128 x 8 chosen from the
quantization-only sweep; locally +0.10 / +0.17 ms over the example SwiGLU at 832 / 2,048 tokens); copy
order by nearest consumer; a prefetch window of two with int8 (the half-precision pf2 gave -1.8 s on
2,048-token rows at +140 MiB; with int8 the extra buffer halves, but it would still lift the jury-row
peak above 450).
