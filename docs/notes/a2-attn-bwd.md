# A2: the attention backward as batched GEMMs (branch claude/e1-exit-attn)

Recommendation of external review 2 (`docs/briefs/astra-answer-2-2026-10-31.md`, section 2): replace
`aten::_efficient_attention_backward` (fmha_cutlassB sm_75, ~0.8 ms per layer at 4 rows of ~200
tokens on the T4) by cuBLAS batched GEMMs and own Triton kernels for the softmax parts.

## What is in the branch

`solution/attn_bwd.py`, switch `SOUP_SOLUTION_ATTN_BWD=math` (default `sdpa` = the original
backward, nothing changes). Wiring: `solution/unpad.py`, three lines in `_VarlenAttention`
(the forward records the step's `Plan`, the backward asks `attn_bwd.backward` first) and one line in
`patch_qwen2_attention` (warm-up and exit log, math mode only). The forward, the attention recompute
of `fused_proj.py` and `_AttnBlock` (SOUP_SOLUTION_ATTN_ONE_FUNCTION, off by default) are untouched.

Per layer: one gather kernel (q, grad_out, k, v token-major -> head-major padded bins), per chunk of
(bin, kv group) entries and per block of query rows five strided-batched GEMMs (S = QK^T fp32 out via
`bmm(out_dtype=float32)`, dV += P^T dO fp32, dP = dO V^T fp32 out, dQ = dS K, dK += dS^T Q fp32) and
two row kernels (causal softmax over each row's own sequence -> P fp16; dS = scale P (dP - sum P dP)
over P), one scatter kernel back to token-major (dK, dV rounded to half once) with zeros for the
alignment tail. GQA: the six query heads of a kv group are consecutive rows of the head-major buffer,
so dK and dV sum them inside the GEMM (no expansion copies, no group sums). Sequences are packed
into bins of LP rows (`layout`: longest row rounded up to NB blocks of 16-multiples, first fit,
longest first): short rows share one square. Query rows go in NB blocks per bin
(SOUP_SOLUTION_ATTN_BWD_BLOCKS, default 2): block t only meets keys 0 .. (t+1) LP / NB - 1, so the
products skip the causal square's upper right part (NB = 2: 3/4 of the square, NB = 4: 5/8). S and
P live in the free tail of fused_proj's shared dequantization buffer (15.75 MiB behind the q|k|v|o
weights the o_proj and q/k/v backward read), as many entries per chunk as fit.
Filter, decided once per step from the row lengths: at least SOUP_SOLUTION_ATTN_BWD_MIN_TOKENS real
tokens (600), the longest row at most SOUP_SOLUTION_ATTN_BWD_MAX_LEN (256), and `work_ratio` (the
math path's score entries over the fmha's causal ones) at most SOUP_SOLUTION_ATTN_BWD_MAX_RATIO;
anything unexpected or an error -> the original backward.

Own kernels: attn_bwd_gather_kernel, attn_bwd_probs_kernel, attn_bwd_dscores_kernel,
attn_bwd_scatter_kernel (check_rules likeness to the example 0.31 / 0.31 / 0.69 / 0.24).

## Numerics

Not bitwise (the brief allows it). Against an fp32 autograd reference (max |err| / max |ref|), random
fp16 q/k/v/dO, 4 rows (T4, below): the original path 2.9e-4 .. 7.9e-4, the math path (NB = 2) 2.5e-4 ..
6.2e-4 (same order,
slightly better: delta from P and dP in fp32 instead of dO . O with O rounded). Tail rows exactly zero.

## Tests (`solution/test_attn_bwd.py`, laptop)

Gradients against the fp32 reference and the original path for four length patterns with and
without the tail (bins 4/3/2/1), the scratch in the shared buffer (held weights untouched, the same
bits as allocated scratch, chunks of 3/3/2 within 1e-3), the allocation cap, the wiring through
`unpad.attention` inside `varlen` (forward bitwise unchanged, one math call; outside the filter the
original backward, dk/dv bitwise), no math call outside `varlen` or with the switch off, guards,
warm-up compile. `test_unpad.py` 30/30 with the hook in place.

## T4: no-go

`solution/bench_attn_bwd.py`, T4 (peagverr, soup-e1-a2b4, f944614, 30 timed calls per entry, medians
of CUDA-event times; one layer's backward alone: q, k, v, grad_out and the forward's out/lse given,
so the attention recompute, which both paths run the same way, is not in the numbers). orig: unpad's
path (k/v expansion, fmha backward, group sums); nb = 1/2/4: the math path with that many query-row
blocks; r2: work_ratio at NB = 2; host: ms of CPU to queue one call (GPU idle); MiB: peak allocated
above the inputs (math: scratch allocated, at most 16 MiB; in training it sits in the shared buffer).

| batch (row lengths) | N | orig ms | nb=1 | nb=2 | nb=4 | r2 | host o / m2 | MiB o / m2 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 208, 204, 196, 188 | 800 | 0.835 | **0.779** | 0.902 | 1.143 | 1.90 | 0.18 / 0.67 | 19.3 / 18.6 |
| 208, 3 x 140 | 640 | **0.733** | 0.757 | 0.873 | 1.076 | 2.95 | 0.17 / 0.68 | 18.7 / 17.6 |
| 256, 252, 244, 236 | 992 | 0.912 | 1.236 | **0.687** | 1.100 | 1.61 | 0.17 / 0.63 | 22.9 / 21.9 |
| 256, 3 x 140 | 688 | **0.501** | 0.821 | 0.668 | 1.088 | 3.16 | 0.18 / 0.63 | 19.2 / 22.7 |
| 320, 316, 308, 300 | 1248 | **0.853** | 1.046 | 0.874 | 1.118 | 1.59 | 0.18 / 0.69 | 30.5 / 30.4 |
| 320, 3 x 140 | 752 | **0.522** | 0.857 | 0.690 | 1.101 | 2.86 | 0.17 / 0.64 | 22.9 / 22.9 |
| 384, 380, 372, 364 | 1504 | **0.862** | 1.542 | 1.218 | 1.111 | 1.57 | 0.19 / 1.06 | 33.4 / 30.3 |
| 384, 3 x 140 | 816 | **0.556** | 1.096 | 0.860 | 1.176 | 3.22 | 0.18 / 0.68 | 23.4 / 29.0 |
| 448, 444, 436, 428 | 1760 | **1.124** | 2.123 | 1.543 | 1.328 | 1.56 | 0.17 / 1.07 | 40.9 / 36.6 |
| 448, 3 x 140 | 880 | **0.666** | 1.120 | 0.854 | 1.163 | 2.32 | 0.18 / 0.69 | 27.9 / 25.4 |
| 512, 508, 500, 492 | 2016 | **1.309** | 2.315 | 1.994 | 2.087 | 1.55 | 0.18 / 1.57 | 44.8 / 39.9 |
| 512, 3 x 140 | 944 | **0.682** | 1.222 | 1.153 | 1.127 | 2.45 | 0.17 / 1.08 | 29.3 / 21.9 |
| 380, 140, 120, 100 | 752 | **0.476** | 0.884 | 0.713 | 1.123 | 2.35 | 0.17 / 0.75 | 23.9 / 20.5 |
| 300, 200, 150, 90 | 752 | **0.494** | 0.825 | 0.732 | 1.130 | 2.87 | 0.17 / 0.77 | 23.1 / 22.8 |
| 250, 230, 180, 120 | 784 | **0.453** | 0.751 | 0.700 | 1.098 | 2.42 | 0.17 / 0.63 | 19.5 / 22.1 |
| 512, 160, 120, 60 | 864 | **0.583** | 1.206 | 1.162 | 1.184 | 2.57 | 0.19 / 1.13 | 27.7 / 21.6 |
| 450, 300, 140, 80 | 976 | **0.668** | 1.630 | 1.305 | 1.170 | 3.26 | 0.17 / 1.13 | 30.3 / 28.3 |

Also measured: the scores GEMM at L = 208 (8 batches of 1248 x 208 x 128) takes 0.082 ms with
`bmm(out_dtype=float32)`, 0.126 ms as a half bmm plus a cast, 0.064 ms as a half bmm alone (so
out_dtype works on sm_75 and is the cheaper way to fp32 scores); the row kernels' warps (1/2/4/8) move
the whole call by under 10 % (2 kept).

Verdict (the brief's rule: stop unless < 0.5 ms per layer at L ~ 208): **no-go, A2 closed.** The best
math variant at 4 x ~200 tokens is 0.78 ms against 0.84; the original is faster on every batch shaped
like the task's data (one long row and shorter ones: 0.45-0.68 ms against 0.70-1.30) and on every even
batch from L = 320 up. The only wins are near-even batches of 4 x 200-256 tokens (-0.06 and -0.22 ms
per layer), which random batches of this data almost never form (2-4 of 64 steps have even a
longest row <= 256 at >= 600 tokens). Why: on the power-limited T4 these strided-batched GEMMs run at
5-7 TFLOPS and cost 0.06-0.08 ms each however small, so five GEMMs per row block plus the four own
launches put a floor of ~0.7 ms under the math path, while the fmha kernel skips the causal half and
all padding (its time follows the sum of L^2 / 2: 0.45 ms at the data's typical batch). Queueing costs
0.6-1.6 ms of CPU per layer against 0.18 (16-44 launches), which would also eat the GPU lag of the
long steps. Nothing here makes the 0.5 ms target reachable without an own tensor-core kernel (review
2, section 2.5: not the plan for today).

The code stays in the branch, off by default (`SOUP_SOLUTION_ATTN_BWD=sdpa`): with the default, the
forward's `taken` and the backward's hook return on their first comparison. Recommendation: do not
merge the `unpad.py` hook and `attn_bwd.py` into main (no gain, more for the jury to read); this note
and the bench numbers are the result. No full run with ATTN_BWD=math was made (the brief asked for it
only after a passing microbench).

## Fallback items after the no-go

(i) Labelled-only backward of the final norm and the last MLP (review 1, last block): not taken. The
last layer's MLP has no LoRA, so its backward is three input-gradient GEMMs plus SwiGLU (~3.2 ms per
step at 830 tokens); rows whose head gradient is zero give zero there, so gathering the labelled rows
would save (1 - labelled share) of that, ~0.1 s per run at best, below one session's noise, and it
needs a gather/scatter through fused_proj's `_MLP`, the offload of gate|up and int8_bwd (C1's files
today). Not worth the risk on the last day.

(ii) `SOUP_SOLUTION_NORM_HALF=1` (review 2, section 4.2; R3's row "float16_copy"), in the branch, off,
not measured on the T4: the orchestrator stopped both fallback items before the ABBA (fused_proj.py
and rmsnorm.py belong to C1 today) and moved me to the save tail (`docs/notes/save-tail.md`). The decoder
layer's input norm and post-attention norm write their output's half copy in the same kernel
(`rmsnorm.rms_norm_rows_fwd_half`), attached to the fp32 output; `_QKVRope` and `_MLP` take it through
`rmsnorm.half_of` instead of `x.to(half)` (fused_proj.py: two lines and the import). The final norm
does not write one (the head gathers and casts its own rows). Bitwise: the copy is the cast's value
(round to nearest even of the same fp32 product), and the new kernel keeps rms_norm_rows_fwd's tile
layout: left to itself Triton gives the half store 8-wide vectors, the tile becomes sizePerThread
[1, 8] instead of [1, 4] and the sum of squares is added in another order (rstd and y then differ in
the last bit); with stride_y16 not specialized the half store stays narrow and rstd/y are the
original kernel's bits (tested: plain and fused-add norms; a tiny 4-bit Qwen2 with LoRA through the
fused path gives equal hidden states, loss and every LoRA gradient with the switch on and off).
`half_of` hands the copy over once (the fp32 output lets go of it, so it lives as long as the
consumer keeps it, as the cast did) and falls back to the cast when the output changed since the
norm (version check), the dtype or shape differs, or no copy exists; `_MLP` keeps its single padded
copy when there is none. Note: test_recompute's decoder-layer memory test fails when it runs after
any tiny-model test in the same process, on main too (order dependence, not this change).
Laptop (RTX 3050 Ti, per norm at 830 / 1,440 rows): norm + cast 157 / 230 us (plain), 199 / 320 us
(with the add) against 110 / 169 and 165 / 250 us: -35..-70 us per norm, 56 norms per step, plus 56
cast launches less. Warm-up: `rms_norm_rows_fwd_half` registered by `patch_qwen2_rmsnorm` (two
classes, as rms_norm_rows_fwd).
