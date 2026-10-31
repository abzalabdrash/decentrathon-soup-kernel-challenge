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
fp16 q/k/v/dO, 4 rows: the original path 3.4e-4 .. 9.6e-4, the math path 2.8e-4 .. 6.3e-4 (same order,
slightly better: delta from P and dP in fp32 instead of dO . O with O rounded). Tail rows exactly zero.

## Tests (`solution/test_attn_bwd.py`, laptop)

Gradients against the fp32 reference and the original path for four length patterns with and
without the tail (bins 4/3/2/1), the scratch in the shared buffer (held weights untouched, the same
bits as allocated scratch, chunks of 3/3/2 within 1e-3), the allocation cap, the wiring through
`unpad.attention` inside `varlen` (forward bitwise unchanged, one math call; outside the filter the
original backward, dk/dv bitwise), no math call outside `varlen` or with the switch off, guards,
warm-up compile. `test_unpad.py` 30/30 with the hook in place.

## T4

(filled below)
