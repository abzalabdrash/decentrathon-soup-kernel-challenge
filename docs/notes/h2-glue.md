# H2: torch glue of the step in own kernels (branch claude/h2-glue, from main 5725d84)

Task (orchestrator, 31 Oct ~20:20): the 1,304 torch elementwise launches per step of main's profile (soup-m10-prof,
`docs/briefs/astra-brief-5b-profile-2026-10-31.md`: 24.8 ms GPU + the launches' CPU), at the same memory and without
touching dropout's RNG (answer 47), the GQA expansions / group sums or the attention's `FillFunctor<uint8>`.

## Where the launches come from

The profile's 28 floor, 85 clamp, 85 fp32 binary, 29 fp32 add, 28 of the 56 half reductions and part of the casts are
`int8_bwd.mlp_backward`'s glue, about 26 torch launches per layer and step (x 28 layers):
- `_quant_rows(grad16)`: abs, amax, cast, compare, clamp_min, `127.0 / t`, where, zeros_like, cast, mul_, add_(0.5),
  floor, clamp_, cast to int8, `amax / 127.0`;
- `torch.maximum(scale_g, scale_u)` and `_inverse(scale_gu * 127.0)` (mul, compare, clamp_min, division, where,
  zeros_like) every step, although both weights are frozen;
- step 7: `acc_x.float().mul_((amax / 127.0)[:, None]).mul_(scale_gu[None, :])`.

A diagnostic run with stacks (Modal h2-diag1, `SOUP_SOLUTION_DIAG_STACK=1`, new in diag.py: record_shapes and
with_stack on the traced steps, `diag_stacks.log` with the counted ops by input shape) attributes the rest:
`direct_copy` (224 per step) is the GQA expansion (`clone` of (1, N, 2, 6, 128), ~6 per layer) plus the copies of
grad_v into the q|k|v gradient (`as_strided(...).copy_(grad_v)`) and a (1, N, 2, 128) clone, 1 per layer each; the
group sums of (1, N, 2, 6, 128) are the 56 half reductions left; `float16_copy` (58) is two (N, 1536) casts per layer
(fp32 <-> fp16 between our norms and the GEMMs); the `zero_` of a 15.7 MB uint8 buffer per layer is the attention's
fill. The large `copy_` entries are PCIe transfers (weight streaming, offload, the head ring), not kernels.

## What the branch does (SOUP_SOLUTION_GLUE, on by default; =0: the torch ops as before)

`solution/int8_glue.py`, three hooks in `int8_bwd.mlp_backward` and one line in its compile thread's `_warm`:
- `quant_rows_kernel` (own): `_quant_rows` in one launch, bit for bit: the absmax of the half row, inv =
  (1 / absmax) * 127 (torch's `127.0 / t` is `t.reciprocal() * 127.0`), 0 for a zero row, q = floor(x * inv + 0.5)
  clamped to [-127, 127] without fused multiply-adds, scale = absmax * fp32(1 / 127) (torch's division by a scalar);
  a row with a nan gets a nan scale (Triton's max drops nans: counted explicitly), an inf an inf scale, so the
  backward stays non-finite for GradScaler as before;
- `dequant_rows_cols_kernel` (own): the input gradient (float(acc) * (amax / 127)) * scale_gu in one launch;
- `gate_up_scales`: max(scale_g, scale_u) and its inverse kept on the gate module while the per-weight scales are
  the same tensors.
Per layer and step 26 torch launches become 2 own ones (~670 launches less per step).

Tests `solution/test_int8_glue.py` (13): both kernels equal the torch ops bit for bit (fp16 and bf16 rows, 1-1440
rows, zero rows), non-finite rows keep non-finite scales, the scales are kept and recomputed when their tensors
change, and the whole int8 MLP backward gives the same dL/dx bit for bit with the glue on and off. With
test_int8_bwd.py and test_offload_int8.py: 31 passed locally.

## T4 (Modal, x16 host; Modal numbers are not jury numbers)

h2-abba1 (c7ba0f4, ABBA 3 rounds + warm): glue off 49.02 s wall / steps 3+ 28.25; **on 48.57 (-0.45) / 27.74**;
steps 0-5 equal (the int8 backward starts after four steps), steps 6+ -0.54 s (-9.4 ms per step); peak 403 MiB and
reserved 274 in both; own kernels 19.4 -> 20.3 %.

## After main faaa190 (P2's early release; merged at 6643497, only diag.py conflicted)

- e4a866c: the MLP backward's gradient cast folded in too: fused_proj's int8 branch hands int8_bwd the gradient
  as it comes (fp32 on the T4) and `quant_rows_kernel` rounds it to the half type and pads it to the GEMMs'
  multiple of 8 itself (bit for bit as `_padded_half` then `_quant_rows`; test_quant_rows_from_fp32_padded, the
  MLP backward test at 829 tokens): one cast (or empty + copy + zero when the tokens are not a multiple of 8)
  less per layer. The int8 backward's compile thread compiles both variants (fp32 rounded, half as is).
- Local: test_int8_glue + test_int8_bwd + test_offload_int8 37 passed; test_fused_proj + test_model_path 43.

## What is left of the glue (h2-diag1's stacks, with the glue on)

- The GQA expansions (`clone` of (1, N, 2, 6, 128), ~6 per layer and step) and their group sums (2 per layer),
  the copy of grad_v into the q|k|v gradient (`as_strided(...).copy_(grad_v)`, 1 per layer), a (1, N, 2, 128)
  clone (1 per layer), dropout's masks, the attention's 15.7 MB `zero_`: off limits for this task.
- One (N, 1536) fp32 -> half cast per layer left: the attention block's `grad.to(half)` before the o_proj input
  gradient GEMM (fused_proj.py:1117); it would go into the producer of that gradient (the post-attention norm's
  backward in rmsnorm.py writing a half copy), another owner's kernel.
- 113 scalar `add_` per step: AdamW's step counters of the 112 LoRA tensors (the optimizer: not ours to change).
- No floor left; one clamp per step (the loss path's).
