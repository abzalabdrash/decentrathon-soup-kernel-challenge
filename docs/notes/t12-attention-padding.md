# T12: packing and attention (branch claude/t12-attention-de5176, commit d4fce94)

## What changed
- `unpad.varlen_model_forward` pads the packed tokens to a multiple of 16 (`SOUP_SOLUTION_PAD_TOKENS`, default 16, 0/1 = off): tail = last real id repeated, positions 0..pad-1; layers run on N_pad, the tail is cut before the final norm, so `fused_ce.py` is untouched (head/loss/metrics/rewind see N real tokens only). Tail rows get exactly zero gradient (tested).
- `unpad.sequences(spans, n)`: span bounds + the tail as one more sequence; `fused_proj._attention_plan` uses it instead of the "spans tile N" check.
- `unpad.attention()`: `aten._efficient_attention_forward/backward` with cu_seqlens (one launch each way per layer), kv heads expanded inside an autograd.Function and not saved (compact q/k/v + out + lse saved; v from the fused qkv view is copied so qkv is not kept). Forward bitwise equal to per-row SDPA (mem-efficient), backward dq to rounding. `SOUP_SOLUTION_VARLEN_ATTN=0` = per-row loop. Fallback to the per-row loop: CPU, fp32 without autocast, bf16 < sm80, dropout > 0, head_dim not %8 or > 128, mem-efficient SDPA disabled, batch != 1. Unpad's own patched forward also uses it; bad spans -> original forward.
- The kernel has no native GQA (`query.size(2) == key.size(2)` check); stride-0 expanded k/v are accepted (two calls per kv group would avoid the expansion copies, not done).

## T4 measurement (--mode diag, vs d9f60bb diag run 20261030-185520-diag-soup-diag)
- wall 70.1 -> 65.7 s; median step 0.4218 -> 0.3756 s; per-step diff median -62 ms, mean -67 ms; timed loop 28.5 -> 24.3 s.
- trace steps 8-9: align1/align2 GEMMs 229.9 ms/step -> 0; all GEMMs 339 -> 235 ms/step; all kernels 460 -> 353 ms/step; launches 2144 -> 1868/step; fmha 112+112 -> 28+28 per step (bwd 23.1 -> 20.6 ms, fwd 5.0 -> 5.4 ms).
- The wall gain is smaller than the GPU gain: the step still has the 152 ms allocator stall in `_MLP` `torch.empty` (offload record_stream, T13) and copy-engine time (HtoD 193 ms/step).
- memory: step-6 peak 1353 -> 1250 MiB; repeat_kv blocks (2 x 62.9 MiB) gone; per-step peak max 1874 -> 1688 MiB; training max reserved 2012 -> 1824 MiB. Run peak (nvidia-smi 2405) is the setup spike (alloc 2513 during setup_transformers, embed fp32 copy), not training.
- checks 2, 3, 4 (own kernels 10.8% -> 13.9%), 9 pass. eval loss 2.04928 -> 2.04945, first loss equal, mean loss +0.02%.

## After merging origin/main (commit 4744723, --mode solution, no diag; 20261030-194842-solution-soup-t12)
- wall 58.2 s, peak 1,991 MiB (embed16 fix removed the setup spike: this is the training peak; torch max alloc 1,688, reserved 1,824), eval loss 2.04946, own kernels 15.3%, checks 2, 3, 4, 9 pass.
- median step 0.3289 s (d9f60bb full: 0.4157), train loop 23.6 s (30.4), timed loop 21.3 s (28.0).
- wall vs d9f60bb full (57.3 s) is not comparable: this Kaggle session reached `apply begin` at +14.9 s vs +6.3 s (harness imports + CUDA context before our code). apply begin -> exit: 41.3 s vs 49.3 s.
- Merge was clean (embed16.py, warmup.py, kaggle_wsl.sh only); pytest solution 168 pass + the pre-existing warmup failure, Soup 79/79.

## Tests
- `solution/test_unpad.py` (27 tests), tail test in `test_fused_proj.py`; full `pytest solution`: 167 pass, 2 fail that are not from this branch: `test_warmup::test_model_step_compiles_nothing_after_warmup` fails identically on clean HEAD b5259a7 (same missing kernel set), `test_weight_stream::test_stream_memory` passes alone on both (flaky in the full suite, shared GPU). Soup 79/79.
- `solution/bench_unpad.py` (RTX 5060 local): attention bwd per layer 0.55 -> 0.36 ms (65 -> 15 launches) at N=832.

## Notes for integration
- No change needed in `apply()`: switches are read by unpad.py at import (`SOUP_SOLUTION_PAD_TOKENS`, `SOUP_SOLUTION_VARLEN_ATTN`); document them in `__init__.py`'s docstring.
- `import torch.nn.functional as F` in fused_proj.py is now unused by the attention half (left in place to avoid touching T13's import block).
- Merge: my fused_proj diff is the import line, `_AttnPlan` slots, the tiling check in `_attention_plan`, a comment in `_QKVRope.forward`, `_fused_attention`, and the docstring attention bullet.

# T16 part 1 (branch claude/t16-saved-memory, commits 18a04d4 + 0150e5d, from main 9059ea4)

## What changed
- (a) `rmsnorm.norm_source(y)` / `rmsnorm.rebuild(node, dtype)`: a norm output recomputed bit for bit from the input (or fused sum), rstd and weight its autograd node saves anyway; one Triton launch `rms_norm_rows_rebuild` (own kernel; warm-up `rmsnorm.warm_rebuild`, registered with warmup.py by `fused_proj.patch_qwen2_attention_fused` as `fused_proj_x16_rebuild`). `_QKVRope` saves no x16 when its input is such an output (`SOUP_SOLUTION_REBUILD_X16=0` keeps it). Layer 0 keeps x16: its norm input (embedding) needs no grad, so the norm has no node. Reusable for T16 part 2 (the MLP's x16 = normed output of the post-attention AddRMSNorm, output_nr 1).
- (b) In `unpad.varlen_model_forward` the decoder-layer patch returns `(mlp_out, residual)` (`_soup_carry=True`) and the next layer takes `_soup_branch=mlp_out`, adding it inside its input norm kernel (final norm too); `SOUP_SOLUTION_CARRY_RESIDUAL=0` off; any layer that cannot fuse adds the branch itself. mlp_out first so weight_stream's `_LayerExit` (output[0]) sits on the gradient path into the layer's weight rebuilds. Final norm now before the tail slice. Forward bitwise equal to the plain loop; backward differs by fp32 FMA contraction in the fused kernel (as the post-attention fusion already does).
- (c) not done: `_attention_plan` is 38 us/layer locally (~1 ms/step); a safe cache key (submodule ids, adapter flags, dropout objects, hooks, weight/quant-state identity) would cost 10-15 us itself.

## T4 (--mode diag), vs T12 diag d4fce94
- trace steps 8-9 per step: launches 1868 -> 1814; my kernels: -28 half casts (1.0 ms), -28 layer-end adds (2.1 ms), -27 grad sums (2.1 ms), +27 rebuild (0.96 ms), +1 fill => about -4.3 ms GPU/step. GEMMs ran 5-10 % faster in the later session at the same counts (session variance).
- memory: step-6 peak 1250 -> 1189 MiB (x16 group 28 x 2.25 MiB gone, 1 left = layer 0); per-step peak max 1688 -> 1574 MiB; torch max reserved 1824 -> 1704; nvidia-smi peak 1883 MiB.
- median step 0.3756 -> 0.3588 s (torch-mul variant 18a04d4: 0.3515; session noise is bigger than the 2 ms between them). Checks 2, 3, 4 (15.6 %), 9 pass, eval 2.04941.
- the 137 ms `_MLP` `torch.empty` stall (offload, T13) is still the largest idle gap.
- attention backward dq is not bit-reproducible run to run (key-split atomics); per-row SDPA was the same: tests compare _QKVRope alone bitwise.

# T16 part 2 (branch claude/t16-part2: 9a66e99 + merge of main 75658f3 = fc90111)

## What changed
- `_MLP` (only these two lines): `ctx.save_for_backward(x16 if plan.recompute else gate_up)` -> with recompute and `n == padded`, nothing saved and `ctx.source = norm_source(x)`; `(saved,) = ctx.saved_tensors` -> `saved = ctx.saved_tensors[0] if ctx.source is None else rebuild(ctx.source, ctx.plan.half)`. Default offload mode: unchanged (it saves gate|up). Recompute mode: 28 x16 fewer (63 MiB at 767 tokens, ~117 MiB at the worst step); not measured on the T4.
- `_QKVRope.backward`: `torch.mul(g_drop[0], mask[0], out=fp32)` + `addcmul_(g_drop[1], mask[1])` -> `masked_pair_sum_kernel` (own, bitwise, signed zeros too); warm-up registered as `fused_proj_masked_pair_sum` by the attention patch.
- `ctx.source` no longer cleared after the backward (retained graphs).

## Per-layer launch audit (fused path, middle layer, local profile; T4 main trace per step in brackets)
61 per layer fwd+bwd: 18 GEMM, 23 own (nf4 14, rmsnorm 4, rope 2, rebuild 1, swiglu 2), fmha 2, 2 weight-stream memcpy, and the small ones: x16 casts 2 [58/step 2.3 ms], strided copies 6 = k/v expansion 4 + compact v + grad_v copy [168/step 3.5 ms], dk/dv group sums 2 [56/step 1.6 ms], LoRA dropout draw + redraw 2 [56/step 5.3 ms: torch's generic kernel because the input is expanded (2, N, D)], mul + addcmul 2 [54/step 4.5 ms] -> now 1 own kernel, adapter cast 1 multi-tensor [28/step 0.3 ms], fmha dq workspace zero-fill 1 [28/step 1.8 ms, inherent to the key-split backward]. No fp32 adds left from the residual carry.

## T4 --mode diag fc90111 vs main diag ba8c1b3 (different sessions)
- trace steps 8-9: kernel time 334.6 -> 322.0 ms/step (mine: -4.47 + 1.71 = -2.8 ms; the rest is T13b's gate|up split GEMMs); launches 1862 -> 1891 (mine -27; split +56).
- median step 0.3534 -> 0.3871 s, min step 0.2912 -> 0.3534: this session's CPU was slower (GPU idle 11.9 % -> 19.1 %, idle at unchanged ops like _QKVRope fwd 3.2 -> 7.6 ms, AddRMSNorm 2.5 -> 8.5 ms per two steps). The new kernel adds ~14 us CPU per call with the harness's Triton timing wrapper (local), +0.4 ms/step.
- worst-step peak allocated 1091 -> 1065 MiB (T13b split), nvidia-smi peak 1383 -> 1363 MiB, reserved 1204 -> 1184; eval 2.04936; checks 2, 3, 4 (16.7 %), 9 pass.

## Not done (need a decision)
- LoRA dropout on a contiguous input (vectorized kernel, ~-3.5 ms/step) changes the masks' random stream (same distribution).
- k/v head expansion + compact v as one own copy kernel each way (~-1.4 ms/step, 56 launches less, exact).
- the attention's x16 from the input norm kernel (~-0.9 ms/step; a second norm kernel variant and a side channel).

# T17 CPU side (branch claude/t17-cpu, 03751ba, from db96cf2 = main + T16p2)

## Measured CPU costs (laptop, harness's Triton wrapper installed, one middle layer, GPU kept busy)
- one layer fwd+bwd: 5.4 ms of CPU (x28 = ~150 ms/step on this CPU; Kaggle's is slower).
- a Triton launch of solution/: ~23 us bare -> ~50 us with the harness wrapper (2 CUDA events + record ~14 us, bookkeeping); nf4._fast_path 13 us more per dequant; a cuBLAS(Lt) GEMM ~35 us (17 per layer, inherent); torch small ops 2-10 us; get/set_rng_state < 1 us; offload pack ~2-7 us per saved tensor; plans: _attention_plan 38 + _mlp_plan 19 us (mostly Params4bit attribute reads through bnb's __torch_function__ and nn.Module.__getattr__).
- per layer: 24 Triton launches (nf4 14), 17 GEMMs, ~40 other aten ops.

## Changes (SOUP_SOLUTION_CPU_DIET, default on; 0 = previous path)
- `_weights`: q/k/v in one launch (`nf4_dequant3_kernel`, own; NF4's per-block math, bitwise), single weights straight to `nf4._launch`, `_fast_path` verdict cached per layer (packed storage, quant state, dtype, device). 14 -> 10 dequant launches per layer.
- `_AdapterHalves`: the 4 LoRA half copies per layer kept and refreshed for all layers in one `_foreach_copy_` per step (28 -> 1 launch); in-place refresh after the optimizer, so an old graph used later raises autograd's in-place error.
- tests: dequant3 bitwise, _weights equal on/off, adapter cache refresh semantics, attention bitwise fwd on/off, warm-up.

## T4
- variants (same session, base = CPU_DIET=0, two rounds interleaved): median 0.3678/0.3933 vs 0.3970/0.3936, steps3+ 23.24/24.72 vs 24.85/24.39 (session drifted after base_1); fastest step 0.2875/0.2863 -> 0.2659/0.2672 (-20 ms), lower deciles -5..-19 ms, upper deciles equal. nf4 GPU time 2246 -> 2128 ms over the run. => CPU-bound (short) steps gain; most steps are GPU-bound.
- diag (diet on): GPU idle 12.3 % of steps 8-9 (main diag 11.9 %, other session); idle at our Functions: _QKVRope fwd+bwd 8.0 ms / two steps (main 12.5), _MLP 1.9 (3.1), AddRMSNorm 9.0; the rest is step-end and between-step: "no CPU op" 29.5, .item 8.1, amp scale 3.5, copy_/add/fill/sum ~15.
- checks 2, 3, 4 (14.1-14.4 %), 9 pass in all runs; eval 2.0494.

## Not done / next
- q/k/v + o in one dequant each way (-2 launches/layer, ~2.8 ms CPU/step, needs a buffer generation check across two Functions).
- plan checks: cache _plain_4bit verdicts (~1 ms/step).
- the idle is now mostly outside the layers (step end: .item, AMP, optimizer, Python between steps).
