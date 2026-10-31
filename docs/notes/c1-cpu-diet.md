# C1: the CPU side of the training step (branch `claude/c1-cpu-diet-0b2063`)

Status (handover, usage limit): branch head below; defaults = all C1 switches on except ATTN_ONE_FUNCTION.

## Step 1: what the CPU does per step (T4, main a6c3fe3 + profiler, session soup-c1-prof, 3ace6b7)

`SOUP_SOLUTION_CPU_PROFILE=1` (`solution/cpu_profile.py`, off by default, wired in `apply()` next to
the diag switch): per step tokens / wall / issue / main-thread and process CPU; cProfile of steps 5-8
(all threads: Python 3.13 profiles every thread); a 1 ms stack sampler over steps 9-12 (main thread
and autograd's device thread, where the backward's Python runs); the synchronizing CUDA calls of step
13 with their call sites; a CUDA-event timeline (where the GPU waits for the CPU) for the other steps.

Counts per step (cProfile, exact): 681 Triton launches of ours (each also pays the harness's two
timed events and two `torch.cuda.current_stream()` chains), 2,457 `Event.record`, 736 `wait_event`,
366 `torch.cuda.stream()` contexts, 2,406 `torch.cuda.current_stream()` calls (1,362 the harness's),
2,265 `Params4bit.__torch_function__` calls (~1,090 in our dequantizations and guards, the rest the
step boundary's walks over the model, task B1), 395 autograd Function round trips, 369 saved-tensor
packs and 503 unpacks, 230 `.item()` (AdamW's step counters, B1). GPU side (diag trace, steps 8-9):
2,090 kernel launches per step (681 Triton, ~1,270 torch, ~140 cuBLAS), 422 memcpy, 107 memset.
Launches by node: AddRMSNorm backward (the attention recompute runs inside it) 166 Triton + 84
copies, `_MLP` forward/backward 112 Triton each, `_QKVRope` backward 112 + 83 GEMMs, varlen
attention 56 head-expansion copies each way plus 56 group sums.

Host synchronizations of a step (sync debug mode, step 13: 22): Soup's rewind
`row_losses_from_token_losses` 8, TRL metrics in trainer_patch 4, transformers' nan check 2,
`GradScaler` 1, ours 7 (unpad.pack_batch 3, the embedding lookup's ids to host 1, `_cu_seqlens` 1,
head_stream's labelled count and `float(num_items)` 2, task B1).

Sampled main thread (steps 9-12, ~400 ms/step): 165 ms waiting in `_engine_run_backward` (the
backward's Python runs on autograd's thread), 41 ms in `Event.record` (the launch queue is full
there: the forward runs ahead of the GPU), 25 ms `_QKVRope.forward`, 22 ms `nf4._launch`, 18 ms
`_MLP.forward`, 21 ms trainer_patch's metrics sync (B1).

## Step 2: what changed (all bit-identical, each behind a switch, defaults on)

- `SOUP_SOLUTION_DEQUANT_REUSE=1` (fused_proj.py, under the CPU diet): q, k, v and o dequantized in
  one launch (`nf4_dequant4_kernel`, replaces dequant3 in the warm-up); the o_proj forward, the
  recompute's o_proj, the o_proj backward and the q/k/v backward read what the shared buffer holds
  (`_HELD`: every `_buffer()` call forgets it, so any other writer invalidates); any other repeat of
  what the buffer holds is skipped too (the last layer's down_proj into the backward). 4 Triton
  launches per layer less (681 -> 569 per step) and 3 dequantizations of GPU work. The kernel's
  operands kept per packed weight (`_nf4_args`, keyed by storage pointer and the ids of the quant
  state's tensors): no Params4bit attribute reads and no views per launch. Guards: imports cached,
  Params4bit getters read with the function override off, bitsandbytes' small-M verdict memoized.
  Dropout RNG state through the generator itself.
- `SOUP_SOLUTION_FEWER_VIEWS=1` (fused_proj.py, rope.py): rope launched from strides
  (`rope._launch_strided`: the same kernel arguments as on the head-major views) and the recompute's
  q, k, v built in the attention kernel's token-major layout directly (same pointers, sizes and
  strides as `kernel_operands`); ~20 view ops less per layer and step. The transposes of the shared
  buffer's weight views and the adapter halves' views are made once (kept with the buffers); a
  layer's q/k/v/o operands are checked once per plan (8.2 -> 1.9 us per call on the laptop).
- `SOUP_SOLUTION_ATTN_ONE_FUNCTION` (fused_proj.py, **off by default**): a recomputing layer's
  `_QKVRope`, varlen attention and o_proj as one autograd node (`_AttnBlock`), the same ops in the
  same order; tested bitwise, faster on short steps, but +4.2 MiB allocated peak (see ABBA 2).
- `SOUP_SOLUTION_MLP_ADDMM_INPLACE=1` (fused_proj.py): the MLP input gradient's second GEMM adds into
  the first's fp32 result in place (torch's out-of-place addmm copied it first): 28 D2D copies and
  one N x 1536 fp32 buffer per layer less.
- `SOUP_SOLUTION_LAYER_EXIT_HOOK=1` (weight_stream.py): the layer's backward marker as a tensor hook
  instead of the `_LayerExit` identity node (autograd runs it at the same point).
- `SOUP_SOLUTION_FAST_STREAMS=1` (weight_stream.py, head_stream.py, offload.py): side streams
  selected by torch's C setter instead of `torch.cuda.stream()` (5.3 -> 0.3 us on the laptop), the
  current stream read as its id triple and its Stream object reused (1.8 -> 0.06 us), one reused
  event for the offloader's side-stream waits.
- `SOUP_SOLUTION_STREAM_TIMING=0`: the attention group's and the head's copies and waits untimed (no
  timing events, no queries); `=1` gives the old log lines.
- head_stream.py: the chunk loop with pre-built slot, host and logits views.

Tests (laptop, fp16 layers): `test_fused_proj.py` (new: one launch for q/k/v/o and the reuse rules;
one node vs three bitwise with dropout 0 and 0.05; in-place accumulate bitwise in both MLP modes),
`test_weight_stream.py` (new: untimed by default, the same result), `test_weight_ring.py`,
`test_head_stream.py` (timed and untimed), `test_recompute.py`, `test_unpad.py`, `test_offload.py`;
Soup's 79 with apply().

## Step 3: T4

**Where the GPU waits** (timeline, soup-c1-prof2, 9e72037 = the first half of the cuts; mean per step
size, outside the profiled steps): steps under 600 tokens are CPU-bound in both passes (the GPU's
lag stays at 0-3 ms: 4.0 ms of CPU per layer forward, 4.2 ms backward); 600-900 and longer steps are
GPU-bound in both passes (lag grows to 20 / 44 ms). The head is PCIe-bound (lag 30 ms at its end),
drained by the metrics sync before the backward (B1). Main's path in the same session, short steps:
forward 4.2 ms/layer, backward 5.1 ms/layer, step 332 ms; with the cuts 296 ms.

**ABBA 1** (peagverr, soup-c1-abba1, 9e72037: DEQUANT_REUSE + FAST_STREAMS + STREAM_TIMING=0 against
the same commit with them off = main's path; 2 rounds + warm run):

| variant | wall s | steps 3+ s | median | <600 tok ms | 600-900 ms | >=900 ms | nvidia-smi | reserved | eval |
|---|---|---|---|---|---|---|---|---|---|
| main path | 45.69 / 45.59 | 27.93 / 27.60 | 0.432 | 331.8 | 398.3 | 600.0 | 401 | 272 | 2.04937 / 2.04938 |
| C1 (9e72037) | 45.04 / 45.14 | 27.19 / 27.12 | 0.426 | 304.6 | 392.3 | 594.6 | **381** | 252 | 2.04935 / 2.04940 |

Per step C1 - main: -9.8 ms mean (faster in 58 of 62 steps): -28 ms on steps under 600 tokens,
-6 ms on 600-900, -5 ms on longer ones. Wall -0.55 s, steps 3+ -0.61 s. Allocated peak the same
(234.6 MiB); reserved one 20 MiB segment less (29 segment allocations instead of 30: the shared
dequantization buffer now first grows to the q|k|v|o size), so nvidia-smi 401 -> 381 MiB; seen in
both sessions (prof2 too). Checks 3, 4, 9 pass in every run (own kernels 12.9 %). T4 pytest of
9e72037: 342 passed, 1 failed (the new reuse test assumed an empty buffer; fixed in b951367).

**ABBA 2** (peagverr, soup-c1-abba2, b951367 = + one-node attention, in-place accumulate, hook marker;
a hot session, main's second run 1.6 s slower than its first):

| variant | wall s | steps 3+ s | median | <600 tok ms | 600-900 ms | >=900 ms | nvidia-smi | reserved | alloc peak |
|---|---|---|---|---|---|---|---|---|---|
| main path | 44.32 / 45.88 | 25.46 / 27.30 | 0.410 | 345.9 | 379.1 | 550.5 | 401 | 272 | 234.6 |
| C1 (b951367) | 43.20 / 44.83 | 24.68 / 26.12 | 0.410 | 296.8 | 368.1 | 545.2 | 401 | 272 | **238.8** |

Per step C1 - main: -15.7 ms mean (faster in 58 of 62): -47 ms under 600 tokens, -11 ms at
600-900, -5 ms above. Wall -1.08 s, steps 3+ -0.98 s. But the allocated peak rose by 4.2 MiB and the
20 MiB segment of ABBA 1 came back: autograd keeps a node's incoming gradient until the node returns,
so in the one-node attention the o_proj output's gradient (N x 1536 halves: 4.2 MiB at the
1,440-token step) lives through the attention and q/k/v backward. So `SOUP_SOLUTION_ATTN_ONE_FUNCTION`
is off by default from bc35c89 (memory first), and its view savings moved to the three nodes
(`SOUP_SOLUTION_FEWER_VIEWS=1`). Checks 3, 4, 9 pass in all five runs.

**Memory attribution** (namedesign, soup-c1-mem1, 4775b01, one run each): allocated peak 238.8 MiB with
the one-node attention (default in 4775b01), 238.8 with MLP_ADDMM_INPLACE=0, 238.8 with
LAYER_EXIT_HOOK=0, **234.6 with ATTN_ONE_FUNCTION=0** (cause confirmed); reserved 272 / nvidia-smi 401 in
all four, so the 252 / 381 of 9e72037 also depends on allocation order of a later change (not found).

**In flight / left:**
- peagverr kernel `soup-c1-abba4` (bc35c89 = final defaults): ABBA main path vs c1 + a profiled c1 run
  + T4 `pytest solution`; fetch with `wsl -d Ubuntu -- bash tools/kaggle_wsl.sh 2 --attach --slug
  soup-c1-abba4 --mode variants --commit bc35c89`; it shows wall/steps3+/short steps/peak of the final
  defaults and the per-layer timeline.
- left: find which change after 9e72037 lost the 20 MiB reserved segment (try FEWER_VIEWS=0 and
  DEQUANT_REUSE variants on Modal), then fill this section and the report.

## Session 2 (31 Oct afternoon, branch `claude/c1-cpu-diet-2` from main ce27a54)

### 1. soup-c1-abba4 (peagverr, bc35c89 = the C1 defaults before B1, against the same commit with them off)

Order warm_0 (main path), main_1, c1_1, prof_1, prof_2, c1_2, main_2 (prof = c1 + `SOUP_SOLUTION_CPU_PROFILE=1`).
Wall: main 43.75 / 43.80 s, c1 42.65 / 42.75 (-1.08 s); steps 3+ 25.50 / 25.41 against 24.61 / 24.53
(-0.89 s). Peak 401 MiB, reserved 272, alloc 234.6 in all seven runs (30 segment allocations each).
Checks 3, 4, 9 pass in all; own kernels 13.9-14.0 % (c1), 14.8-15.1 % (main path). T4 pytest:
**349 passed, 5 skipped, 0 failed** (no failing test in this session).

Per step c1 - main (mean of the two runs each, by the step's token count; 62 timed steps):

| tokens | steps | mean diff | median diff |
|---|---|---|---|
| < 400 | 3 | -52.3 ms | -52.2 |
| 400-500 | 3 | -53.7 ms | -55.0 |
| 500-600 | 6 | -31.8 ms | -32.0 |
| 600-900 | 30 | -8.9 ms | -7.3 |
| >= 900 | 20 | -5.5 ms | -5.5 |

Means per bucket: < 600 tokens 331.2 / 329.5 ms (main) against 288.2 / 287.6 (c1); 600-900 368.6 /
367.7 against 359.5 / 358.9; >= 900 523.6 / 521.4 against 518.4 / 515.5.

Profiled run (prof_1, the timeline outside the profile window, mean per step size): steps under 600
tokens issue the forward at 3.55 ms of CPU per layer with the GPU 0.6-3.1 ms behind (CPU-bound), the
head 110.7 -> 120.4 ms CPU while the GPU ends it at 149.9 (PCIe-bound, 29 ms lag), the backward at
4.0 ms of CPU per layer with the lag growing 0 -> 7.3 ms (the GPU catches up and falls behind at the
end), optimizer at 282.9 ms, step end 288.2. Steps of 600-900 tokens: forward 3.6 ms CPU against 4.7
ms GPU per layer, backward 4.0 against ~5.7 (GPU-bound, lag 29 ms at the head, 50 ms at the end).
Against prof2 (9e72037): forward 4.0 -> 3.55, backward 4.2 -> 4.0 ms per layer. The cProfile part
(pre-B1) still shows the FLOP counter's walk (`floating_point_ops` -> `num_parameters`, 31,942
`named_modules` calls per step) and `model.train()` over 933 modules: both are B1's boundary.py fixes,
in main since 1be9e55, so not present in main's step any more.

**The 20 MiB reserved segment** (soup-c1-mem2, peagverr, ce27a54 = main with B1, one run each,
no pytest): default, `FEWER_VIEWS=0`, the three switches added after 9e72037 off
(`FEWER_VIEWS=0, MLP_ADDMM_INPLACE=0, LAYER_EXIT_HOOK=0`) and `ATTN_ONE_FUNCTION=1`: nvidia-smi
**371 MiB, reserved 250, 28 segment allocations in all four**; alloc 225.5 MiB (one node 229.7). So
with B1's single-thread backward (one cuBLAS workspace less: alloc -9.1, reserved -22) the page is
gone whatever the C1 switches: before B1, 9e72037 had 252 against 272 and bc35c89 272; after B1 every
variant has 250. Reserved - allocated is now 24.5 MiB (one 20 MiB page of the large pool plus the small
pool); 230 MiB reserved would need the fragmentation at the peak under 4.5 MiB, which no C1 switch
changes. Nothing to recover here; 371 MiB is main's number. (The single runs' walls rise with their
position in the session: 41.6, 41.6, 43.6, 44.3 s; not a comparison.)

### 2. Two nodes instead of one (`SOUP_SOLUTION_ATTN_TWO_NODES=1`, fused_proj.py, on by default)

Why one node cost +4.2 MiB: autograd holds a node's incoming gradient until the node returns, and
`_AttnBlock`'s incoming gradient is the o_proj output's (N x 1536 half), alive through the attention
backward, the layer's peak. `_QKVAttn` is `_QKVRope` + the variable-length attention as one node
(the reshape to the o_proj input a view of its output); the o_proj stays `_Linear4bitHalf`. Its
backward consumes the output's gradient first, as with three nodes; `_QKVAttn` then holds its own
incoming gradient (the attention output's, N x 1536 half) through the q/k/v backward, in place of the
q/k/v gradients (N x 2048 half) that `_QKVRope`'s node held there, and the q/k/v gradients die as soon
as they are in `g` (`_qkv_backward(box=...)` takes the only references). Same ops, operands and order
as `_AttnBlock`'s token-major path. `SOUP_SOLUTION_ATTN_ONE_FUNCTION=1` still wins when set; eval, the
non-recomputing layers and attention dropout take the three nodes.

Tests (`test_fused_proj.py`): `test_attention_two_nodes_is_bitwise` (three, two and one node: output,
input gradient and LoRA gradients bit for bit, LoRA dropout 0 and 0.05, node types checked);
`test_attention_two_nodes_backward_peak` (one layer, the output gradient made inside the backward:
allocated peak two <= three; one = three + N x 2048 bytes on the laptop). The other attention tests pin
`two_nodes` off where they compare three-node variants. T4 pytest of 1c66f61 (soup-c1-nodes):
**405 passed, 5 skipped, 0 failed**; Soup's 79 with apply() pass on the laptop.

### 3. The embedding lookup with host ids (`SOUP_SOLUTION_HOST_IDS=1`, step_pipeline.py, unpad.py, head_stream.py)

The host plan (B1) records the packed ids' CPU copy for their GPU view (`step_pipeline.remember`);
`unpad.varlen_model_forward` builds the alignment tail on the CPU copy too (same fill) and records it
for the tail-extended GPU ids; `head_stream.lookup` gathers with `step_pipeline.host_ids(ids)` (identity,
shape and dtype checked) and reads the GPU ids back only when there is none. The forward then starts
without a host wait. Tests: `test_host_ids_reach_the_embedding_with_the_tail` (37 and 48 tokens, tail id
None or set, switch on/off), `test_embedding_lookup_with_host_ids` (same rows, no `.to("cpu")` of the
GPU ids, a copy of another shape ignored), the tiny SFT run with all B1 switches still identical.

### 4. Pre-test "one autograd.Function per decoder block" (`solution/bench_nodes.py`, T4, soup-c1-nodes)

Host cost of one autograd node (a chain of 6 custom Functions against one Function running the same
6 ops, median of 100): **14 us forward + 19 us backward** at 300, 830 and 1,440 tokens (13.8-14.4 /
18.4-19.8 us). The attention block of one layer with the recompute (real fused code, 4 rows, host ms
to issue from an idle GPU, median of 100; kernel launches equal in all three):

| tokens | three nodes fwd / bwd | two nodes fwd / bwd | one node fwd / bwd | MLP fwd / bwd |
|---|---|---|---|---|
| 300 | 1.196 / 1.857 | 1.021 / 1.763 | 0.987 / 1.728 | 0.713 / 0.781 |
| 830 | 1.248 / 1.870 | 1.179 / 1.787 | 1.142 / 1.740 | 0.726 / 0.830 |
| 1,440 | 1.678 / 1.895 | 1.596 / 1.805 | 1.580 / 1.778 | 0.669 / 0.743 |

The attention block as one Function against today's three nodes: -11 % host time at 300 tokens
(3.05 -> 2.72 ms), -8 % at 830, -6 % at 1,440; two nodes take most of it (-9 %, -5 %, -5 %). The whole
layer costs 3.55 + 4.0 ms of host time per layer in training (prof_1 timeline, short steps); one
Function per decoder block would remove at most the attention's 0.34 ms plus the two norms' and the
MLP's node round trips (3 x 33 us) and their glue: about 0.5 ms, 6-7 % of the layer. Below the 15-20 %
threshold of the review: **not rewritten**. Most of what a merge saves is the Python glue between
nodes (three -> two: 0.27 ms, of which one node's round trip is 0.03 ms), not the autograd node itself.

### 5. ABBA of 2 + 3 (peagverr, soup-c1-abba5, 3e347e7 = ce27a54 + two nodes + host ids; `main` = the same commit with `ATTN_TWO_NODES=0, HOST_IDS=0`, i.e. ce27a54's paths; 2 rounds ABBA + warm run, with pytest)

| variant | wall s | steps 3+ s | median | < 600 tok ms | 600-900 ms | >= 900 ms | nvidia-smi | reserved | alloc peak | own |
|---|---|---|---|---|---|---|---|---|---|---|
| main path | 41.69 / 41.89 | 23.70 / 23.54 | 0.383 | 265.2 / 263.7 | 346.8 / 343.9 | 505.7 / 502.8 | 371 | 250 | 225.5 | 14.3 / 14.4 % |
| two nodes + host ids | 41.79 / 41.89 | 23.57 / 23.51 | 0.379 | 260.5 / 262.4 | 345.3 / 343.8 | 504.0 / 502.4 | 371 | 250 | 225.5 | 14.3 / 14.4 % |

Per step (mean of two runs each): < 400 tokens -3.3 ms, 400-500 -5.0, 500-600 -1.9, 600-900 -0.8,
>= 900 -1.0. Steps 3+ -0.08 s, wall +0.05 s: **in the noise** (below the 0.3 s mark). Memory the same
in every run (371 / 250 / 225.5 MiB). `64 lookups with host ids` in each c1 run (0 in main). T4 pytest
**405 passed, 5 skipped, 0 failed**; checks 3, 4, 9 pass in all five runs. Session note: the poller's
download broke mid-run (IncompleteRead) and returned three runs; `--attach` fetched the rest (output
dir 20261031-120802).

Why so little: the bench's -0.27 ms of host time per layer applies only where the CPU is the
bottleneck, the forward of the ~12 short steps (prof_1: the GPU 0.6-3 ms behind there); the
backward of short steps and every pass of the longer steps are GPU-bound. The CPU side of the step
has about this much left on the public rows; further CPU cuts are worth < 0.1 s each.

### 6. GPU glue from R3: k/v expanded once in the recompute (`SOUP_SOLUTION_EXPAND_KV_ONCE`, off)

`_AttnRecompute._rebuild` expands k and v to the 12 q heads once and hands the same tensors to its
attention forward and to the attention backward (which expanded the compact ones a second time); the
three backward sites expand only compact k/v and take the kv head count from q's heads / groups.
Bitwise in all three node layouts and with the recompute's own check (`test_attention_expand_kv_once_is_bitwise`;
also 2,047, 2,048 and 1,999 + 49 tokens on the laptop). T4 (peagverr, soup-c1-kv1, d1c252d, ABBA 2
rounds + warm run, no pytest): c1 wall 41.53 / 42.53 s, steps 3+ 23.75 / 24.20; kv1 41.63 / 41.82,
steps 3+ 24.00 / 24.04; **alloc peak 225.5 -> 224.1 MiB but reserved 250 -> 270, nvidia-smi 371 ->
391** (the expanded k/v live from the rebuild to the attention backward and fragment the pool). No
time gained, a 20 MiB page lost: **stays off**.

### 7. Why s is lower in the jury order (analysis of downloaded logs, no new T4 runs)

**First line: the growth sits in the training loop, in the GPU-bound steps of 600+ tokens** (the jury
rows have more of them: 30 and 43 steps of >= 900 tokens against 20, and at equal token count our step
loses 6-12 % against the base's); not in the start-up, the host, PCIe or the short CPU-bound steps
(7 and 3 of them in the jury rows).

Runs (main checkout `runs_kaggle/`): cold full soup-m2-full (ce27a54), m3 (8d2d0bd), m4 (0f9bc57), m6
(ca6db38), all on the public rows; jury soup-m2-jury (ce27a54, rows 15000:15256, order base / sol /
sol / base) and soup-m5-jury (6050a09, rows 18000:18256). Tokens per step from TRL's `num_tokens`
(4 significant digits), step times from `metrics.json`; "per token" = step time at equal token count
(jury steps matched to full steps within +-25 tokens, median ratio).

| | m2-full | m3 / m4 / m6 full | m2-jury sol1 / sol2 | m5-jury sol1 / sol2 |
|---|---|---|---|---|
| s | 1.929 | 1.934 / 1.960 / 1.960 | 1.80 | 1.85 |
| tokens, steps 3+ | 51,571 | 51,571 | 56,541 (+9.6 %) | 61,986 (+20 %) |
| our loop / base loop, s | 27.4 / 61.8 | 29.5-27.8 / 65.0-62.3 | 33.1, 33.6 / 65.9, 68.3 | 33.0, 32.3 / 66.3, 69.6 |
| our wall - loop, s | 16.7 | 15.7 / 16.1 / 15.6 | 15.9 / 16.1 | 17.4 / 17.3 |
| first step after process start, s | 15.0 | 14.4 / 14.4 / 14.7 | 14.8 / 15.1 | 16.4 / 16.3 |
| setup host -> GPU, GB/s | 10.6 | 10.8 / 10.4 / 10.8 | 11.0 / 10.6 | 11.2 / 10.8 |
| ours per 1k tokens, ms | 507 | 548 / 535 / 514 | 561 / 569 | 509 / 498 |
| ours at equal tokens vs full | | | 1.03-1.12 | 0.95-1.01 |
| base at equal tokens vs full | | | 0.95-1.00 | 0.88-0.93 |
| 2nd run / 1st run in session, loop | | | ours +1.5 %, base +3.7 % | ours -2.3 %, base +4.8 % |

Findings (facts from the table):
- The whole difference is the training loop: loop ratio base/ours 2.25 (m2-full) against 2.02 (m2-jury)
  and 2.08 (m5-jury). Start-up is the same (apply 3.3-3.6 s, setup 3.1-3.4 s, first step 14.4-15.1 s
  after process start) except m5's prolog: `apply begin` at +6.5 s against +5.3-5.8 (the harness and
  imports before apply(), host time, ~1 s in that one session).
- No sign of a slower host or PCIe: the load's host -> GPU rate 10.4-11.2 GB/s and its tensor loop
  1.9-2.2 s in every run; weight streaming and the head report 0 late copies everywhere (their waits are
  untimed by default, so not in the logs).
- Heat inside the jury session costs the base more than us: our second run -2.3 to +1.5 %, the base's
  last run +3.7 to +4.8 % (it raises the base mean, which helps s).
- The jury rows carry 10-20 % more tokens per step, and at equal token count our step loses 6-12 %
  against the base's relative to the public rows, in both jury sessions against all four cold ones
  (m2-jury: ours slower per token, base unchanged; m5-jury: ours unchanged, base cheaper per token).
- Session spread on the same rows: ours 507-548 ms per 1k tokens, base 1,135-1,212, moving together.

Not confirmed: which property of the rows does it (padding per batch for the base, row lengths for
attention, labelled share for our head). The per-row token counts are in each run's `rewind.jsonl`,
which tools/kaggle_run.py does not download (file pattern), and the jury batches are not in the logs.

My opinion: nothing to do about hot-card sensitivity on our side (our runs vary +-2 % within a session,
less than the base). The jury gap is per-token GPU cost on token-heavy rows; levers are per-token ones
(the int8 MLP backward, measured -12 to -36 ms per step), not start-up or CPU.

### State of the branch (end of session 2)

`claude/c1-cpu-diet-2` = main 141818e + this session. Defaults of the new switches are **off** (the
orchestrator's decision: in the noise on the T4): `SOUP_SOLUTION_ATTN_TWO_NODES=0`,
`SOUP_SOLUTION_HOST_IDS=0`, `SOUP_SOLUTION_EXPAND_KV_ONCE=0`; with them off the step runs main's code
paths. Tests for all three stay in `test_fused_proj.py`, `test_step_pipeline.py`, `test_head_stream.py`
(they set the switches themselves). `solution/bench_nodes.py` is the one-Function pre-test.
`test_attention_two_nodes_backward_peak` tolerates N x 1024 bytes on the laptop (three nodes' peak
depends on the tests run before it there); the T4 allocated peak is equal (abba5).
