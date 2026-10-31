# C1: the CPU side of the training step (branch `claude/c1-cpu-diet-0b2063`)

Status: draft, T4 numbers below are filled in as the sessions finish.

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
- `SOUP_SOLUTION_ATTN_ONE_FUNCTION=1` (fused_proj.py): a recomputing layer's `_QKVRope`, varlen
  attention and o_proj as one autograd node (`_AttnBlock`), the same ops in the same order; two
  Python round trips into autograd and five view nodes less per layer and direction.
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

(pending)
