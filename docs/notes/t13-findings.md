# T13 findings, 30 Oct ~21:00 (branch claude/t13-recompute-594cc5)

1. The "119 ms stall in torch.empty inside _MLP.forward" of the d9f60bb trace is `cudaHostAlloc`: the
   offload's pinned host stack grows by one 256 MB slab (`torch.empty(SLAB_BYTES, pin_memory=True)`, an
   aten::empty, called from the pack hook inside _MLP) whenever a step has more tokens than any before.
   Chrome trace: both 132 ms aten::empty spans contain a single `cudaHostAlloc` of 133.3 / 131.9 ms.
   6 slabs per run ("pinned stack 1536 MiB in 6 slabs") = ~0.8 s per run, not per step. Not the
   caching allocator, not record_stream. Fix (offload.py): prepin 1536 MiB in a background thread
   when the offload mode is applied (SOUP_SOLUTION_OFFLOAD_PREPIN_MB, now default 1536; thread joined
   at exit).
2. The reserved pool never shrinks during training, with offload or recompute: alloc_watch on T4 shows
   0 alloc retries, 0 segment frees, 0 syncs of all streams; reserved 592 -> ~1300 after step 1 -> max
   (1804-1824 on main 4744723); nvidia-smi peak = reserved + ~175 MiB. Check 9 passes in every run.
3. MLP recompute (main 4744723, diag runs of one commit): steps 3+ 24.86 s vs 22.52 s with main's
   offload (+~50 ms GPU per step: the gate|up GEMM runs twice), peak 1979 vs 2003 MiB. Not worth it
   while PCIe has room; kept as SOUP_SOLUTION_MLP_MEMORY=recompute. Without T12's padding it was far
   worse (+160 ms/step: cutlass align1 kernels); the fused MLP now pads N to 8 itself (no-op after T12).
4. Offload without record_stream and without CPU waits (stream waits on events; 795 CPU syncs per run
   before): steps 3+ 21.51 s vs 22.52 s, same memory.
5. GU_SPLIT=1 (recompute mode): -40 MiB nvidia-smi and reserved (1979 -> 1939, 1804 -> 1764), ~+4.5 ms/step
   (one more dequant per layer in the recompute backward; offload mode has no extra dequant).

# T18 jury emulation (31 Oct ~00:00, branch claude/t18-jury @ 060f366, pushed)

Tools: tools/jury_data.py (other rows of the dataset: --train-rows/--eval-rows ranges, or --select
longest|shortest|extremes; long rows only when the prompt leaves 64 answer tokens) and
tools/kaggle_run.py --mode jury (--jury-data=<args> with "=", --jury-seed N): run_task --repeats 2 --seed,
check_rules with Soup's tests, score; summary prints check lines, per-run numbers, tokens per step,
and the check 5/6 gaps of every run against base_1.

Analysed:
- J1 peagverr/soup-t13, commit 4167416 (= main 37dc053 + tools), rows 10000:10256 / 50000:50256, seed 7:
  ALL 10 CHECKS PASS. Gaps vs base_1 (limits 1% first, 3% others): solution loss first 0.013%, mean
  0.036%, entropy 0.067%, accuracy 0.25%, rewind row loss 0.20% (>= 12x margin); eval +0.004% (limit 0.5%);
  own kernels 14.9%; time x0.506, memory x0.098; peaks base 14,371 / solution 1,413 MiB (both runs);
  max 1,508 tokens in a step (above the public 1,427), 59/64 steps not a multiple of 16 (unpad pads).
- J2 peagverr/soup-t13d, commit 4167416, --select longest (unfiltered), seed 13: every run (base too)
  stopped in Soup's data prep: "train row 16: no causal-loss target remains after truncation at
  max_length=512" (rows whose prompt fills 512 tokens). Data artifact, fixed in jury_data (9dd950d).
- J3 peagverr/soup-t13e, commit a6e15a2, extremes unfiltered: same data error, every run.
- J3' peagverr/soup-t13e, commit 060f366 (= main 88e1afd + tools), --select extremes (128 shortest +
  128 longest trainable rows interleaved), seed 21: both SOLUTION runs finished (peak 1,755 MiB,
  reserved 1,586, median step 0.45 s, own kernels 13.9%, check 10 PASS) but BOTH BASE runs failed:
  "GPU ran out of memory during training" at peak 14,719 MiB. The base cannot train this row mix on a
  T4, so the jury's hidden rows cannot be like it; not our failure, but the base's headroom on long
  batches is thin (14.37 GB on ordinary rows).

Not run (stopped at shutdown, nothing on Kaggle): J2' longest rows filtered, seed 13, commit 9dd950d+.
To do next (cloud): `python tools/kaggle_run.py --user peagverr --mode jury --slug soup-t13d --commit <main>
--jury-seed 13 "--jury-data=--select longest"` (may OOM the base too: then use a range of rows with
many long rows, e.g. --train-rows 20000:20256 --eval-rows 30000:30256, and a second seed on J1's rows).
Re-analyse any finished kernel: `python tools/kaggle_run.py --user peagverr --slug soup-t13e --mode jury --attach`.

Next steps after T18 (from docs/RESEARCH.md):
1. Step-boundary pipeline (exp. 1, -15..-20 ms/step expected): ~26.5 ms GPU idle per boundary (grad-norm,
   AdamW 3 ms, .item(), _amp_update_scale_, DataLoader __next__ 2.8 ms). T13c measured that deferring
   the metric .item()s alone gives nothing (branch claude/t13c-syncs, not merged): the head's own
   labelled-count .item() already syncs at the end of the forward. The part to build: fetch and prepare
   the next batch (DataLoader next, collate, unpad.pack_batch spans, head_stream's host embedding
   gather + its H2D copy) before the scaler/clip syncs of the current step, e.g. a prefetching wrapper of
   the train dataloader iterator plus a hook that runs the next batch's host prep while the GPU still
   executes the optimizer step. Guard like the other trainer patches; keep values bit-identical
   (same batches, same order); measure with --mode variants against a switch.
2. PCIe balance (exp. 5): per step, recompute gate|up for k layers instead of offloading (both paths
   exist: fused_proj._MLP saves x16 when plan.recompute; offload skips small tensors). Needs a per-module
   flag (MLP layer index by first-call order) read in _mlp_plan, k chosen from the step's token count
   (k ~ 3 at N = 832); T12 works in _MLP save/rebuild lines, coordinate. Measure with variants on
   short and long steps; T13b measured full recompute +5.8 s/run vs offload in one session.
