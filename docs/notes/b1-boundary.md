# B1: the step boundary and the host waits inside the step (branch claude/b1-boundary-e6120e)

## Report

Branch `claude/b1-boundary-e6120e`, head 5f2f302 (code = 1be9e55 + a test fix). New
`solution/step_pipeline.py`; `boundary.py` taken over from night-boundary and fixed; `trainer_patch.py`,
`fused_ce.py`, the labelled-count line (+ import) of `head_stream.py`; tests in
`test_step_pipeline.py`, `test_trainer_patch.py`, `test_boundary.py`. `__init__.py` untouched.

Switches, all on by default: `SOUP_SOLUTION_STEP_BOUNDARY`, `SOUP_SOLUTION_NAN_FILTER=gpu` (=host
off), `SOUP_SOLUTION_PREFETCH_BATCH`, `SOUP_SOLUTION_HOST_PLAN`, `SOUP_SOLUTION_DEFER_METRICS`,
`SOUP_SOLUTION_BWD_SINGLE_THREAD`; `SOUP_SOLUTION_BOUNDARY_TIMING=1` is diagnostics only.

T4, ABBA, 3 rounds each, solution-only runs:
- var1: base 44.62 s wall / 26.25 s steps 3+; all switches but single-thread 43.29 / 25.24
  (-1.33 / -1.0 s).
- var3: base 42.18 / 24.73; all 40.99 / 23.67; plus single-thread 40.81 / 23.60, peak **371 MiB**
  instead of 401 (one cuBLAS workspace fewer: alloc -9.1, reserved -22).
- var4 (hot session): base 44.89; all 44.08 (-0.8). Peak unchanged except for the single-thread -30.

Boundary idle (trace, profiled): ~31 ms down to ~13 ms. What is left: Soup's tracker SQLite commit,
3.3-3.6 ms per log (~0.22 s per run, Soup's code, needs the organizers); AdamW's CPU after
`found_inf.item()`, ~3 ms (it stays).

Full run (1be9e55): all 10 checks, 42.5 s / 371 MiB against 80.0 s / 14,345 MiB, eval -0.012 %.
Jury run (seed 57, rows 20000/35000): all 10 checks, 46.2 / 47.9 s against 88.3 s, 411 MiB,
eval +0.008 %.

Why night-boundary did nothing: every forward slices `layers[:n]`, which re-registers the layers,
and that dropped its cached lists every step.

Follow-ups: the embedding lookup's `.to("cpu")` (unpad/C1 files). Also warmup `step_ops` creates
tensors without a dtype: it races with the loader's fp16 default dtype and fails at times.

## Where the time went (T4 trace of main a6c3fe3, soup-b1-diag0, steps 8-9 under the profiler)

- Harness: `Probe.on_step_begin` and `on_step_end` call `torch.cuda.synchronize()` (harness/_train.py),
  so the queue is empty at both ends of every step; anything between `on_step_end` and the next
  `on_step_begin` (log, Soup callbacks, DataLoader) runs with the GPU idle whatever we do.
- Inside the step: the forward is GPU-bound (the head's `int(labelled.sum().item())` waited 69.6 ms
  for it), the head is PCIe-bound, the TRL metric `.item()` waited 33 ms for the head, the backward
  is GPU-bound in long steps (transformers' nan/inf check of the loss waited 62.9 ms for it) and
  CPU-bound in short ones. The in-step drains cost only ~2-4 ms of GPU idle (backward start).
- Boundary (end of backward 530.8 ms -> first layer kernel 562.0 ms): ~31 ms GPU idle: FLOP
  counter walk 4.9 ms, clip launches ~5 ms (Python over ~450 parameters), AdamW 2.8 ms (GPU 0.8),
  after Adam 3.2 ms, log + callbacks 4.6 ms, DataLoader fetch/collate 2.7 ms, step start 3.4 ms,
  forward start (pack_batch nonzero/tolist, embedding host gather, cu_seqlens copy) 2.7 ms.
- Why night-boundary measured nothing: (1) every forward slices `layers[:num_hidden_layers]`
  (unpad.varlen_model_forward and HF's Qwen2Model alike); a sliced ModuleList registers all 28
  layers into a new ModuleList, which fired torch's global registration hook and dropped the cached
  lists every step, so the FLOP counter rebuilt them (8.8 ms per call on the T4, timing run); (2)
  its `train()` fast path never engaged (Qwen2 models use `PreTrainedModel.train` in
  transformers 5.16, and its own `PeftModel.train` failed its "plain" check). Both fixed: only
  registrations on the listed model's own modules drop the lists; PreTrainedModel.train is accepted
  while `use_kernels` is off (checked per call).

## What changed (each behind a switch; all on by default after 1be9e55)

- `SOUP_SOLUTION_STEP_BOUNDARY` (boundary.py, from night-boundary, fixed as above): FLOP count, clip,
  zero_grad, train() from per-run lists.
- `SOUP_SOLUTION_NAN_FILTER=gpu` (step_pipeline.py): transformers' nan/inf filter of the logged loss
  as `torch.where(isfinite(loss), loss, _tr_loss / k)` returned by training_step, the host check
  switched off for the epoch (restored after it and around checkpoint saves): the same addend bit for
  bit, no wait after the backward; the CPU queues the clip during the backward.
- `SOUP_SOLUTION_PREFETCH_BATCH` (step_pipeline.py): the batch after next is fetched and collated
  right after the backward is queued (main thread, not a background thread or process), same batches
  in the same order, the rewind FIFO gets the same indices in the same order.
- `SOUP_SOLUTION_HOST_PLAN` (step_pipeline.py, fused_ce.py, the labelled-count line of
  head_stream.py): the unpad plan, the head's labelled count and `num_items_in_batch` from the batch's
  CPU tensors where the DataLoader hands it over, one non-blocking pinned copy; matched to the GPU
  tensors by identity, anything else takes the GPU path.
- `SOUP_SOLUTION_DEFER_METRICS` (trainer_patch.py): TRL metrics and Soup's rewind rows stay on the
  GPU until TRL's log(), read there in one device-to-host copy (float64 holds every fp32 value and
  count exactly); same floats in the same log entries, same rewind.jsonl lines.
- `SOUP_SOLUTION_BWD_SINGLE_THREAD` (step_pipeline.py, from the external review): the trainer's
  `accelerator.backward` under `torch.autograd.set_multithreading_enabled(False)`: the backward runs
  on the calling thread, so one cuBLAS handle/workspace serves forward and backward (the autograd
  device thread made a second one). Bitwise the same step (test through the weight ring, the MLP
  offload's events and the attention recompute; the backward hooks run on the calling thread).
- `SOUP_SOLUTION_BOUNDARY_TIMING=1` (diagnostics only): host ms of every callback per event, the log,
  Soup's tracker and display, the boundary's own wrappers, list rebuilds.

## T4 measurements (variants sessions, ABBA, 3 rounds + warm-up run, solution-only runs)

soup-b1-var1 (27e3d4b, namedesign): wall / median step / steps 3+ / peak
- base 44.62 s / 0.402 / 26.25 / 401 MiB; boundary 44.67 / 0.405 / 26.16 (lists rebuilt every step);
  + nan filter 44.14 / 0.393 / 25.73; all 43.29 / 0.390 / 25.24 / 401 (-1.33 s wall, -1.0 s steps).

soup-b1-var3 (ffed6bf, namedesign, a cooler session):
- base 42.18 / 0.387 / 24.73 / 401 (reserved 272, alloc 234.6); all 40.99 / 0.371 / 23.67 / 401;
  all + single-thread backward 40.81 / 0.373 / 23.60 / **371 MiB** (reserved 250, alloc 225.5:
  one cuBLAS workspace, 9.1 MiB, less; -30 MiB in nvidia-smi); the same with timing 40.90.
- Timing (all + single thread, per step): Soup's `ExperimentTracker.log_metrics` (SQLite commit)
  3.3-3.6 ms, Soup's whole on_log 3.7 ms, TRL/transformers log path 8 ms in all, the FLOP counter
  8.8 ms (the rebuild, fixed in 7893770), clip 1.4 ms, get_batch_samples 0.5 ms; the display 0.03 ms.
  The tracker commit (~0.22 s per run with the GPU idle) is Soup's; changing it needs the organizers.

Trace with all switches (soup-b1-diag1, 27e3d4b): the prefetch (3.5 ms) and the clip run in the
backward's shadow, the CPU waits 73 ms at `found_inf.item()`; GPU idle from the backward's end to
the first layer kernel ~13 ms (was ~31): AdamW 3.5, after it 1.2, log 4.7 (tracker commit 4.3),
step and forward start ~3. In short CPU-bound steps the backward ends before the CPU gets ahead
and the prefetch is on the critical path (it is there in main as well, between the steps).

soup-b1-var4 (7893770, peagverr, a hot session, base rounds 44.3-45.3 s):
- base 44.89 / 0.422 / 26.93 / 401; without boundary.py 44.29 / 0.420 / 26.50; all (boundary.py
  with the fixed lists) 44.08 / 0.418 / 26.25; all + single-thread backward 44.08 / 0.418 / 26.39 /
  371 MiB. boundary.py now adds ~-0.2 s; the single-thread backward is time-neutral in both sessions.

Full run (soup-b1-full, 1be9e55 = all defaults on): **all 10 checks, 42.5 s / 371 MiB** against the
base's 80.0 s / 14,345 MiB (hot session: s 1.884, m 38.67), eval -0.012 %, loss curve first step
0.01 %, mean 0.04 %, check 5 identical within tolerance, Soup tests 79/79. Kaggle pytest 374 passed,
5 skipped, 1 failed: the new tiny-SFT test went DataParallel on the notebook's two T4s (fixed in
5f2f302: the test's trainer on one GPU; the harness runs one GPU).

Jury run (soup-b1-jury, 1be9e55, seed 57, train rows 20000:20256, eval rows 35000:35256, base /
solution / solution / base): **all 10 checks**, solution 46.18 / 47.94 s against 88.48 / 88.08 s,
peak 411 MiB (base 14,407), eval +0.008 %, loss curve 0.02 % / 0.04 %, check 5 identical within
tolerance.

(a last diag trace with pytest on 5f2f302, soup-b1-diag2, was queued behind other sessions' runs)

## Not done / follow-ups

- The embedding lookup in head_stream.lookup still reads the packed ids back (`.to("cpu")`, one
  wait at the forward start, ~0.1-0.5 ms with the host gather): removing it needs the tail-extended
  ids from unpad.varlen_model_forward, the orchestrator's and C1's files.
- Soup's tracker commit per log: report to the organizers before touching it.
- warmup.py's `step_ops` job sometimes fails ("Attempting to unscale FP16 gradients", seen in this
  branch's diag1 and in c1/e1 runs): it creates `torch.zeros(...)` without a dtype while the model
  loader has set the default dtype to fp16; then AdamW/clip kernels load in the first step. Fix:
  `dtype=torch.float32` in warm_step_ops (warmup.py is not this task's file).
