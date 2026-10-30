# T15c hand-over (laptop shutdown): branch claude/t15c-startup, head 3f5b14c (pushed)

## State of the warm-up (in-process only)
- The helper PROCESS is gone from the code (718f68e), not just switched off: no subprocess, no
  `python -m solution.warmup --helper`, no offline driver, no stdin protocol. Everything compiles
  in the training process.
- warmup.py now: `start_early()` (first call in apply()) starts triton's own hash (`triton_key`,
  415 MB of libtriton.so) in a thread of its own next to the imports, under a lock;
  `start_background_warmup(final=True)` at the end of apply() queues every registration in the
  registry (this file's, fused_proj's rebuild, head_stream's kernels, cublas, step_ops) into a
  queue-fed thread with WORKERS workers. With SOUP_SOLUTION_WARMUP_START=apply_begin the thread
  starts in start_early on this file's registrations and the final call queues the late ones.
- The thread path covers all kernels the run launches (18 variants on Kaggle before the trim;
  after 7a74def the warm-up no longer compiles fused_ce_rows_kernel with head_stream on, nor the
  fp32 RoPE with fused_proj on: both never launched in the run). `_shutdown` at exit stops it
  between registrations (Soup tests exit 0, tested).
- Measuring switches: SOUP_SOLUTION_WARMUP_START, _WORKERS, _SWITCH_MS, _SKIP=name:name.
- Tests: 27 warm-up/startup/load_stream tests and Soup's 79 pass on 3f5b14c (merge of main 88e1afd).

## Kaggle (account 2 = peagverr), queued when the laptop went down: fetch with --attach or rerun
- slug soup-t15, --mode variants, commit 3f5b14c, --repeats 2: end (default), begin
  (WARMUP_START=apply_begin), w2 (WORKERS=2), sw5 (SWITCH_MS=5), nowarm. Was waiting for a slot
  (T13 held both). If it never started: rerun the same command.
- slug soup-t15p, --mode solution, commit 3f5b14c, SOUP_SOLUTION_PROFILE_STARTUP=1: profile of the
  model setup on merged code (T14b load_stream). Also waiting for a slot.
- Parse phases per variant with a parser over runs_var/*/solution_1/train.log marks relative to
  `apply begin` (apply, cli, model, trainer, to_step, step0, first_step_at).

## Measured so far (same-session variants, means of 2 rounds, seconds from apply begin)
- 00499585 (before trim, before T14b): helper 11.44 to first step (+0.51 step 0); in-process
  thread with early triton_key 12.02 (+0.50); no warm-up 9.07 (+6.67). Thread cost vs helper
  about 0.6 s; vs no warm-up about 3 s before the first step, 6 s saved in step 0.
- 3344e16 (helper era): helper workers 1/2/3 gave 11.07/11.04/11.35 s to the end of step 0;
  GPU warm-ups (cublas, step_ops) about neutral (model setup -0.46, step 0 +0.25 without them).
- Import diet worth 5.2 s, dataset hashing 1.9 s (variants of 2ed0117).

## Next steps
1. Read the variants session above: pick start point / workers / switch interval for the thread;
   report its cost against the helper numbers (apply 3.57, setup 5.41, step 0 0.51 in 00499585).
2. Hashing before the load is implemented (start_early); verify on Kaggle that the setup phase
   no longer shows hashlib/`read` of libtriton.so in the profile (soup-t15p run).
3. Re-split the model setup on merged code (load, load_stream pinning thread, weight/head
   takeover) from the soup-t15p profile; remember cProfile on Python 3.13 records all threads.
4. torchvision: still imported by the first step before 3deb206; 3deb206 extends the window to
   trl's trainer import (image_processing_utils / _backends via ProcessorMixin): check the
   "at the first step, loaded of ...: none" line.

## RESULT of the variants session (soup-t15, 3f5b14c, merged T14b; runs_kaggle/20261030-233456-variants-soup-t15 in the t15 worktree)
Means of 2 rounds, seconds from apply begin (apply / cli / model setup / trainer / first step at / step 0):
- end (default: thread at apply end, 3 workers, 1 ms): 3.44 / 0.50 / 5.05 / 1.29 / 10.46 / 0.50
- begin (WARMUP_START=apply_begin):                     4.56 / 0.72 / 3.37 / 1.21 / 10.01 / 0.48
- w2 (WORKERS=2):                                       3.34 / 0.39 / 4.71 / 1.12 /  9.73 / 0.48
- sw5 (SWITCH_MS=5):                                    3.37 / 0.50 / 4.95 / 1.13 / 10.11 / 0.51
- nowarm:                                               3.34 / 0.38 / 3.20 / 1.32 /  8.41 / 6.61
Best: w2 (first step at +9.73 s, under the 10 s target; walls 41.6 s vs 44.2 for the default).
The thread now costs 1.3 s before the first step against no warm-up and saves 6.1 s in step 0;
against the helper (00499585: 11.44 + 0.51) it is better, but sessions differ (T14b merged since).
Next: try begin + w2 together, and w2 + sw5; then set the default (WORKERS=2 is one line in
warmup.py: `WORKERS = ...` or the env default) and remeasure. The soup-t15p profile run (setup
split) was still queued: rerun if it did not start.
