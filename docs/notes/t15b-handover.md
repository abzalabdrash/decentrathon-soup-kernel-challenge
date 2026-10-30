# T15b hand-over (usage limit reached): branch claude/t15b-startup, head 67b30dc (pushed)

Not yet measured on Kaggle: a clean `--mode solution` run of 67b30dc was relaunched on account 2,
slug soup-t15 (results under the t15 worktree: runs_kaggle/<stamp>-solution-soup-t15/). Check:
phases (python phases parser: marks relative to `apply begin`), the "warmup done ... N compiled
here" line (expect 0 compiled here: head_stream_kernels and fused_proj_x16_rebuild now go to the
helper), "at the first step, loaded of ...: none" (scipy.optimize / torchvision skipped for real),
step_s[0] vs median. Tests: 228 solution tests pass, Soup's 79 pass (rc 0 every run).

## Commits on the branch (from main ba8c1b3)
- 6634925 startup profile also covers `_setup_transformers` (note: Python 3.13's cProfile records
  all threads, so the warm-up thread's triton_key hashing / gcc builds appear in it).
- 12ca43f warmup: every registration made after the helper started (fused_proj's rebuild,
  head_stream's kernels) is sent to the helper over stdin by name (module-level function or
  functools.partial with JSON args) and compiled there; only lambdas/closures stay in-process.
  Fixes main's "4 compiled here" (1.7 s of GIL contention during the model load).
- 67b30dc warmup: `step_ops` (gpu=True) runs AdamW(wd 0.01) + GradScaler + clip_grad_norm_(1.0)
  + first backward on two dummy fp32 params in the warm-up thread (no RNG drawn: tested), for step
  0's first-use costs (AdamW step 0.13 s, clip 0.10 s in main's profile). startup: while
  transformers.loss imports, is_scipy_available / is_torchvision_available answer False
  (scipy.optimize 0.28 s + torchvision 0.14 s on Kaggle, object-detection losses only); the
  first-step mark logs which skipped modules are loaded.

## Findings (main ba8c1b3, Kaggle profile run 20261030-211038-solution-soup-t15)
- apply ~4 s is almost all base-paid imports (transformers models package, peft, bnb, datasets,
  trl, torch._dynamo). Replacing torch.compile during the bnb import gains nothing (masking_utils
  imports dynamo + most of inductor anyway: measured 0.90 vs 0.91 s).
- Model setup (main thread, cProfile): tokenizer load 0.89 s (base-paid), tensor materialization
  0.80 s (pageable H2D of the bf16 mmap + cast; reading the slice is free, so read-ahead does not
  help; a pinned double-buffer would be the only lever, ~0.3 s, intrusive), weight_stream pinned
  buffers 0.45 s (fill_ 0.38 s) and head_stream 0.32 s (both T14's files: propose to T14), peft 0.21,
  conversion-mapping regex compiles 0.27 (base-paid).
- Trainer build: Soup's own row formatting 0.86 s (base-paid) + remaining fingerprint 0.45 s
  under cProfile.
- Proposal (trainer_patch.py, not mine): `_log_trl_metrics` does 3-4 `.item()` right after the
  forward; keep entropy/tokens/accuracy on the device (accuracy via torch.where) and convert in
  SFTTrainer.log, where a sync happens every step anyway: same logged values, no sync at the
  forward/backward boundary.

## Result of 67b30dc (Kaggle, account 2, slow session: prelude 15.4 s; run 20261030-212908-solution-soup-t15)
Wall 54.4 s, peak 1371 MiB, checks 2/3/4/9 PASS. From apply begin: apply 3.64 s, CLI 1.53, model
setup 5.31, trainer build 1.04, train->step 0 0.17 (first step at +8.05... +12.05 s after apply
begin, i.e. 27.07 - 19.02 + 3.64 = 11.7 s). Warm-up: 18 kernels loaded from the cache, 0 compiled
in-process (head_stream / rebuild now in the helper). step_s[0] 0.47 s vs median 0.38 (main: 0.64).
Open: torchvision is still loaded by the first step (someone imports it outside the loss window:
find the importer with --mode diag); the helper now takes 9.4 s (head_stream_kernels 3.4 s) and is
done during the model setup; model setup 5.31 s in this slow session vs main's 6.1 s in a normal
one: needs a same-session comparison before claiming the gain.
