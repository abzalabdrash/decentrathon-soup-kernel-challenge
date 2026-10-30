# T15 report: time before the first step (branch claude/t15-startup-24deb2, head 0cdd92d)

Measured: Kaggle T4, --mode solution, commit f613b3c (= 0cdd92d minus a docstring), account 2.
Run dir (t15 worktree): runs_kaggle/20261030-205514-solution-soup-t15. Wall 43.5 s, peak 2031 MiB,
loop 25.74 s, median step 0.369 s, checks 2/3/4/9 PASS, no Triton compile outside the warm-up.

## Composition, seconds from `apply begin` (prelude excluded: 6.1-16 s, session-dependent, same for base)

| phase                         | d9f60bb (full run) | f613b3c |
|-------------------------------|-------------------:|--------:|
| apply()                       | 8.15 | 4.07 |
| Soup CLI -> model setup       | 0.73 | 0.67 |
| model load + kbit/LoRA setup  | 5.70 | 4.59 (4.01 in the 7cca7a0 run) |
| trainer build                 | 3.10 | 1.05 |
| train() -> first step         | ~0.47 (derived) | 0.18 |
| first step step_s[0]          | 1.39 | 1.09 |
| total to end of step 0        | ~19.5 | 11.65 |
| exit tail (exit mark -> wall) | 1.66 | 0.06 (0.64 in 7cca7a0) |

## What changed
- warmup.py: helper process `python -m solution.warmup --helper <arch> <half> <names>` started first
  in apply(); CUDA_VISIBLE_DEVICES="" (no CUDA context), offline triton driver for the device's
  target, nice 10, builds launchers too. Thread at apply end: triton_key, waits for helper, loads
  14/14 kernels from the disk cache (no C build in-process); falls back to compiling. rope fp16
  variants (fused_proj) added. watch_compiles logs disk-cache loads vs compiles (compilation
  listener). `_shutdown` at exit replaces main's `_wait_at_exit`: no new registration, helper
  killed, thread joined (Soup tests exit rc 0, fast). Triton cache writes are atomic (temp dir +
  os.replace, group index last), so a killed helper leaves no half entry.
- startup.py: prepare_imports() (scoped: trl gets transformers' packages_distributions mapping;
  transformers.utils.is_sklearn_available False while candidate_generator and transformers.data
  import; transformers.is_wandb_available False while trl's trainer imports (on the sys.modules
  object: transformers swaps its module on first attribute read); TORCHDYNAMO_DISABLE=1 while
  bitsandbytes imports, so its CPU-fallback torch.compile wrappers stay plain);
  freeze_imported_objects() = gc.freeze() at apply end; patch_soup_dataset_hashing() (datasets'
  pickler dispatch for the tokenizer type during Soup's setup: _vocab/_merges C-pickled, 0.09 s
  instead of ~1 s; removed when setup returns); first-step phase mark; SOUP_SOLUTION_PROFILE_STARTUP.

## apply() diff (solution/__init__.py on the branch)
Begin: `start_compile_helper()` if SOUP_SOLUTION_WARMUP; `prepare_imports()` if
SOUP_SOLUTION_IMPORT_DIET. After patch_soup_phases: patch_soup_first_step_mark();
patch_soup_dataset_hashing() if SOUP_SOLUTION_DATASET_HASH; patch_soup_startup_profile() if
SOUP_SOLUTION_PROFILE_STARTUP=1. Warm-up block: watch_compiles() before patch_soup_warmup.
End: freeze_imported_objects() if SOUP_SOLUTION_IMPORT_DIET.

## C (CUDA warm pass): not done, on purpose
A synchronous warm pass moves lazy-loading cost from step 0 into setup and adds its own compute
(net loss). A background pass through the real model would consume the CUDA RNG (LoRA dropout 0.05)
and race with TRL/Trainer init (set_seed in Trainer.__init__, hook registration). Step 0's remaining
excess (~0.7 s) is mostly offload.py allocating its pinned host stack in step 0 (0.43 s self under
cProfile, 596 torch.empty): T13/T4's file, see t15-first-step-composition.md.

## Not taken (options)
transformers.loss.loss_for_object_detection imports scipy.optimize (0.28 s) + torchvision (0.14 s)
for DETR losses (is_scipy_available / is_vision_available flip, base pays too, unclear whether
later phases import them anyway); entry_points() cache (~0.1 s, Soup plugins); bitsandbytes'
torch.compile still imports the torch._inductor package (~0.26 s beyond dynamo).
