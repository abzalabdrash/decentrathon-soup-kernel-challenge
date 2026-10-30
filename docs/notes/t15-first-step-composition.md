# T15: what the first training step spends (Kaggle T4, commit 6812ff4 = pre-T12 main + T15, cProfile)

Run: runs_kaggle/20261030-195922-solution-soup-t15 in the t15 worktree, SOUP_SOLUTION_PROFILE_STARTUP=1
(solution/startup.py prints three cProfile tables into train.log: trainer build, train() up to the
first step, the first step). cProfile inflates Python; shares count.

First step (1.87 s under cProfile, step_s[0] 1.51 s against 0.38 median):
- 0.50 s offload.py pack -> _offload -> push -> torch.empty (596 calls, 0.43 s self): the pinned
  host stack of the MLP offload is allocated (cudaHostAlloc) during step 0. NOT T15's file (T13/T4).
  If offload stays, pre-pinning it during setup (or in a background thread during the trainer
  build, which is CPU tokenization) would remove most of it; if recompute replaces offload, gone.
- 0.11 s accelerate clip_grad_norm_ first call (foreach norm kernels, lazy CUDA module load).
- ~0.1 s two rope_qk_kernel variants compiled in the step (fused_proj's fp16 q/k path): fixed in
  T15's warmup.py (7cca7a0).
- 0.09 s fused_proj _weights first-time buffers; 0.07 s nf4 dequant first launches.

train() up to the first step (0.64 s): accelerate free_memory -> gc.collect 0.28 s (T15: gc.freeze()
at the end of apply(), the collection then skips the ~0.5 M imported objects); Soup's
attach_plugin_callback -> importlib.metadata.entry_points() 0.16 s (full scan of Kaggle's
distributions; base pays it too).

Trainer build (3.1 s real; 9.4 s under cProfile): 8.0 s of it was datasets fingerprinting Soup's
map: dill pickles the tokenizer's Python _vocab (151,643) and _merges (151,387) in pure Python.
T15 fix (startup.py, patch_soup_dataset_hashing): during Soup's setup, datasets' reducer for the
tokenizer type gets those two containers C-pickled (same content): 0.09 s instead of ~1 s locally.
The rest is the real formatting/tokenization of 256 rows (~0.9 s under cProfile).
