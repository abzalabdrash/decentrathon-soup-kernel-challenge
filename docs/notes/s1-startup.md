# S1: apply begin -> first training step (branch claude/s1-startup-f929a7)

Work in progress; numbers are T4 (Kaggle), one session per comparison, means of 2 ABBA rounds after a
warm run, seconds from `apply begin` unless noted.

## Where the time went (main a6c3fe3, soup-s1-diag0 startup profile and soup-s1-knobs)

- apply 3.3 s: imports 2.8-3.0 s (torch._dynamo 0.89 via bitsandbytes' torch.compile wrappers;
  transformers.masking_utils imports it anyway, so it cannot be skipped; datasets 0.85, peft 0.45,
  transformers.models 0.34 through squad), our modules and patches ~0.3 s.
- Soup CLI 0.55 s, tokenizer 0.59 s, model load 3.6 s (tensor loop 2.25 s), trainer build 1.8 s,
  train() -> step 0.17 s, step 0 0.53 s (median step 0.40-0.44).
- Without the warm-up (SOUP_SOLUTION_WARMUP=0, knobs session) the first step comes at 7.67 s
  against 9.39 s: the warm-up threads cost the main thread 1.7 s (GIL), mostly in the model load
  (2.26 -> 3.58 s); its own length was 5.5 s from apply end, ending just before train begin.
- Main-thread items seen in the profile: the embedding's `.to(float16)` 0.59 s + its copy into the
  pinned buffer 0.08 s; bitsandbytes' `get_4bit_type` 0.44 s (a host wait per weight); Soup's row
  tokenization 1.66 s with 23 % of a CPU (`encode_batch` releases the GIL per row and waits ~2 ms to
  get it back while the warm-up runs); accelerate's nvidia-smi 64 ms.
- Warm-up composition: triton's C launcher builds (gcc) 2.3 of 7.2 warm-up thread-seconds during the
  model load, on the workers' critical path.

## Switches (all off by default until the T4 session decides)

- SOUP_SOLUTION_HELD_ENCODE (trainer_build.py): during Soup's setup a single-text `encode_batch` is
  answered with the Rust tokenizer's `encode` (keeps the GIL; same encodings, tested on the 256 rows).
- SOUP_SOLUTION_EMBED_ASYNC (load_stream.py, head_stream.py): the embedding's job returns its pinned
  buffer and a thread casts the mmap slice into it (`copy_`, the kernel `.to()` runs); joined before the
  load returns. Bit-equal (tests, tiny Qwen2 with and without the pinned copies, failing thread).
- SOUP_SOLUTION_LOAD_CODE_TABLE (load_stream.py): bitsandbytes' NF4 table built once per load and
  cloned on the GPU for every weight (same values, own tensor per quant state).
- SOUP_SOLUTION_LAUNCHERS_EARLY (warmup.py): at apply begin a thread probes every warm-up's launch
  signatures (triton's jit_cache_hook answers "skip" for that thread), builds the launcher sources
  with triton's `make_launcher` and gcc during the imports (two at a time); the warm-up's builds find
  them by source text. Locally (4 CPUs): 11 of 11 launchers ahead, warm-up 4.0 -> 2.8 s.
- SOUP_SOLUTION_EARLY_PROBES=p2p (trainer_build.py): only accelerate's nvidia-smi check in the thread.

## Measured and rejected

- GIL switch interval 0.2 ms / 0.1 ms during the warm-up (knobs session): first step 9.43 / 9.71 s
  against 9.39 s; one worker 10.27 s (waits 1.8 s for the warm-up).
- GCC precompiled header for the launchers: 0.12 -> 0.08 s per launcher, but two header builds of
  ~0.25 s each eat it (1.02-1.08 s against 0.85 s for 7 launchers locally).
- TORCHDYNAMO_DISABLE is already used for bitsandbytes; `torch.compile` itself imports
  torch._inductor.compiler_bisector -> torch._dynamo, which masking_utils needs anyway: nothing to win.
- filelock's audit hook: 39k audit events in apply at ~0.27 us = 11 ms.
- Meta-device init of the model skeleton (Linear.reset_parameters on meta): 0.02 s locally.
