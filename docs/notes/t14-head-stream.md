# T14: head streaming (branch claude/t14-head-stream-c5d5d5)

## Finding worth knowing for everyone: step 1 is skipped in our runs (not in the base)

Kaggle full run 20261030-185502 (d9f60bb), train.log: base step 1 `grad_norm 1.533`, ours step 1
`grad_norm nan`, and our lr is then one step behind the base for the whole run (step 2: base lr 1e-4,
ours 0). Cause: `fused_ce.FusedLMHeadLoss.backward` did `dh16 * grad_loss.to(fp16)`; GradScaler's
first scale is 65536, which is inf in fp16 -> inf/nan grads -> GradScaler skips step 1 and halves the
scale. The base never sees this (its fp32 CE gradient is cast to fp16 after the scale is applied and
divided by num_items). Fixed on this branch in two ways: the streamed head keeps dL/dh in fp32, and
the resident fallback head multiplies in fp32 (test_fused_ce.py::test_backward_with_the_first_amp_scale).
Expect our loss curve to track the base slightly closer from step 2 on.

## Kaggle T4 diag, run 1 (d520268 = d9f60bb + T14, before T12; runs_kaggle/20261030-195529-diag-soup-t14)

- nvidia-smi peak 1,729 MiB (d9f60bb: 2,385), torch max reserved 1,550, largest step peak allocated
  1,393 (d9f60bb: 1,874), median step 850; check 9 clean; own kernels 11.1%.
- setup: alloc 1,100 MiB right before prepare (the loaded model: NF4 layers + fp16 embedding), after
  prepare 64 MiB, setup peak 1,146 MiB (was 2,513). What remains is the model load itself.
- static 151 MiB (was 596). Step 6 (767 tokens) peak 843 MiB (was 1,353), now set by the last
  layer's MLP forward (fused_proj gate|up 2 x 26 MiB + 13 MiB), not the head.
- median step 0.4246 s vs 0.416 (d9f60bb): the head waits 10.1 ms/step for PCIe (19 chunks x 2.0 ms
  at 12.2 GB/s = 445 MB/step HtoD); head_chunk_kernel 4.3-4.7 ms/step with 8 warps (low occupancy on
  sm_75) -> 16 warps in f49c2a3.
- the ring prefetched at the embedding was alive at that MLP peak (+48 MiB): f49c2a3 prefetches at
  the final norm instead (SOUP_SOLUTION_HEAD_PREFETCH=embedding|norm|head).
- the allocator stalls (aten::empty 120 ms x2 in steps 8-9) are still there on that base (T13's).

## Kaggle T4, runs 2-3 (main 4744723 with T12 merged + T14; prefetch at the final norm)

- solution mode (a1f55b4, runs_kaggle/20261030-202134-solution-soup-t14) vs T12's solution run
  (20261030-194842): nvidia-smi peak 1,651 vs 1,991 MiB; torch max reserved 1,484 vs 1,824; median
  step 0.3504 vs 0.3289 s (+21 ms); own kernels 14.6%; check 9 clean; eval loss 2.04941.
- careful: diag mode adds ~47 ms to the median step (T12: diag 0.3756 vs solution 0.3289); compare
  diag with diag only (ours diag 0.3716 vs T12 diag 0.3756).
- diag (f49c2a3): setup peak 1,146 MiB (the loaded model before prepare: 1,100 alloc); static 151;
  largest step peak allocated 1,205 MiB (median 730); step 6 peak 719 MiB, at the head (12 MiB
  logits buffer + 48 MiB ring); head waits 8.8 ms/step for PCIe; head_chunk_kernel 4.5 ms/step.
- the step-time cost is the PCIe floor: 445 MB/step at 11.7-12.2 GB/s = 38-40 ms vs ~27 ms for the
  resident head's GEMMs; SOUP_SOLUTION_HEAD_PREFETCH=embedding buys ~4 ms back for +48 MiB at the
  last layer's MLP.
- flaky on the T4 in our suite (not T14): test_weight_stream::test_stream_memory (passed in run 2),
  test_warmup::test_model_step_compiles_nothing_after_warmup (torchao ImportError on Kaggle; locally
  it fails at b5259a7 too).

## What the branch does

- `solution/head_stream.py`: the tied embedding / lm_head weight goes to pinned host memory right
  before peft's `prepare_model_for_kbit_training` (inside Soup's `_setup_transformers` only); the
  Parameter keeps a 4-byte fp32 placeholder (stride-0 expand) so peft does not cast it and
  model.dtype/device stay as in the base. Embedding lookup = host gather + 2.5 MB copy (bit exact).
  The head = one pass over vocabulary chunks (8192 rows, 2-slot ring on a copy stream, first two
  chunks prefetched at the embedding lookup), own kernels `head_chunk_kernel` (online max/sum/
  entropy/argmax/target + P in place + acc rescale) and `head_finish_kernel`, cuBLAS GEMMs
  (`addmm(..., out_dtype=float32)` for the gradient accumulation).
- Integration: `fused_ce.patch_qwen2_causal_lm()` calls `head_stream.patch_head_stream()` unless
  `SOUP_SOLUTION_HEAD_STREAM=0`; no change in `__init__.py` needed. `embed16.py` leaves a streamed
  weight alone and compares in 8192-row blocks (T13's note: the 2,513 MiB setup peak).
- Warm-up: registers `head_stream_kernels` with warmup.register; warmup.py's own `warm_fused_ce`
  now compiles a kernel the streamed path never launches (harmless; can be skipped when streaming).

## T14b (branch claude/t14b-setup-peak): load-time streaming, head prefetch default

- solution/load_stream.py: inside Soup's _setup_transformers, for a bnb 4-bit on-the-fly load of a
  CUDA Qwen2: caching_allocator_warmup skipped; the embedding read to the host (get_device -> cpu for
  its key) and installed as head_stream's placeholder; each decoder layer's NF4 tensors to a pinned
  buffer in weight_stream's layout as soon as its last Linear4bit is quantized; WeightStreamer adopts
  the buffers (no second copy). Pinned buffers are made by a thread alongside the load (sizes from the
  meta modules, exact). Declined streaming puts everything back. SOUP_SOLUTION_LOAD_STREAM=0 turns it off.
- T4 diag (5b7b4f3, runs_kaggle/20261030-222019-diag-soup-t14): setup peak 64 MiB allocated (was 1,146;
  1 MiB right before prepare), largest training step 1,065 MiB, nvidia-smi 1,361, reserved 1,182.
- T4 variants session (b3ba871, 2 rounds each, runs_kaggle/20261030-222843-variants-soup-t14v):
  setup_transformers phase 4.69 s (load on) vs 5.17 s (off); steps 3+ equal (25.9 s); torch max
  allocated 1,065 vs 1,136 (the setup); nvidia-smi 1,349 vs 1,351 (training's reserved sets it now).
  Head prefetch at the embedding: nvidia-smi 1,309 / reserved 1,142 vs 1,349 / 1,182 at the norm, same
  time -> default flipped to "embedding" (89fa27f). High-priority head copy stream: no effect.
- First run of a variants session is a cold-cache outlier (+8 s before setup).
- Variants session 2 (34560e8, prefetch at the embedding by default; runs_kaggle/20261030-225509-variants-soup-t14v):
  head_chunk_kernel 6.2-6.9 ms/step on the T4 whatever the warps (4/8/16) or chunk rows (4096/8192);
  steps 3+ within noise (base 26.6, rows4k 28.1, w4 27.3, w8 27.6 s); peak 1,309 MiB for all. Defaults
  kept (8192 rows, 16 warps). The head waits ~5 ms/step for PCIe, so a faster chunk kernel would not
  shorten the step at typical token counts; a redesign (several rows per program, or a sub-block loop
  with a reference max) only pays when the head becomes compute-bound.

## T19 (branch claude/t19-gemm-bench): cuBLAS GEMMs on the T4, measurement only

- solution/bench_gemm.py + tools/kaggle_run.py --mode bench. Three Kaggle sessions
  (runs_kaggle/20261030-232837, -233619, -234435 bench-soup-t14, gemm.log in output/repo/runs_bench).
- The T4 throttles hard under back-to-back GEMMs: SM clock 527-1,397 MHz (max 1,590), the 8960-wide
  GEMMs averaging 600-900 MHz, q|k|v and o 1,000-1,370. So 17-25 achieved TFLOPS in the bench and the
  gap to 65 TFLOPS is mostly power/clock, not tile scheduling. Clock-normalized (us x MHz) noise ~ +-10%
  (gate_fwd: the same 128x128 kernel under both libraries differed by 11%).
- Robust across runs: qkv_bwd split-K in two K-halves into the fp32 output +31..34% at M=832 (the
  default 256x128 kernel makes 48 tiles = 1.2 waves on 40 SMs), -21..27% at M=1440; o_fwd via cublasLt
  (256x64_sliced instead of 128x128) +11..22%; down_fwd via cublasLt (256x64_sliced instead of 128x256,
  42 tiles) +12..24% in 3 of 4 readings. Everything else (gate/up/down 8960, qkv_fwd) within noise;
  two streams never helped; split-K for fp16 outputs always worse.
- Estimated gain if applied selectively: ~0.1 ms/layer qkv_bwd (M < ~1100) + ~0.15 ms/layer down_fwd +
  ~0.03 ms/layer o_fwd = ~7 ms/step at ~830 tokens (~2% of a step). preferred_blas_library("cublaslt")
  process-wide: layer totals +3% (M=832) / +6..10% (M=1440) in cycles, inside the noise for most
  GEMMs and flagged experimental by torch: not worth proposing on its own.

## Status at shutdown (30 Oct, late): T19b in flight

- Branch claude/t19b-gemm-tune, commit 7ed1b27 (pushed, complete, not WIP): fused_proj.py gets three
  switched GEMM changes, all OFF by default until measured: SOUP_SOLUTION_QKV_SPLITK=1 (q|k|v backward
  in two K-halves into the fp32 output when M < 1100; one extra fp32 rounding), SOUP_SOLUTION_O_LT=1
  (o_proj forward through cuBLASLt via torch._C._set_blas_preferred_backend around the one call),
  SOUP_SOLUTION_DOWN_LT=1 (down_proj forward through cuBLASLt). solution/test_gemm_tune.py (4 tests);
  test_fused_proj + test_model_path pass with all three on (36); Soup's 79 pass.
- Kaggle variants session still running at shutdown: account 1, slug soup-t14v (kernel version 3),
  commit 7ed1b27, variants base / qkvsk / olt / downlt / all3, 2 rounds. Fetch tomorrow from the
  worktree root: `wsl -d Ubuntu -- bash tools/kaggle_wsl.sh 1 --attach --slug soup-t14v --mode variants`.
  Then set the defaults of the switches that win (steps 3+ and median; expectation ~7 ms/step at 830
  tokens) in fused_proj._GEMM and push.
- Next (orchestrator's list, after T19b): branch from main + merge claude/t19-gemm-bench (bench_gemm.py
  and --mode bench live only there) and extend the bench:
  (1) dequant side stream (RESEARCH exp. 3): microbench nf4.dequantize_nf4 of a (8960,1536) weight on a
      side stream concurrently with a layer GEMM on the main stream (bnb quantize_4bit for the packed
      weight): alone / sequential / concurrent; if most of the ~32 ms/step of dequant hides, implement
      a double buffer in fused_proj._weights (dequantize the next weight into the second buffer on a
      side stream, events only, +20-26 MiB);
  (2) token chunking (exp. 4): sweep M 512..1440 for each layer GEMM (default library, as trained,
      interleaved across M), TF and clock-normalized cycles per flop; only if TF falls with M in
      isolation, split the big GEMMs into two M-halves above a threshold;
  (3) qkv_fwd without the bias epilogue (mm instead of addmm(bias), bias added later) in the same bench.
