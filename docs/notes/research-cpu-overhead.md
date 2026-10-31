# R1: host-side overhead of an eager PyTorch 2.11 training step, without CUDA graphs or compile

Date: 2026-10-31. Branch: `claude/r1-cpu-research-640d26`. Research only: no code under `solution/` changed.
Scope: where the host (CPU) time of one Soup step goes, which techniques cut it, and which the rules allow.
Calibration scripts and raw numbers: `docs/notes/r1-bench/` (`results-local-wsl2.txt`).

Legend: **[V]** verified in source (file:line, versions as installed) or measured here; **[I]** inference from [V] items;
**[S]** search snippet only, not read in full, do not rely on it.

## 1. Short answer

1. **Dispatch is not the 0.30 s.** On this machine one CUDA op costs about 8 µs of host time, a Triton launch 10–13 µs,
   an autograd.Function call 3–4 µs. Our ~1,800 launches per step therefore cost about 15–25 ms of host time, not 300 ms [V measured, §3; launch count from RESEARCH.md:39].
2. **The gap is Python glue, and it is not located yet.** A plain bitsandbytes NF4 stack (7 `Linear4bit` per layer, no own kernels, no hooks)
   costs **at most 1.5 ms of host time per layer** for forward plus backward (upper bound, §4). Our layer costs **5.4 ms** (STATUS.md:268,
   measured on the laptop with the harness wrapper). About 3.9 ms per layer is outside dispatch [V measured; the cause is [I]].
   The first step is a profile of one layer, not a new kernel.
3. **Cheap and bit-identical:** remove host waits (`.item()`, pageable copies), group CUDA events, bind ops and streams once, build the
   parameter lists once per run, use `_foreach` for the walks, one autograd node per MLP, fewer saved tensors.
4. **Banned:** CUDA graphs for our kernels, `torch.compile` as our kernels, the banned libraries, any helper or background process.
   **Grey:** a Python prefetch thread for the next batch (the rulings allow an in-process thread for Triton compile, not for batch prep; ask).

## 2. Method and sources

**Read in full** (v2.11.0 source at `https://github.com/pytorch/pytorch/blob/v2.11.0/<path>`, or the installed wheel where noted):
- Horace He, *Making Deep Learning Go Brrrr From First Principles*, https://horace.io/brrr_intro.html (overhead section: Python 32 M additions/s, PyTorch 280 k ops/s; CUDA graphs and tracing as the way around it).
- `torch/_tensor.py` `Tensor.__torch_function__` (L1682–1705) and `_convert` (L1882); `torch/overrides.py` `get_default_nowrap_functions` (L396–405).
- `torch/csrc/autograd/python_variable.cpp` property getters (`THPVariable_get_shape` L2991, `requires_grad` L2767): each calls `check_has_torch_function` first.
- `torch/csrc/utils/disable_torch_function.cpp` `has_torch_function_attr` / `check_has_torch_function` (L311–327): a Python attribute lookup per call; `torch/csrc/utils/python_arg_parser.cpp` `handle_torch_function_getter` (L176–186).
- `torch/csrc/autograd/python_function.cpp` `THPFunction_apply` (L1330–1446): three `getattr`s, ctx allocation, `PyNode`, `RECORD_FUNCTION`, `process_outputs`; forward runs under `AutoGradMode(false)`.
- `torch/nn/parameter.py` L104 (`nn.Parameter.__torch_function__ = _disabled_torch_function_impl`).
- `torch/_ops.py` `OpOverloadPacket.__call__` (L1257–1269), `__getattr__` (L1203), `_OpNamespace.__getattr__` (L1364) [installed wheel].
- `torch/amp/grad_scaler.py` `_maybe_opt_step` (L363–373; `.item()` at L371) [installed wheel].
- `torch/cuda/__init__.py` `set_sync_debug_mode` (L1246) and `c10/cuda/CUDAFunctions.cpp` `warn_or_error_on_sync` (L158).
- `aten/src/ATen/native/cuda/Copy.cu` `copy_kernel_cuda` (L371–480): non-blocking branch L457–476, blocking branch `memcpy_and_sync` L478.
- `aten/src/ATen/native/native_functions.yaml` L8089–8093 (`_local_scalar_dense` dispatch).
- `docs/source/notes/cuda.rst`: sync semantics (L283–300), pinned memory (L1235–1262), graph capture constraints (L1400–1425, banned for us).
- `torch/nn/utils/clip_grad.py` (L87–96, `_foreach_norm` over device/dtype groups); `torch/nn/modules/module.py` `zero_grad(set_to_none=True)` (L2949) [installed wheel].
- bitsandbytes 0.50.2 (installed in the soup env): `bitsandbytes/nn/modules.py` `Params4bit` (L213–484) and `Linear4bit.forward` (L609–637); `bitsandbytes/autograd/_functions.py` `MatMul4Bit` (L300–398) and `matmul_4bit` (L407–491).
- peft 0.20.0 (installed): `peft/tuners/lora/bnb.py` `Linear4bit.forward` (L513–560; defensive `result.clone()` at L533).
- transformers 5.16.1 (installed): `trainer.py` L1784 (`floating_point_ops` per micro-step), L1805 (`model.zero_grad()`), L1930 (`model.train()`), L2548 (`accelerator.clip_grad_norm_(model.parameters())`).
- Triton 3.6.0 (installed): `triton/compiler/compiler.py` `CompiledKernel.__getitem__` runner (L474–500).
- Unsloth `unsloth/kernels/fast_lora.py` at `main` = d8efd157 (2026-10-31): `LoRA_MLP` (L435–557), one autograd Function for the whole gate/up/SwiGLU/down block. Pattern only; no code copied.
- modded-nanoGPT README, https://github.com/KellerJordan/modded-nanogpt: grep only (L48 CUDA-graph capture of the training step, L98 compile latency, L372 `@torch.compile`).
- Organizer rulings in this repo: `docs/RULINGS.md` L14 (in-process thread for Triton compile allowed), L19–20 (no caches between runs), L29 (helper process banned); `harness/common.py` L19–24 (`BANNED_MODULES`).

**Not found / not used:** dev-discuss.pytorch.org posts on dispatcher or `__torch_function__` cost (the search returned none). `docs/source/notes/autograd.rst` was downloaded but not used for any number. Two search hits were not read in full and are [S] only: pytorch/ao issue 1930 (eager dispatch longer than an int4 kernel) and arXiv 2509.07003 (veScale: per-op dispatch bounds MoE runtime). Nothing above depends on them.

**Measured here** (`docs/notes/r1-bench/`): WSL2 Ubuntu, conda env `soup` (torch 2.11.0+cu130, triton 3.6.0, bitsandbytes 0.50.2), CPU AMD Ryzen 7 260 (16 logical), GPU RTX 5060 Laptop, driver 582.05. This is **not the Kaggle T4 host**; the Kaggle CPU is slower per thread (STATUS.md:268). Numbers are per-call **enqueue** times (median of 7); the loop ends before the GPU sync and `n` stays below the launch queue. Where the GPU is the bottleneck the enqueue number is an upper bound (marked).

## 3. Table 1: host cost per operation

| Operation | Measured here (µs) | Other numbers | Mechanism, source |
|---|---|---|---|
| Attribute read, plain `Tensor.shape` | 0.06 | — | getter has no torch-function hit (python_variable.cpp:2991) [V] |
| Attribute read, `nn.Parameter.shape` | 0.08 | — | `__torch_function__ = _disabled_torch_function_impl` (parameter.py:104) [V] |
| Attribute read, bnb `Params4bit`: `.shape`, `.requires_grad`, `.dtype`, `.is_cuda`, `.ndim`, `.grad` | 0.85–1.0 | `.data`: 1.17, and it returns a plain `Tensor` with no `quant_state` | getter → `handle_torch_function_getter` (python_arg_parser.cpp:176) → `Params4bit.__torch_function__` (modules.py:446) → `Tensor.__torch_function__` (_tensor.py:1682) [V] |
| Eager CPU op, `add_` on 16 elements | 1.9 (method), 2.4 (`torch.ops.aten` chain) | PyTorch 280 k ops/s ≈ 3.6 µs per op (Horace He, brrr_intro) [V] | `OpOverloadPacket.__call__` → `self._op` (_ops.py:1257); on CUDA the `torch.ops.aten.X.Y` chain costed 0.6 µs more per call than a pre-bound overload (7.6 vs 7.1 µs) [V measured] |
| Eager CUDA op, `add_` on 16 elements | 7.5 (method), 7.1 (pre-bound), 7.6 (chain) | — | host cost of dispatch plus the driver launch [V measured] |
| CUDA allocation without launch (`torch.empty`) | 1.3 | — | caching allocator path [V measured] |
| Triton JIT launch, trivial 1-CTA kernel | 9.9 and 13.3 (two runs) | our T4 launcher bench: 19.9 (`-O3`), 19.1 (`-O0`) per launch, 20 000 launches (STATUS.md:96) | runner passes all args to `self.run` (compiler.py:490–500) [V] |
| cuBLAS `addmm`, fp16 with bias, M=1 / M=832 | 19.4 / 13.4 | — | enqueue only [V measured] |
| bnb `Linear4bit` forward, 1536→1536, M=1 | ~38 (no_grad) | — | `Linear4bit.forward` (modules.py:609–637) [V measured] |
| bnb `Linear4bit` forward, M=832 | 93 (no_grad), 124 (grad on) | — | grad on goes through `MatMul4Bit.apply` (_functions.py:491; class L300) [V measured; upper bound, queue back-pressure possible] |
| peft LoRA module eager: clone, dropout, A, B, scale, add (M=832) | 89–96 | — | `result = result.clone()` (peft bnb.py:533) plus the other eager ops of the forward [V measured] |
| `autograd.Function.apply`, identity, grad on / no_grad | 3.9 / 2.8 | native `view_as` under grad: 1.0 | `THPFunction_apply` (python_function.cpp:1330–1446) [V measured] |
| Python saved-tensor hook (pack + unpack, identity) | +4.8 per saved tensor (4.74 → 9.57 per op) | — | `saved_tensors_hooks` [V measured] |
| `cuda.Event.record()` / `wait_event` (done) / `with torch.cuda.stream(s)` | 3.8 / 0.6 / 6.2 | — | event and stream path, not read in source [V measured] |
| `.item()` on an idle GPU (round trip) | 33 | a busy queue adds the drain time (not timed here) | `_local_scalar_dense` is `_local_scalar_dense_cuda` (native_functions.yaml:8089–8093) [V] |
| `_foreach_add_` over 450 tensors / Python loop over the same 450 | 62–70 / 4 420 per pass (≈ 63×) | — | `_foreach_norm` grouping in clip_grad.py:87–96 [V measured] |
| Blocking H2D/D2H copy (`non_blocking=False`) | not timed | — | `memcpy_and_sync` (Copy.cu:478) synchronises, by its name [I]; a non-blocking copy from pinned memory is `cudaMemcpyAsync` plus a host-allocator event (Copy.cu:457–476) [V] |
| Host waits and walks in the step | — | ~29 host waits per step (RESEARCH.md:87); ~40 ms of idle GPU at the boundary in the trace (STATUS.md:73) | `found_inf` `.item()` is a wait (grad_scaler.py:371); `floating_point_ops` walks the parameters on the host each micro-step (trainer.py:1784) [V] |

`set_sync_debug_mode` (cuda/__init__.py:1246) does **not** give a full list of host waits: it only warns from sites that call `warn_or_error_on_sync` (CUDAFunctions.cpp:158), and I did not confirm that the `.item()` and copy paths call it. Use the profiler trace (the team's `trace_gaps_report` tool) for host waits.

## 4. Calibration: where the host time goes

- **Launch arithmetic.** ~65–81 launches per layer: 65 in the team's count; 24 Triton, 17 GEMMs and ~40 other ATen ops in the t12 notes (shared notes folder, `t12-attention-padding.md`, line 70). At ~9 µs average that is ~0.6–0.7 ms per layer, about 12 % of the 5.4 ms layer measured on the laptop (STATUS.md:268).
- **Plan-building.** The team's t12 notes (line 69): `_attention_plan` 38 µs and `_mlp_plan` 19 µs per layer, mostly `Params4bit` attribute reads.
- **bnb-only floor** (`bench_bnb_stack.py`, M=1, 28 layers, Qwen2.5-1.5B shapes, backward to the input): forward 0.56 ms, backward 0.98 ms per layer, **1.54 ms per layer** in total. At M=1 the GPU wall is 78 ms per step, so the queue may have back-pressured: read this as an upper bound.
- **M=832 is GPU-bound on this laptop** (GPU wall 226 ms per step against an enqueue of 127 ms; the enqueue split varied between runs from 184 to 127 ms), so the M=832 CPU numbers are not a floor.
- **Gap:** 5.4 − 1.5 ≈ 3.9 ms per layer, ~110 ms per step on the laptop. The known items add up to a fraction of that: saved-tensor hooks (~5 µs per saved tensor), plan-building (~57 µs per layer), ~40 `Params4bit` reads per layer (~40 µs), event pairs (~4 µs each), stream contexts (~6 µs each). **Not explained yet; this is the first measurement to make:** cProfile of one layer's forward and backward on the laptop, then py-spy in the Kaggle run.
- **[I] Scaling to the T4 host.** The laptop layer loop alone is 5.4 ms × 28 ≈ 151 ms per step; the team's measured CPU floor is 0.30 s per step (STATUS.md:123). The ratio of ~2 would be the host factor if the rest is the same work. Plausible, unmeasured.
- **[I] The boundary.** The team's estimate of 12–15 ms (RESEARCH.md, STATUS.md) is more than the attribute reads: 1,180 reads × 0.9 µs ≈ 1.1 ms on the laptop. The rest is the four walks and the waits. This fits the team's measurement "in the noise" for `night-boundary` (STATUS.md, branch table).

## 5. Table 2: techniques

Saving figures are for one step at our shapes, laptop-equivalent, from the Table 1 costs. Bit-identity risk is against the current path.

| # | Technique | Expected saving per step | Bit-identity risk | Where the cost is | Status |
|---|---|---|---|---|---|
| 1 | Profile one layer fwd+bwd (cProfile here, py-spy on Kaggle) before any change | the 3.9 ms/layer gap is the target | none | — | allowed (measurement) |
| 2 | Bind ops and streams once per run (pre-bound aten overloads, reused stream and event objects) | 0.6 µs per aten call, 6 µs per stream context: ~1–2 ms [I] | none | _ops.py:1203, 1364 | allowed |
| 3 | Group CUDA events (one per layer group, not per matrix) | 3.8 µs record + 0.6 µs wait per event; ~1 ms if the count halves [I] | none (same ordering) | weight-ring hooks | allowed |
| 4 | Pinned + `non_blocking` for every host copy in the step; no pageable copies | removes waits; the trace shows ~40 ms GPU idle at the boundary (STATUS.md:73) | none | Copy.cu:457 vs :478 | allowed |
| 5 | Defer logging `.item()` reads by one step (RESEARCH #1) | ~2.5–5 ms of drain per boundary (STATUS.md:73) | none | trainer.py:1857, 2101; fused_ce.py:161 (team's citation) | allowed |
| 6 | Build the parameter and module lists once per run; no `Params4bit` attribute reads in the boundary | ~1.1 ms plus walk overhead (boundary, measured in noise) | none | _tensor.py:1682; trainer.py:1805, 1930, 2548 | allowed (within-run cache only) |
| 7 | One autograd Function per MLP or attention block, not per linear (Unsloth `LoRA_MLP` pattern) | 5 fewer applies per layer × 3–4 µs ≈ 15–20 µs per layer, ~0.5 ms per step, plus fewer saved tensors [I] | low: same kernels in the same order | python_function.cpp:1330–1446 | allowed (pattern only) |
| 8 | Fewer saved tensors (recompute cheap ones; save only what backward reads) | ~5 µs per saved tensor on the laptop; our pack bodies are heavier, so more [I] | medium: changes what is recomputed; the team's attention recompute is bitwise (RESEARCH.md) | saved_tensors_hooks | allowed |
| 9 | Launch our Triton kernels through a cached runner, skipping the JIT dispatch (measure first; the runner is compiler.py:490–500; not benchmarked, my direct call failed on argument count) | JIT share (not measured; the total is 10–13 µs per launch) × 672 launches: 2–4 ms if the share is 3–6 µs [I] | none (same kernel) | compiler.py:474–500 | allowed if our kernel, not a graph; pin Triton 3.6.0 |
| 10 | `_foreach` for clip-norm, `zero_grad`, scaling and any remaining per-tensor loops | 4.4 ms → 0.06 ms per 450-tensor pass; ~0.5–1 ms [I] | low for `add_`/scale; norms change reduction order | clip_grad.py:87–96; module.py:2949 | allowed |
| 11 | Remove every bnb `Linear4bit` call from hot paths (our own dequant + GEMM already exist in `solution/`) | 70–120 µs per call: up to ~14 ms if a path still goes through bnb [I] | none if the same dequant and GEMM | modules.py:609–637; _functions.py:300–491 | allowed |
| 12 | Own LoRA forward: drop `result.clone()` and the generic loops (no in-place write to `result`) | ~20–30 µs per module of the 89–96 µs measured; 56 modules: ~1.1–1.7 ms [I] | none | peft bnb.py:533 | allowed |
| 13 | Prefetch the next batch (collate, plan) in the shadow of backward on a Python thread | DataLoader `__next__` 2.8 ms (RESEARCH.md:38); the compile thread's GIL cost ~0.3 s on early steps (STATUS.md, int8 note) | none | — | **grey**: the rulings allow an in-process thread for compile only; ask |
| 14 | Remove the GradScaler `found_inf` sync | one drain per step | **not bit-identical** if done with multiplicative masks (NaN × 0 = NaN) | grad_scaler.py:371 | grey; leave |
| 15 | CUDA graphs, `make_graphed_callables`, `torch.compile` on our kernels | large | — | cuda.rst:1400–1425 (capture constraints); modded-nanoGPT README L48 | **banned** for our kernels |
| 16 | Helper or any background process | — | — | RULINGS.md:29 | **banned** |
| 17 | Banned libraries, or copying their code (the design pattern in row 7 is fine) | — | — | harness/common.py:19–24 | **banned** |
| 18 | `gc.freeze()` / `gc.disable()` in the loop | in the noise (team's measurement) | none | — | allowed, not worth it |

## 6. Allowed, grey, banned

- **Allowed:** rows 1–12 and 18. All are bit-identical or verifiable by the team's bitwise tests, and none is a graph, a compile or a helper.
- **Grey:** row 13 (batch-prep thread: RULINGS.md:14 covers Triton compile in a thread, not batch preparation; ask the organizers before building it) and row 14 (cannot be bit-identical; leave it).
- **Banned:** rows 15–17, plus caches or pre-computed tensors between runs (RULINGS.md:20).

## 7. Ten most valuable changes, ranked

1. **Profile one layer fwd+bwd** (cProfile on the laptop, py-spy on Kaggle) to name the 3.9 ms per layer that is neither dispatch nor launches. Horace He's first rule applies: find the regime before changing code. https://horace.io/brrr_intro.html
2. **Take the host waits out of the step body and the boundary.** Defer the logging `.item()` reads and keep every host copy pinned with `non_blocking`; each wait drains the GPU queue, and the trace shows ~40 ms of idle GPU at the boundary. https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/cuda.rst
3. **One autograd Function per MLP** (Unsloth `LoRA_MLP` pattern) with fewer saved tensors: each apply costs 3–4 µs and each saved tensor about 5 µs of hooks. https://github.com/unslothai/unsloth/blob/d8efd157abbb2f160fb55c77a3044d83955949f4/unsloth/kernels/fast_lora.py
4. **Remove every bnb `Linear4bit` call from hot paths.** The laptop measures 70–120 µs of host time per call; our own dequant and GEMM do the same work with less Python. https://github.com/bitsandbytes-foundation/bitsandbytes
5. **Launch our Triton kernels through a cached runner** (measure first; the runner is at compiler.py:490–500). The JIT share of the 10–13 µs per launch is the saving, with 672 launches per step. https://github.com/triton-lang/triton
6. **Build the boundary's parameter and module lists once per run.** Each `Params4bit` attribute read costs 0.9 µs through a Python `__torch_function__`, and the boundary repeats the walks. https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/utils/disable_torch_function.cpp
7. **`_foreach` for clip-norm and `zero_grad`.** 450 tensors take 4.4 ms as a loop and 0.06 ms with `_foreach_add_`. https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/utils/clip_grad.py
8. **Group CUDA events** in the weight ring: one event per layer group instead of per matrix; each record is 3.8 µs. https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/cuda.rst
9. **Bind aten overloads and stream objects once per run** (0.6 µs and 6 µs per call). https://github.com/pytorch/pytorch/blob/v2.11.0/torch/_ops.py
10. **Own LoRA forward without peft's `result.clone()`**: one copy and one launch saved per LoRA module, 56 per step. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/bnb.py

## 8. Corrections to existing notes

- `RESEARCH.md` ("~1,800 launches = 10–20 ms CPU per step") holds in order of magnitude: at our measured 8–13 µs per launch it is 15–25 ms. The 0.30 s floor is not launches.
- The boundary's "12–15 ms" (RESEARCH.md, STATUS.md) is mostly the walks and the waits, not the attribute reads (~1 ms).
- `set_sync_debug_mode` is not a full sync finder (§3). Use the trace.

## 9. Open items

- Profile one layer (row 1). Without it, rows 2–12 are estimates.
- Test row 9's runner path; my direct call failed on argument count (`TypeError: 16 arguments expected`).
- Count saved tensors per layer in `solution/` (row 8).
- Ask the organizers about a batch-prep thread (row 13).
- Reproduce on a Kaggle T4 run before any number here goes into the README: these are laptop numbers.
