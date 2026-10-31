# R5: CUDA graphs for our step (research, 31 Oct 2026)

Question: can CUDA graphs buy us time in the step, and how? No code in `solution/`.

Evidence tags: **[S]** PyTorch source, tag `v2.11.0` (fetched from GitHub; paths relative to repo root). **[D]** PyTorch docs `docs/source/notes/cuda.rst` at `v2.11.0`. **[N]** NVIDIA troubleshooting guide (docs.nvidia.com/dl-cuda-graph). **[W]** web search snippets, not opened unless stated. **[M]** our measurement, local RTX 5060 Laptop GPU (sm_120), torch `2.12.0.dev20260312+cu128`, Windows Python 3.10. **That is not T4 and not 2.11**; use it for ratios and mechanisms, not for absolute numbers on the jury's T4. Scripts: `docs/notes/r5-bench/`.

## 0. Verdict

- **Graphs remove CPU launch cost only.** Our per-step GPU time is unchanged. Only short steps (under about 600 tokens, about 13 of 64) are CPU-bound, so the gain is capped by that share.
- **Upper bound for a full-layer graph:** about 2.7 s per run if every layer's forward and backward on short steps became one replay. Net of capture and warm-up, about 0.9 s, before the engineering costs below. Realistic first prototype (head chunk, no dropout): about 0.1–0.25 s per run (estimate: 38 chunks × 13 short steps × 0.2–0.5 ms CPU per chunk; per-chunk CPU is not yet measured).
- **Three things can cost more than they save:** (a) warm-up and capture each run the layer eagerly, so each graph costs about 4 eager passes; (b) private memory pools for each graph bucket threaten the 371 MiB memory target; (c) dropout masks change if the captured tensor is padded (section 2.3). Dropout is the constraint that decides which pieces can be graphed at all.
- **Cheapest safe first candidate:** a dropout-free piece with fixed shape. Options: one vocabulary chunk of the head (38 repeats), or the `_MLP` block (LoRA is only on q and v, so MLP has no dropout). Measure per-chunk CPU first (section 6).
- **Correction to `docs/RESEARCH.md` ("Внешнее ревью 3", п. 3):** in 2.11, `torch.cuda.graph.__enter__` no longer calls `gc.collect()` by default. It is gated by `torch.compiler.config.force_cudagraph_gc`, default `False` [S: `torch/cuda/graphs.py:241-247`, `torch/compiler/config.py:133`]. What remains is `synchronize`, `empty_cache`, and `_host_emptyCache`.
- **Conflict to resolve in `CLAUDE.md` and `docs/HANDOVER.md`:** both list "CUDA graphs для своих ядер" as banned. Organizers' answer 46 (`docs/briefs/organizers-qa-2026-10-31.md:97`) allows graphs, with the 5 % caveat. The answer is newer, so the banned line should be updated. We did not edit it here.

## 1. What `torch.cuda.graph` does (Q1)

`graph.__enter__` [S: `torch/cuda/graphs.py:237-262`]:

1. `torch.cuda.synchronize()` (line 239): waits for all device work. In our pipeline that wait equals the CPU-ahead lag: 20–44 ms on long steps [C1 note, `docs/notes/c1-cpu-diet.md` step 3].
2. `gc.collect()`: only if `force_cudagraph_gc` (line 241, default off).
3. `torch.cuda.empty_cache()` (line 249): `_cuda_emptyCache` → `CUDACachingAllocator::emptyCache` [S: `c10/cuda/CUDACachingAllocator.cpp:2131-2135`]. With no capture running it calls `release_cached_blocks` on all pools, which cudaFrees every free block [S: lines 3443-3457]. Later allocations then cudaMalloc again.
4. `torch._C._host_emptyCache()` (line 251): frees free pinned blocks in the host caching allocator [S: `torch/csrc/cuda/Module.cpp:555-562`, `aten/src/ATen/core/CachingHostAllocator.h:453-471`]. A pinned block we allocate and drop is cudaFreeHost-ed here, and the next pinned allocation pays for cudaHostAlloc again.
5. `stream_ctx.__enter__()`, then `capture_begin` [S: `torch/cuda/graphs.py:255-262`].

`CUDAGraph.capture_begin` [S: `aten/src/ATen/cuda/CUDAGraph.cpp:60-124`]:
- Registers the default CUDA generator (lines 65-68), then calls `capture_prologue` on each registered generator state.
- Requires a non-default stream (lines 77-80).
- Calls `beginAllocateToPool` on the device and host allocators (lines 109-113). Allocations on the capturing stream go to a private pool.
- Calls `cudaStreamBeginCapture` with mode `global` by default [S: `torch/cuda/graphs.py:219`, `capture_error_mode: str = "global"`].

`capture_end` [S: lines 126-159]: `cudaStreamEndCapture`, `endAllocateToPool`, `capture_epilogue` for generators. Unless `keep_graph=True`, it instantiates the graph and destroys the raw `cudaGraph_t`.

`replay` [S: lines 192-209]: `replay_prologue` for each RNG generator (below), then `cudaGraphLaunch` on the current stream. A graph may be replayed on any stream [D: "Non-constraints", `cuda.rst:1437-1440`].

**Measured [M]** (72-launch block = 12 layers × {layer_norm, linear, dropout, silu, linear, add}; N=64, D=256 for the CPU-bound config):

| item | value |
|---|---|
| raw `capture_begin`/`capture_end`, 72 launches | 1.2–2.7 ms (medians 1.4–1.7) |
| `with torch.cuda.graph`, steady state | 2.2–2.9 ms (≈0.5 ms above raw) |
| first capture in a process (lazy init) | 7–11 ms |
| `__enter__` extras, first call after work | sync 44 ms, `empty_cache` 52 ms (queue drain plus cudaFree) |
| `__enter__` extras, steady state | sync 0.08 ms, `empty_cache` 0.01 ms, host empty 0.00 ms |
| capture cost vs size (min of 3) | 6 launches 1.0 ms, 12: 1.5 ms, 24: 1.7 ms, 72: 3.8 ms |

Capture costs about one eager pass over the region plus fixed overhead. Our `_QKVRope`/`_MLP` launches run Python on every eager call, so capturing a layer costs roughly one layer's Python time (C1: about 4 ms forward and 4.2 ms backward per layer on T4).

Reading: **avoid `torch.cuda.graph` in the step and use `CUDAGraph.capture_begin/capture_end` inside our own `torch.cuda.stream(...)` block.** That skips the drain and the pinned free. Keep the rest of the contract: non-default stream, warm-up on a side stream, no non-captured CUDA work on any thread during capture.

## 2. Dropout under capture (Q2)

### 2.1 Mechanism [S]

- Dropout calls `gen->philox_cuda_state(counter_offset)` [S: `aten/src/ATen/native/cuda/Dropout.cu:398-403`]. The increment is `((nelem-1)/(block_size*grid.x*UNROLL)+1)*UNROLL` (line 398), so it depends on the element count. The generic grid policy caps `grid.x` at SMs × blocks-per-SM [S: `aten/src/ATen/native/cuda/DistributionTemplates.h:49-60`]. The kernel draws from `curand_init(seed, idx, offset)` [S: `Dropout.cu:146-150`].
- **Eager** (`CUDAGeneratorImpl.cpp:467-480`): returns `PhiloxCudaState(seed_, philox_offset_per_thread_)`, then `increase(increment)` adds to the host counter (lines 95-127).
- **Under capture** (same function, first branch): returns device pointers `seed_extragraph_` and `offset_extragraph_`, and the intragraph offset. The intragraph offset starts at 0 in `capture_prologue` (lines 199-204) and grows with each call (lines 101-114). Host counter untouched.
- **Replay** (`replay_prologue`, lines 219-230): fills the two device tensors with the current host seed and offset, then `increase(total)` on the host. Two `fill_` kernels run per replay (outside the graph). Our replay issue time includes them.
- The generator must be registered before capture. The default generator is registered automatically (`CUDAGraph.cpp:65-68`). Any other `torch.Generator` must be registered with `register_generator_state` [D: `cuda.rst:1419-1420`].

**Consequence:** a captured dropout reads the same seed and an offset equal to the eager offset at the same point, so the mask should be bit-identical to eager.

### 2.2 Measured [M]

`docs/notes/r5-bench/graph_bench.py`, block with 12 dropouts (p=0.05, fp16, N=512, D=1536):
- Two eager runs with the same `manual_seed`: identical.
- Consecutive eager calls differ: dropout is active.
- Replay vs eager, 6 of 6 calls: `torch.equal` True, max abs diff 0.0.
- Eager and replay interleaved (replay on odd calls, eager on even): 6 of 6 equal to pure eager.

Caveat: one GPU, one dtype, torch 2.12.0.dev, not 2.11.0.

### 2.3 Traps for our base-equivalence requirement

1. **Padding changes masks even when the padded tensor's real rows match.** The counter increment depends on `nelem` (Dropout.cu:398). `docs/notes/r5-bench/pad_mask_test.py` measured it: the padded dropout's real rows equal the base mask, but the *next* dropout does not (padded 300→320, 384, 512: next mask differs; 300→300 equal). Organizers' answer 27 allows padding to 16/128/256, but dropout must stay on exact shapes or the offsets diverge. **Dropout must not run on padded tensors** unless we consume the same increments as the base.
2. **`make_graphed_callables` warm-up consumes RNG.** Its `num_warmup_iters` (default 3) run eagerly [S: `torch/cuda/graphs.py:325-326, 425-452`]. Eager calls advance the host offset. Capture does not. To keep base offsets, save with `torch.cuda.get_rng_state()` before warm-up and restore with `set_rng_state()` after capture. This follows from the source (`CUDAGeneratorImpl.cpp:122-125`) and was not tested.
3. **Replays use only the device seed/offset at replay time.** Changing the seed does not change a captured mask unless the device tensors are refilled, which `replay_prologue` does. So reseeding between runs is safe.

Verdict: **graphs with dropout inside are not base-equivalent unless the tensor shape and offset sequence match exactly.** The first candidates must be dropout-free (MLP, head chunk), or keep the q/v LoRA dropout eager and graph only what surrounds it.

## 3. Variable N, graph count, pools (Q3)

- Captured shapes are fixed [D: `cuda.rst:1433-1434`]. Buckets need padding (organizers' answer 27 allows multiples of 16, 128, or 256). Padding is safe for MLP and head (no dropout), not for dropout (2.3).
- **Graph count:** per callable and per bucket, one forward and one backward graph. Capturing 28 layers × 2 buckets (256 and 512, the only buckets that matter below 600 tokens) gives about 112 graphs.
- **Pools:** each `CUDAGraph` gets a private pool unless `pool=` is passed [S: `CUDAGraph.cpp:85-96`]. A pool lives as long as its graph and its captured tensors [D: `cuda.rst:1682-1693`]. Passing a pool shares memory, but only if graphs replay in the same order they were captured [D: `cuda.rst:1706-1708`]. Within one pool, memory freed during capture is reused [S: `CUDACachingAllocator.cpp:103-133` note]. So one pool per bucket, with layers captured in step order, costs about one layer's high-water per bucket.
- **Memory estimate (not measured):** an MLP intermediate at N=512 is 512 × 8960 × 2 B ≈ 9.2 MB per tensor. One layer's high-water might be 20–30 MB per bucket. Two buckets ≈ 40–60 MB, which is 10–15 % of the 371 MiB budget. Measure before adopting.

## 4. Side streams, cuBLAS, Triton, autograd (Q4)

- **Multi-stream rule** [D: `cuda.rst:1615-1638`]: fork from the capture stream with `wait_stream`, rejoin with `wait_stream` before `capture_end`. NVIDIA codes: 904 unjoined, 905 isolation ("captured stream depends on uncaptured work") [N: `capture-failures.html`].
  - **Trap for our streaming:** an event recorded outside capture and waited inside gives 905. Events used inside a graph must be recorded in the same capture.
  - Join every side stream explicitly before `capture_end`. Do not rely on the allocator's own stream bookkeeping.
- **H2D from pinned memory:** capturable. The graph bakes host and device addresses, so host slots must not move, and the CPU must not overwrite a slot between its copy in the graph and the replay that reads it. Ring refills must happen between replays, outside the graph.
- **Pinned allocation during capture:** forbidden in `global` mode (900), and the pinned allocator's `cudaEventQuery` in `allocate` can fail capture too [N: `capture-failures.html`, "Pinned Memory"]. All pinned buffers must exist before capture. Our ring and offload buffers already do.
- **`_host_emptyCache` inside capture:** `cudaFreeHost` is forbidden in `global` mode [N]. `__enter__` runs it before `capture_begin`, so it is safe but costly (see 1).
- **cuBLAS workspace:** keyed by (handle, stream) and allocated on first use from the caching allocator [S: `aten/src/ATen/cuda/CublasHandlePool.cpp:241-243, 249-279`]. A new capture stream, or the first GEMM on a new stream during capture, allocates into the pool. **Warm every GEMM on the capture stream before capture.** `torch._int_mm` / cuBLASLt workspace: not checked.
- **Triton 3.6:** `JITFunction.run` reads the current stream and launches through the C launcher each call [S: `python/triton/runtime/jit.py:695-746`, v3.6.0]. The launch should be captured. Replays skip the Python `run` wrapper, so the harness's launch counter will not see them, as organizers said. **Warm every kernel and specialization before capture** (compile and module load outside capture). Not verified in source.
- **Autograd / `make_graphed_callables`:** the wrapper copies inputs into static buffers when pointers differ and replays forward and backward graphs [S: `torch/cuda/graphs.py:527-557`]. Constraints [S docstring, lines 333-364]: `requires_grad` in `sample_args` must match real inputs; module buffers must not be trainable; no double backward; no hooks registered before wrapping; `sample_args` must be tensors only; autocast caching disabled. Backward is captured with `torch.autograd.grad` [S: lines 486-493]. C1 notes say backward runs on autograd's device thread. In `global` mode, another thread's CUDA call during capture errors; `thread_local` only if that thread does no CUDA work during capture.
- **Stream propagation** [D: `cuda.rst:1618`]: capture mode propagates to streams that sync with the capture stream.

## 5. Numbers (Q5)

Laptop, 72 launches (`graph_bench2.py`, median of 5 trials):

| config | eager issue | eager GPU | replay issue | replay GPU |
|---|---|---|---|---|
| N=64, D=256 (CPU-bound) | 1.7–2.1 ms/block, 24–30 µs/launch | ≈ same as issue | 0.09–0.10 ms, 1.3 µs/launch | 0.14–0.16 ms |
| N=512, D=1536 (GPU-bound) | 2.1 ms issue | 2.2–2.4 ms | 0.66 ms issue | 2.35 ms |

- CPU-bound: replay saves about 95 % of CPU per block. GPU time is the floor (about 2 µs per tiny node).
- GPU-bound: no gain, as expected. One eager event-timed run read 14 ms against 2.3 ms wall, which we could not explain; laptop thermal or power state is suspected. Treat with caution.
- T4 numbers: none. C1 gives 4.0 ms CPU per layer forward and 4.2 ms backward on short steps, about 57 µs per launch including our Python.
- Upper bound on savings: 28 layers × (4.0 + 4.2 − ≈0.3) ms ≈ 0.2 s per short step; × 13 short steps ≈ 2.7 s. Capture and warm-up: about 32 ms per layer-bucket (three eager passes plus one capture, using C1's 8 ms per layer pair). 56 layer-buckets ≈ 1.8 s. **Net ≈ 0.9 s**, before memory and sync costs.

Public data: PyTorch's 2021 blog reports 1.70× for Mask R-CNN and 1.12× for BERT at 4096 GPUs, both for graphs over most of a model, and says graphs help when CPU overhead is significant [W/fetched: pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/]. NVIDIA's performance guide says the same [W]. A forum thread reports a 12-layer model running about 70 µs slower with graphs in one configuration [W, snippet only]. **No public measurement of CUDA graphs on LoRA or QLoRA training was found** (WebSearch).

## 6. First prototype (Q6)

**Candidate A (cheapest): one vocabulary chunk of the head.**
- Dropout-free. The head's H2D chunk copy and decode stay outside the graph. The graph covers GEMM, online max/sum, and loss.
- 38 repeats per step, fixed shape; the last partial chunk stays eager (as the brief says).
- Ring slots: capture one graph per slot (2–3 graphs), with pre-allocated pinned and device buffers.
- Gate: measure the head's per-chunk CPU time with cProfile or the CPU-profile switch (`SOUP_SOLUTION_CPU_PROFILE=1`). If below about 0.4 ms per chunk, the gain is under 0.25 s and not worth it.
- Expected gain: about 38 × (per-chunk CPU − 0.1 ms) on short steps only. The head is PCIe-bound on long steps, so no gain there.

**Candidate B: `_MLP` forward+backward on a fixed bucket.** No LoRA on MLP, so no dropout. Replay saves about 0.6 ms per layer forward and about 0.6 ms backward (C1 sampled `_MLP.forward` at 18 ms per step). Bigger effort: layer weights must sit in static ring slots, norms must be copied into static buffers, and captures must match the streaming order.

**Break-even criterion:** replays × Δt_replay > t_warm + t_capture + extra copies. For a graph of one layer-bucket: t_warm + t_capture ≈ 32 ms (see 5). Each short-step replay saves about 7.5 ms (fwd + bwd, C1). **A layer-bucket needs at least about 5 replays to pay off.** On 13 short steps split over two buckets, each layer-bucket has about 6–7 replays: marginal. Memory and sync costs push it further negative.

**Checklist (traps):**
- Use `CUDAGraph` directly, not `torch.cuda.graph`, to avoid the drain and the pinned free.
- No dropout inside any graph on padded or bucketed shapes.
- Warm all GEMMs (cuBLAS workspace) and all Triton kernels on the capture stream before capture.
- All pinned buffers exist before capture. No event recorded outside capture and waited inside.
- Save RNG state before warm-up and restore after capture if dropout is ever inside.
- Keep `global` capture mode; `thread_local` only after checking the autograd thread.
- Count memory with `torch.cuda.memory_reserved()` and nvidia-smi before and after each capture.
- Replays and capture must happen in the same order each step (pool sharing).

## 7. What to measure on T4 (next session)

1. Per-chunk CPU of the head's chunk loop (cProfile, sync-off).
2. One head-chunk graph: capture time, replay time, mask-free equality with eager, memory delta. Compare replay vs eager GPU time per chunk.
3. The padded-dropout offset test on T4 (`pad_mask_test.py`).
4. Only if (1) shows ≥0.4 ms per chunk: a layer-MLP graph for bucket 512 with `pool=` shared across layers.

## Sources

- PyTorch v2.11.0 source: `torch/cuda/graphs.py` (lines cited above), `aten/src/ATen/cuda/CUDAGraph.cpp`, `aten/src/ATen/cuda/CUDAGraph.h`, `aten/src/ATen/cuda/CUDAGeneratorImpl.cpp`, `aten/src/ATen/cuda/CublasHandlePool.cpp`, `aten/src/ATen/native/cuda/Dropout.cu`, `aten/src/ATen/native/cuda/DistributionTemplates.h`, `c10/cuda/CUDACachingAllocator.cpp`, `aten/src/ATen/core/CachingHostAllocator.h`, `torch/csrc/cuda/Module.cpp`, `torch/compiler/config.py`, `torch/cuda/memory.py`, `torch/cuda/__init__.py`. Base URL: `https://github.com/pytorch/pytorch/blob/v2.11.0/`.
- PyTorch docs `docs/source/notes/cuda.rst` at v2.11.0: [CUDA Graphs](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/cuda.rst) (constraints 1402-1435, partial capture 1503-1523, multi-stream 1615-1638, graph memory 1682-1708).
- NVIDIA: [Capture Failures](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/capture-failures.html) (opened).
- PyTorch blog: [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) (opened).
- Triton v3.6.0: [`python/triton/runtime/jit.py`](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/runtime/jit.py) lines 695-746.
- [W, snippets only, not opened]: vLLM PR #10059 (gc/empty_cache patched for non-first graphs); PyTorch forum thread on NCCL gaps with graphs (the 70 µs anecdote); TransformerEngine PR #2092 (gc during capture).
- Local measurement: `docs/notes/r5-bench/graph_bench.py`, `graph_bench2.py`, `pad_mask_test.py`.
