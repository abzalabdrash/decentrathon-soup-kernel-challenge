# G1: reusable CUDA graph template of the MLP (branch `claude/g1-mlp-graph-6c3983`, from main 834e5ad)

Task: one graph template of the fused MLP (`fused_proj._MLP`: NF4 dequantization into the shared
buffer, gate|up GEMMs, P1's SwiGLU + int8 form, down GEMM; backward `int8_bwd.mlp_backward` on the int8
form), captured once on static buffers and the type ring's fixed slots, replayed for all 28 layers, ring
acquire/release, offload and dropout outside. Order: micro-bench on the T4, then integration behind
`SOUP_SOLUTION_MLP_GRAPH`, ABBA, backward.

**Verdict: NO-GO before integration.** Measured on a T4: the template removes real CPU time, **0.29 ms
per layer forward and 0.87 ms backward** (53 % of the autograd node's 0.80 + 1.30 ms), GPU time about the
same as eager (backward +0.1 ms for the input copies). But (1) the static buffers and the private pool of a
forward + backward pair at bucket 512 keep **+52 MiB allocated / +82 MiB reserved** resident for the whole
run, four times the +20 MiB budget, and the peak (long steps, the head) gains all of it; (2) only steps
whose tokens fit the bucket and are CPU-bound turn the 32.5 ms per step (28 x 1.16 ms) into wall time:
about 6 of 62 public steps are <= 512 tokens (C1's table), about 13 <= 640, so the ceiling is ~0.2-0.3 s
with bucket 512 and ~0.4-0.6 s with bucket 640 at ~+100 MiB reserved (the upper ends if Kaggle's CPU is
1.5x slower than Modal's), against the GO bar of -0.5 s at +0..20 MiB; (3) the harness makes capture
during timed steps impossible (trap 1), and the int8 paths only start at step 8. Nothing was integrated;
`solution/bench_mlp_graph.py` is the only code (measurement only, not used by apply()).

## Measurements

`python -m solution.bench_mlp_graph` (bench mode). Per layer, Qwen2.5-1.5B MLP shapes (1536 / 8960), NF4
weights resident (the ring's slots behave the same for the GPU), int8 backward and int8 (gate|up) form on;
the template's dequantizations, GEMMs and Triton kernels are exactly the eager path's (`_gate_up_matmul`,
`swiglu_quant`, `_weights`, `int8_bwd.mlp_backward`), on static buffers; the replay column includes what
stays eager around a replay (input copy, zeroing of the padding rows, a per-layer constants copy, output
copy; backward also the int8 (gate|up) copied into the static buffer).

### Laptop (RTX 5060, WSL, torch 2.11.0+cu130; CPU faster than Kaggle's)

| N | B | fwd CPU eager | replay | saved | fwd GPU eager | replay | bwd CPU eager | replay | saved | bwd GPU eager | replay | capture f/b ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 296 | 512 | 0.250 | 0.104 | 0.146 | 1.271 | 1.655 | 0.723 | 0.081 | 0.642 | 1.004 | 0.920 | 3.11 / 2.81 (first) |
| 512 | 512 | 0.296 | 0.072 | 0.224 | 1.647 | 1.659 | 0.650 | 0.073 | 0.577 | 1.021 | 0.942 | 0.76 / 2.18 |

The full autograd node (`h = h + mlp(h)`, offload context with the early int8 copy, int8 backward) at
416 tokens: forward 0.675 ms, backward 1.093 ms of CPU per layer. The template removes ~45 % of it.
Replay is bitwise equal to eager when B equals the eager padded N (512); at 296 -> 512 the forward output
differs in low bits (cuBLAS picks other kernels for M = 296 and 512), the int8 form of (gate|up) matches.
Memory, forward + backward sharing one pool at bucket 512: static 29.3 + 10.6 MiB, allocated +52.3 MiB,
reserved +82.0 MiB (the backward's intermediates: int32 product 18.4 MiB, int8 gradient 9.2 MiB, and the
capture stream's own cuBLAS workspace).

### T4 (Modal soup-g1-mbench1, e7c2666: Tesla T4, PCIe x8 host, CPU 3.1 GHz, Python loop of 1e7 additions 1.34 s)

All six Kaggle slots were taken by I1, P2 and T1 for the whole window, so the bench ran on a Modal T4 (the
queued Kaggle run soup-g1-bench1 was withdrawn before it started). GPU and memory numbers are the T4's;
CPU numbers are Modal's CPU (Kaggle's is likely slower, which scales the savings up, not the verdict).

| N | B | fwd CPU eager | replay | saved | fwd GPU eager | replay | bwd CPU eager | replay | saved | bwd GPU eager | replay | capture f/b ms | static+pool MiB f/b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 296 | 512 | 0.367 | 0.065 | 0.302 | 1.777 | 1.629 | 0.953 | 0.086 | 0.866 | 1.597 | 1.724 | 1.33 / 13.73 (first) | 29.5 / 13.6 |
| 416 | 512 | 0.345 | 0.065 | 0.280 | 1.587 | 1.638 | 0.943 | 0.086 | 0.857 | 1.523 | 1.723 | 0.85 / 2.05 | 29.5 / 13.6 |
| 512 | 512 | 0.351 | 0.052 | 0.298 | 1.624 | 1.664 | 0.950 | 0.071 | 0.879 | 1.627 | 1.724 | 1.06 / 2.13 | 29.5 / 13.6 |
| 640 | 640 | 0.348 | 0.052 | 0.296 | 2.092 | 2.120 | 0.946 | 0.075 | 0.871 | 1.915 | 2.051 | 1.04 / 1.89 | 36.9 / 17.0 |
| 832 | 896 | 0.362 | 0.066 | 0.296 | 2.631 | 2.695 | 0.955 | 0.089 | 0.866 | 2.586 | 2.921 | 0.93 / 1.95 | 51.7 / 23.7 |
| 1440 | 1536 | 0.343 | 0.068 | 0.275 | 4.674 | 4.610 | 0.969 | 0.088 | 0.881 | 4.309 | 4.649 | 0.89 / 2.15 | 88.6 / 40.7 |

- CPU: replay plus its eager glue costs 0.05-0.09 ms against 0.35 (forward core) and 0.95 ms (backward
  core) whatever N: 1.16 ms per layer and pass pair, 32.5 ms per step over 28 layers. The full autograd
  node (`h = h + mlp(h)` under the offload context, int8 backward) costs 0.87 / 1.35 ms per layer at 416
  tokens and 0.80 / 1.30 at 832, so the rest (autograd, the offload's pack and early copy, plan, ring
  calls, the residual) stays.
- GPU: the forward replay at the bucket is as fast as eager at N (at 296 tokens faster: M = 296 tiles
  badly); the backward replay is 0.1-0.35 ms slower (the int8 (gate|up) and gradient copied into the
  static buffers, and the bucket's extra tokens). On long steps (GPU-bound) a template only costs.
- Memory: bucket 512, forward and backward in one pool: +51.8 MiB allocated, +82.0 MiB reserved, as on
  the laptop; grows linearly with the bucket (640: 53.9 MiB static + pool before the workspace).
- Bitwise: replay = eager whenever B is the eager padded N (512, 640); otherwise the forward output
  differs in low bits (other cuBLAS kernels for another M), the int8 form of (gate|up) still matches.

## Traps found (measured)

1. **The harness cannot time a step in which we capture.** `harness/_train.py` TritonWatch records two
   timing events around every Triton launch of `solution/` during steps 3+ on the current stream. Inside
   a capture they become graph nodes, and `own_ms()` at the end calls `elapsed_time` on them: measured,
   it raises `CUDA error: invalid argument` (laptop and T4), which would end the run without metrics
   (and with them all 10 checks).
   So a template can only be captured in steps 1-2 (or between steps), never lazily when a new bucket
   first appears; the int8 backward and int8 offload switch on at step 8 (START_CALLS = 196), so their
   templates could only be captured early after compiling their kernels synchronously in step 1-2, or
   by launching the compiled kernels around the harness's wrapper, which a reviewer would read as
   evading the measurement. Either is a cost or a risk the gain does not pay for.
2. **External events do not keep the ring's pipeline inside the graph.** An external event recorded
   between two halves of a graph released a side stream only at the end of the graph (T4: side op done
   at 324.1 ms of a 323.9 ms graph; laptop the same); an external wait inside the graph on an event
   recorded before the replay did not wait (replay done at 0.0 ms, T4 and laptop). So the type ring's acquire (3 waits) must precede
   the replay and its release follow it: the copy of the next layer's gate starts after the whole MLP
   instead of after gate's dequantization (~3 ms later at 830 tokens), which stalls GPU-bound steps.
3. **Padding to a bucket changes bits and PCIe bytes.** On the T4 the forward GEMMs at 512 are not slower
   than at 296 (on the laptop +0.38 ms), but the int8 (gate|up) and its D2H/H2D grow with the bucket
   (wire bytes x B / N), and the output is not bitwise the eager one (fine for the checks, but a
   deviation per step size).
4. Capture itself is cheap (0.8-3 ms per template, raw `capture_begin/capture_end` on a side stream after
   one eager pass there for the cuBLAS workspace and Triton loads), but the capture stream gets its own
   cuBLAS workspace (part of the +52 MiB) and `_HELD` (the shared buffer's reuse key) must be cleared
   before and after a capture or a dequantization is skipped in the graph.

## I1's objections, checked

I1 put the ceiling of graphs at ~0.4 s because a layer is torn into pieces of 1-2 launches by the ring's
events, the D2H, dropout and the varlen attention. The MLP template does sidestep that (one replay per
pass, ring and offload outside, no dropout inside), and the CPU it removes is larger than I1 assumed
(1.16 ms per layer and pass pair). The ceiling still lands where I1 put it, for another reason: only
CPU-bound steps that fit the bucket pay, and the bucket's resident memory grows with its size. Own
kernels: in m11-full the MLP's Triton kernels are ~3.5 s of 5.2 s of own time; graphed on short steps
only the share would drop from ~20 % to ~17-18 %, on all steps to ~6.5 %.

## Why the memory cannot be cut enough

The forward template needs (gate|up) for B tokens static (2 x 8960 x 512 x 2 B = 18.4 MiB; hidden could
live in its up half with a kernel change, -9.2 MiB), the backward its int32 product (18.4 MiB) and int8
gradient (9.2 MiB) unless the backward is rewritten allocation-free. Token-chunked templates (128 tokens,
replayed N/128 times) would need all three dequantized matrices at once (the shared buffer holds one) or
a dequantization per chunk. Freeing the templates between short steps means capturing again in timed
steps (trap 1). Sharing the memory with the head's buffers at the peak would alias across modules with
the last layer's (gate|up) still on the GPU. None fits the +20 MiB budget at a gain under 0.5 s.

## Cheaper ways to the same CPU (not done here; follow-ups for the orchestrator)

- `int8_bwd.mlp_backward` recomputes per-layer constants on every call: `scale_gu = maximum(scale_g,
  scale_u)` and `_inverse(scale_gu * 127)` (7 torch launches per layer backward, 196 per step); caching
  them with `_column_scales` is bitwise identical.
- `_quant_rows` is 15 torch launches per layer backward and the dL/dx epilogue 4: one Triton kernel each
  (bitwise with IEEE division) would cut ~17 launches per layer backward (~480 per step), memory-neutral,
  and add own-kernel time.
