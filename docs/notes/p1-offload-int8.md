# P1: the MLP's (gate|up) crosses PCIe as int8 (branch `claude/p1-offload-int8-8429f6`, from main 8387938)

Organizers' answer 43: a tensor saved only for the backward may be stored in int8 with a scale (lossy)
when the forward, the loss and the metrics do not change and all 10 checks pass.

## 1. Is the (gate|up) traffic on the critical path at all? (ceiling, T4, ABBA 2 rounds + warm run)

Variants of main 8387938 (`a66accd` only adds `--variant-data` to the runner), solution-only runs:
`base` (defaults), `nogu` = `SOUP_SOLUTION_OFFLOAD_MIN_NUMEL=100000000` (the (gate|up) stays on the GPU,
nothing of it crosses PCIe; the layer inputs still go), `pf2` = `SOUP_SOLUTION_OFFLOAD_PREFETCH=2` (the
backward asks for the previous layer's (gate|up) one tensor earlier, at the start of this layer's MLP
backward instead of its post-attention norm).

| rows | variant | wall s (2 runs) | steps 3+ s | nvidia-smi MiB | alloc peak MiB |
|---|---|---|---|---|---|
| public (soup-p1-ceil-pub) | base | 47.14 / 49.79 | 27.37 | 395 | 229.3 |
| | nogu | 49.70 / 48.89 | 27.73 | 1735 | 1546.0 |
| | pf2 | 50.36 / 50.16 | 28.17 | 495 | 278.4 |
| `--select longest` (2,048 tokens every step, seed 157; soup-p1-ceil-long) | base | 75.79 / 77.55 | 52.60 | 453 | 284.5 |
| | nogu | 73.39 / 74.40 | 50.06 | 2373 | 2156.9 |
| | pf2 | 74.59 / 75.15 | 51.17 | 593 | 354.3 |

On the public rows the (gate|up) traffic is hidden (nogu is no faster). On 2,048-token steps it is not
in that session: without it the run is 2.8 s shorter (steps 3+ -2.5 s), and starting the previous
layer's copy one tensor earlier alone gives -1.8 s (at +140 MiB). Session matters: in soup-p1-abba-long2
the card throttled (its first run 75.5 s, the later ones of the same variant 82 s) and the steps were
compute-bound; there the copies were not the bound (section 5).

## 2. What is in the branch (cebfd3e)

`solution/offload_int8.py` (switch `SOUP_SOLUTION_OFFLOAD_INT8`, on by default only in the `soup train`
process, like int8_bwd.py; =1 also under pytest, =0 off):

- The int8 form: per element (i, t) of the gate half one int16, the gate's int8 in the low byte and the
  up's in the high byte, plus one fp32 scale per token and group of 128 features of each half (absmax /
  127, round to nearest). 2 I M bytes + scales = 51.6 % of the half (gate|up) (14.7 MiB instead of 28.4
  at 832 tokens, 36.1 instead of 70.0 at 2,048).
- `gu_quant_kernel` (SWIGLU): computes the MLP's SwiGLU forward exactly as the example kernel does
  (`gate * sigmoid(gate) * up` in fp32, one rounding: hidden bit for bit, test) and in the same pass
  writes the int8 form in place over the gate half of (gate|up) (each element's two bytes, loaded before
  by the same program; no other program touches them). No extra read of (gate|up), no new buffer of its
  size; only the scales are allocated.
- fused_proj.py, MLP forward (a few lines): `offload.offload_early(gate_up)` at the gate|up GEMM returns
  "int8" after reserving the host slot; then `swiglu_quant` replaces the example SwiGLU launch and
  `offload_early(gate_up, wire=...)` starts the D2H of the int8 form (half the bytes) at once. In flight
  the offload holds (gate|up)'s storage, as the half offload does, for half as long.
- The backward brings back one uint8 buffer (half the bytes, half the GPU slot). With the int8 MLP
  backward and `SOUP_SOLUTION_OFFLOAD_INT8_FUSED=1` (default) fused_proj's backward unpacks inside
  `offload_int8.raw(True)` and gets that buffer: `int8_swiglu_bwd_kernel` (INT8_IN) reads the int16
  pairs and scales (rounded to half as the rebuild pass would, so bit for bit the same result, test) and
  writes d_gate / d_up interleaved into the two halves of its own int32 product (no new buffer);
  `quant_t_int8_kernel` (IL) reads that layout. Otherwise (fp16 backward: the first four steps, or the
  last layer, still on the GPU) `gu_dequant_kernel` rebuilds the half tensor.
- The kernels compile in a thread started at the first MLP backward; the first 112 offloads (four steps)
  stay half, the next one waits for the thread: a fixed switch point. `SOUP_SOLUTION_OFFLOAD_INT8_GROUP`,
  `_TILE_M`, `_WARPS` are experiment knobs (defaults 128, 128, 8).

Tests: `solution/test_offload_int8.py` (kernels against torch, SwiGLU bit for bit, all three ways back
under GPU delays, the version guard, the switch point, no record_stream / CPU wait, the fused MLP forward
bit for bit and dL/dx within 2 %, the raw path bit for bit the rebuild path, the int8 backward's memory).
Locally (RTX 5060, WSL): test_offload_int8 + test_int8_bwd + test_offload + test_xoffload +
test_jury_offload_marks 46 passed, test_fused_proj 40 passed.

## 3. Numerics (solution/bench_offload_int8.py on the T4, soup-p1-bench, 047d9a4: the same arithmetic)

Relative error against the fp16 MLP backward with the half offload, one Qwen2.5-1.5B-shaped block
(fused attention with LoRA r=16 on q and v, then the fused MLP, both residual; NF4 weights):

| tokens | path | dL/dx | LoRA A_q, B_q, A_v, B_v |
|---|---|---|---|
| 826 | int8 MLP backward (main today), half offload | 0.0224 | 0.0226 0.0223 0.0239 0.0236 |
| 826 | fp16 backward, int8 offload, group 128 | 0.0049 | 0.0049 0.0049 0.0053 0.0052 |
| 826 | int8 backward + int8 offload, group 128 | 0.0230 | 0.0232 0.0229 0.0244 0.0238 |
| 2048 | int8 MLP backward, half offload | 0.0224 | 0.0226 0.0226 0.0235 0.0231 |
| 2048 | int8 backward + int8 offload, group 128 | 0.0229 | 0.0234 0.0232 0.0247 0.0235 |

Groups of 64 / 256 features: 0.45 / 0.53 % alone, the same 2.3 % combined. The int8 offload adds
~0.5 % in quadrature to the 2.2 % that the int8 MLP backward already puts on the gradients.

## 4. Cost and gain per layer (T4, soup-p1-bench2, launch latency excluded)

Copies (pinned host, one layer): D2H 2.35 -> 1.19 ms at 832 tokens, 5.61 -> 2.91 ms at 2,048; H2D
2.54 -> 1.28 and 5.96 -> 3.10. A separate quantization pass (the first design) cost 0.28 ms at 832 and
0.55 ms at 2,048 per layer at its best tile (128 tokens x 4-8 warps; 64 x 4: 0.32 / 0.75), the rebuild
pass about the same: 28 layers x 64 steps of that is 0.5 s (public) to 1-2 s (2,048) of GPU per run,
which is what the long-row ABBA below paid. Folded into the SwiGLU (cebfd3e), the forward adds only the
int8 write: locally 0.10 ms at 832 and 0.17 ms at 2,048 over the example SwiGLU (T4: TODO).

## 5. ABBA (solution-only runs, same session, 2 rounds ABBA + warm run)

| commit, design | rows | variant | wall s | steps 3+ s | step 0 s | nvidia-smi | alloc |
|---|---|---|---|---|---|---|---|
| 8c9e37b: separate quant into a new buffer, rebuild pass (soup-p1-abba-pub) | public | off | 46.34 / 47.24 | 26.14 | 0.500 | 395 | 229.3 |
| | | i8 | 47.09 / 47.14 | 26.32 | 0.631 | 455 | 239.5 |
| d0d48a0: quant after the down GEMM into hidden's storage (soup-p1-abba-pub2) | public | off | 45.90 / 46.39 | 25.20 | 0.512 | 395 | 229.3 |
| | | i8 fused (fp16 dgate/dup buffer) | 45.29 / 45.60 | 24.25 | 0.651 | 435 | 254.7 |
| | | i8 rebuild pass | 46.19 / 46.14 | 24.80 | 0.628 | 455 | 229.3 |
| d0d48a0 (soup-p1-abba-long2, card throttling) | longest, 2,048 | off | 82.34 / 82.04 | 59.06 | 1.093 | 453 | 284.5 |
| | | i8 fused | 83.18 / 83.36 | 59.84 | 1.115 | 513 | 320.6 |
| | | i8 rebuild pass | 84.53 / 84.49 | 61.05 | 1.077 | 553 | 284.5 |

| 1f44bbc: + d_gate/d_up interleaved as half words over the int32 product (soup-p1-abba-jury) | jury-like (seed 157, train 42000:42256, eval 51800:52056) | off | 50.35 / 52.91 | 30.16 | 0.665 | 453 | 267.6 |
| | | i8 | 55.96 / 56.12 | 34.15 | 0.740 | 453 | 268.6 |
| 1f44bbc (soup-p1-abba-pub3) | public | off | 47.60 / 47.50 | 26.80 | 0.501 | 395 | 229.3 |
| | | i8 | 49.85 / 50.36 | 29.13 | 0.622 | 395 | 230.1 |
| cebfd3e: int8 form made in place by the SwiGLU forward (soup-p1-abba-jury2) | jury-like | off | 50.98 / 51.13 | 30.90 | 0.667 | 453 | 267.6 |
| | | i8 | 53.65 / 53.90 | 33.57 | 0.717 | 453 | 268.6 |

Bytes to the host over a long-row run: 143.5 GiB (off) -> 87.9 GiB (i8). Step 0 is +0.13 s with int8:
the compile thread competes with the first step (62c2094 starts it after the first step).

1f44bbc and cebfd3e lost 2.6-4.4 s to two slow kernels, not to the offload: with d_gate / d_up written as
interleaved half words (stride-2 stores) and per-element scale gathers, int8_swiglu_bwd_kernel took 0.94
ms instead of 0.23 at 832 tokens (2.16 / 0.62 at 2,048) and the transposing quantization 0.36 / 0.09
(own-kernel share 29-30 % against 16-17 %). 1e0eac4 stores one 32-bit word per element and loads one
scale vector per 32-row tile: 0.190 ms from the int8 form against 0.225 from the half one at 832 tokens,
0.545 / 0.625 at 2,048; quantization 0.080 / 0.093 and 0.299 / 0.303 (RTX 5060). The memory of
1f44bbc / cebfd3e is that of off (453 MiB on the jury rows, 395 public): the earlier +40 / +60 MiB are
gone.

TODO: 1e0eac4 on jury-like and public rows.

## 6. Full + jury

TODO
