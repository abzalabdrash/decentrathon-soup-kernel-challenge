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

On the public rows the (gate|up) traffic is hidden (nogu is no faster). On 2,048-token steps it is not:
without it the run is 2.8 s shorter (steps 3+ -2.5 s), and starting the previous layer's copy one
tensor earlier alone gives -1.8 s. The bound is the backward's H2D: the prefetch of the next layer's
(gate|up) (70 MiB at 2,048 tokens, ~5.9 ms at the T4's ~12 GB/s) does not fit under one attention
backward together with that layer's NF4 weights.

## 2. What was built

`solution/offload_int8.py` (switch `SOUP_SOLUTION_OFFLOAD_INT8`, on by default only in the `soup train`
process, like int8_bwd.py; =1 also under pytest, =0 off):

- `gu_quant_kernel`: (gate|up) (2I, M) half -> int8 with one fp32 scale per token and group of 128
  features (absmax / 127, round to nearest; a group never straddles gate and up). The int8 values plus
  scales are 51.6 % of the half tensor's bytes (14.7 MiB instead of 28.4 at 832 tokens).
- `gu_dequant_kernel`: the inverse (the rebuild pass).
- offload.py: `offload_early(gate_up)` at the gate|up GEMM only notes the tensor when int8 applies;
  the second call, `offload_early(gate_up, spare=hidden)` right after the down GEMM (one line in
  fused_proj.py's MLP forward), quantizes into the dead `hidden` activation's storage (exactly
  2 I M bytes, the int8 size; only the scales are allocated) and starts the D2H of the two parts into
  one pinned slot. The half (gate|up) is no longer held while the copy runs (only the int8 form is),
  so it is freed when the MLP's forward returns. The prefetch and the on-demand fetch bring back one
  uint8 buffer (half the bytes, half the GPU slot).
- Unpack: the half tensor is rebuilt on the compute stream (the MLP backward's contract does not
  change), or, with `SOUP_SOLUTION_OFFLOAD_INT8_FUSED=1` (default) and the int8 MLP backward, fused_proj's
  backward unpacks inside `offload_int8.raw(True)` and `int8_swiglu_bwd_kernel` (`INT8_IN`) reads
  q * scale itself, rounded to half exactly as the rebuild writes it (bit-identical, tested); no
  rebuild pass. A layer whose int8 pair is still on the GPU at the backward (the last one) is rebuilt.
- The two kernels compile in a thread started at the first MLP backward; the first 112 offloads (four
  steps) stay half, the next one waits for the thread: a fixed switch point.

## 3. Numerics (solution/bench_offload_int8.py on the T4, soup-p1-bench, 047d9a4)

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

## 4. Cost and gain per layer (T4 bench)

TODO: tile sweep with the launch latency excluded.

## 5. ABBA

TODO

## 6. Full + jury

TODO
