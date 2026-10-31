# P2: which copies sit on the step's critical path, and cutting their waits (branch `claude/p2-pcie-critical`, from main 23f92af)

Brief 5b (soup-m10-prof, main d341d0c before P1): per step H2D 222 ms, D2H 77, GEMM 172, all compute ~286 ms
against a 406 ms wall under the profiler: "~120 ms/step of waits". Target: wall -0.7 s in ABBA, peak
<= 420 MiB on the public rows.

## 1. Measuring the waits

- The diag chrome trace (solution/diag.py) runs the traced steps CPU-bound: with CPU activity 5.1-5.6 s of
  CPU-side gaps over 2-4 steps, and still 4.9-5.6 s with `SOUP_SOLUTION_DIAG_TRACE_CPU=0` (new: CUDA
  activity only; CUPTI's runtime callbacks alone slow the CPU that much). Its waits are lower bounds;
  the brief's 406 ms "wall" was measured the same way. `tools/trace_waits.py` (new) still attributes every
  compute-stream gap whose next kernel was launched before the gap began (a dependency wait) to the copy
  that ended last inside it.
- The cheap, unbiased numbers come from CUDA events around each wait: weight_stream's
  `SOUP_SOLUTION_WEIGHT_RING_TIMING=1` / `SOUP_SOLUTION_STREAM_TIMING=1` (exist), and new
  `SOUP_SOLUTION_OFFLOAD_TIMING=1` for the offload's waits (backward: each (gate|up) and layer input,
  prefetched / fetched / still on the GPU; forward: the D2H window and drain), in the summary line.

**Main 218c174 (= main 23f92af + diag knobs), one run each (soup-p2-trace-pub / -jury, timing variant):**

| rows | weight ring, forward waits | weight ring, backward waits | of them MLP matrices | head waits |
|---|---|---|---|---|
| public | 30.8 ms in 7,168 | **308.0 ms** in 7,168 | **300.0 ms** | 5.2 ms per step (~0.33 s per run) |
| jury-like (seed 157, train 42000:42256, eval 51800:52056) | 22.2 ms | 119.6 ms | 113.3 ms | 1.5 ms per step |

(per run of 64 steps; 0 late loads; copies at 11.6 GB/s, head chunks at 11.5-11.8 GB/s). The traces
(CPU-bound, lower bounds) put the remaining dependency waits on the head's 12 MiB chunks, the type ring's
6.77 MiB matrices and the offload's int8 (gate|up) forms (11-20 MiB) and layer inputs (7.6 MiB at 1,330
tokens). So the copy waits on the public rows are of the order of 10-20 ms per step, not 120: most of the
gap between the profiler's compute sum and its wall is the CPU (the profiled steps are CPU-bound, and the
public rows' short steps are CPU-bound unprofiled too, C1 section 1).

## 2. Levers built (all behind switches)

- `SOUP_SOLUTION_RING_EARLY_RELEASE` (default on): the int8 MLP backward releases each type-ring matrix
  slot right after its decode (`int8_bwd.mlp_backward(..., done=)`), not after the whole MLP backward, so
  the next layer's down / gate / up copies start under this backward's GEMMs instead of under the
  attention backward only. Test: type ring + int8 backward with GPU sleeps before every decode, early and
  late, bit for bit against the resident model.
- `SOUP_SOLUTION_RING_SLOTS_GATE/_UP/_DOWN=2` (default 1): two slots for an MLP matrix type (+6.8 MiB each):
  layer j in slot j % 2, the release of layer i loads i + 2*step into the slot it frees: copies start a
  layer earlier, and the last two layers' matrices stay resident across the forward/backward turn. Tests:
  fp16 and int8 backward, sleeps before every dequantization and decode, bit for bit, no late load,
  layers 0 and 1 resident after the backward.
- Existing knobs in the comparison: `SOUP_SOLUTION_OFFLOAD_PREFETCH=2` (the offload's prefetch one tensor
  deeper), `SOUP_SOLUTION_HEAD_SLOTS=3` (a third head chunk slot, +12 MiB around the head; head_stream.py is
  H1's file, only its switch is used).

## 3. ABBA

**Early release, jury-like rows (soup-p2-abba-er-jury, 431b500):** late 47.89 / 48.23 s, early 48.14 /
48.14 s (steps 3+ 27.98 against 27.99): no effect, as the 113 ms of backward MLP waits per run there
predicted.

**Every wait timed, public rows (soup-p2-timing, b8d5902, one run each with all three timings on;
per run of 64 steps):**

| | late (main) | early release | early + two slots per MLP type |
|---|---|---|---|
| offload: backward (gate|up) prefetched | **1,248.3 ms** in 1,728 | 24.0 ms | 27.7 ms |
| offload: (gate|up) still on GPU / layer inputs prefetched | 0.2 / 5.2 ms | 6.3 / 7.7 ms | 1.2 / 7.0 ms |
| offload: forward drain (D2H window) | 119.8 ms | 81.2 ms | 103.3 ms |
| weight ring: forward (MLP matrices) | 35.2 (25.4) ms | 25.3 (16.8) ms | 29.6 (20.5) ms |
| weight ring: backward (MLP matrices) | 147.2 (138.1) ms | 34.8 (29.3) ms | 34.5 (28.9) ms |
| head | 9.5 ms per step | 6.5 ms per step | 7.9 ms per step |
| peak nvidia-smi | 395 | 395 | 415 |

On main the backward's (gate|up) prefetch of layer L-1 is issued at layer L's post-attention norm,
behind the 21 MB of layer L-1's MLP matrices that the late release only issues after layer L's whole
MLP backward: on the one H2D engine it arrives late, and its wait (taken first, at the unpack) also
absorbs the matrices' own lateness. Early release puts the matrices on the engine under layer L's
backward GEMMs: 1.39 s of compute-stream waits per run become 0.05 s.

**Public ABBA (soup-p2-abba-pub, b8d5902, 4 variants x 2 rounds + warm run, same session):**

| variant | wall s | steps 3+ s | median step | nvidia-smi | alloc |
|---|---|---|---|---|---|
| late (main) | 43.62 / 43.82 (43.72) | 23.74 | 0.369 | 395 | 230.1 |
| early release | 43.42 / 43.17 (**43.30**) | **23.20** | 0.366 | 395 | 230.1 |
| early + two slots per MLP type | 43.26 / 43.43 (43.35) | 23.16 | 0.363 | 415 | 250.4 |
| early + offload prefetch 2 | 43.53 / 43.99 (43.76) | 23.26 | 0.363 | 455 | 255.3 |

Early release: **-0.43 s wall, -0.55 s steps 3+** on the public rows, both rounds, same peak, bit for bit
the same results. Two slots per type add nothing on top (+20 MiB); a deeper offload prefetch nothing
(+60 MiB). On the jury-like rows early release is neutral (above): their longer steps already hid most
of these copies (113 ms of backward matrix waits per run there against 1.4 s here).

What is left after early release (public): the head, 6.5 ms per step (~0.4 s per run): its 445 MB at
11.5 GB/s take ~39 ms against ~32 ms of head compute, so it is bandwidth-bound from the first chunk on;
only fewer bytes (H1's int8 head) or an earlier start of its copies help. The forward drain (81 ms per
run) and the weight ring (60 ms per run) are small.
