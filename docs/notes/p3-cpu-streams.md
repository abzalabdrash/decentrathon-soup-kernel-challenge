# P3: the CPU cost of the copy and streaming machinery (branch `claude/p3-cpu-streams`, from main da895ed)

Task: fewer events, stream switches and Python per layer in weight_stream.py, offload.py, offload_int8.py and
head_stream.py, same ordering guarantees; GO at -0.5 s of wall on the public rows. After early release
(P2) the public step is CPU-bound in both passes (P2 note: 49 ms forward + 26 ms backward of GPU idle per
step waiting for the CPU to submit), so CPU time on the main thread is wall there.

## Attribution (main da895ed, public rows, `SOUP_SOLUTION_CPU_PROFILE=1`, soup-p3-cpuprof)

cProfile on steps 5-8 (inflated per call), sampler on steps 9-12. Counters per step:

| counter | per step | whose |
|---|---|---|
| Triton launches | 1,195.5 | launch paths (I1) |
| torch `current_stream` calls | 1,195 | Triton's launcher, once per launch (I1) |
| CUDA events made / `Event.record` | 2,501 / 1,922 | ~1,200 of each: the harness's own-kernel timing wraps every Triton launch with two events (harness/_train.py:60, 9.7 ms/step self in cProfile; not ours to change); 111 `record_event()` in offload.py; the rest the fixed events of weight_stream / head_stream / offload |
| `Stream.wait_event` | 736 | weight ring, offload windows and reads, head |
| `torch.cuda.stream()` contexts | 3.2 | (weight_stream's switches already go through C1's fast streams) |
| `item()` | 226.5 | outside these files |

Time inside our streaming functions per step (sampler, includes the CUDA calls they make):

| function | ms per step |
|---|---|
| weight_stream `_load` (214 copies issued per step) | 14.9 |
| offload `_offload` | 6.2 |
| offload `_wait` (wait_event wrapper) | 6.2 |
| offload `_prefetch_before` | 3.7 |
| offload_int8 `views` (56 calls: six tensor views each) | 3.5 |
| offload `drain_inflight` | 1.4 |
| head_stream `head` (includes the head's compute launches) | 11.4-14.0 |

Locally (RTX 5060 host) the pieces of one `_load` cost ~1 us each (record, stream switch, wait) and the H2D
`copy_` 7-10 us; on Kaggle's CPU the whole call is ~50-70 us, without one dominant part.

## Built: `SOUP_SOLUTION_STREAM_DIET` (08d4a29; off by default since the ABBAs below)

The orchestrator approved the small cuts at a lowered bar (-0.2 s on the public rows, Modal ABBA):

- offload.py: the side stream's events come from a pool and are recorded again instead of made per copy
  (`_side_event`; a copy's event goes back when its handle dies, a prefetch's when its unpack has waited
  for it, a read's once the side stream has waited for it). Re-recording an event on the same side stream
  can only make a stale wait wait longer, never shorter. The forward's drain and window wait once, for the
  newest copy in flight (the side stream completes its copies in order). Event record / wait call torch's C
  class directly (no Python wrapper). The int8 form's (q, s) views of a pinned slot are cached
  (`offload_int8.host_views`, keyed by slot address and shape, cleared by `release_host_memory`).
- weight_stream.py: an untimed copy or wait calls the events directly and switches to the copy stream only
  around the `copy_` itself.
- Tests (test_offload.py): the drain's single wait with the side stream held back by a GPU sleep before
  every copy and the freed blocks overwritten at once on the compute stream (a mutant that waits for the
  oldest copy instead fails it); event reuse: a second pass through the same Offloader makes at most one
  new event and every value comes back exactly. test_weight_stream's race-detection test runs with the
  diet off (it injects the bug through the Python override of `Event.record`, which the diet bypasses).
  Locally: test_offload, test_offload_int8, test_xoffload, test_jury_offload_marks 44 passed;
  test_weight_ring, test_weight_stream, test_int8_bwd 74 + 21 passed (`test_ring_switch_and_fallbacks`
  fails on this laptop on main too); Soup's 79 with SOUP_APPLY_SOLUTION=1 passed.

## ABBA on Modal T4 (tripleacodex; 08d4a29, jury rows via edddae6's `--variant-data`): NO-GO

Both variants in one session each, alternating, after a warm run; same host within a session.

| rows | host | diet on | diet off | steps 3+ on / off | peak nvidia-smi |
|---|---|---|---|---|---|
| public (soup-p3-diet-pub, 3 rounds) | PCIe x8, 3.1 GHz CPU | 45.78 / 45.54 / 45.38 (**45.57**) | 45.60 / 45.57 / 45.43 (**45.53**) | 27.95 / 27.93 | 396 / 396 |
| jury-like (soup-p3-diet-jury, seed 157, train 42000:42256, eval 51800:52056, 2 rounds) | PCIe x16, 2.0 GHz CPU | 53.36 / 52.35 (52.86) | 52.22 / 52.42 (52.32) | - | 454 / 454 |

The public rows show nothing (+0.04 s, inside the round-to-round spread of 0.2-0.4 s), against the -0.2 s
bar; the jury-like +0.54 s rests on one 53.36 s outlier, the other pair is even. The cut CPU (estimated
2-4 ms per step from the attribution, ~0.15-0.25 s per run) does not reach the wall: either the pieces
cost less than the sampler's in-function totals suggested (those include the CUDA driver calls, which
the diet keeps), or the freed main-thread time only moves the wait to the next Triton launch. The switch
stays in the code, off by default; its tests run with it forced on.

## What can be cut, and why it was not done before the deadline

- Safe, small: the forward drain could wait once on the newest copy instead of once per tensor in flight
  (copies on one side stream complete in order; ~28 waits per step), and an event pool could replace the
  111 `record_event()` constructions per step (events can be re-recorded once no future wait refers to
  their last record: after the handle's unpack, after the side stream waited for a read). Estimate 2-4 ms
  per step of CPU, ~0.15-0.25 s on the public rows: below the GO. (Built above as the diet, with the cached
  int8 views and a lighter `_load`: no measurable gain.)
- Larger: one copy per layer for the MLP matrices instead of three (contiguous gate|up|down slots and host
  ranges in the type ring) would cut ~2/3 of `_load`'s calls (~8-10 ms per step), but it changes the type
  ring's slot layout and early release's per-matrix timing; not something to land in the last hours.
- Outside these files: Triton's per-launch overhead (launch paths: fewer launches, the launcher's
  current_stream) and the harness's per-launch timing are the larger CPU items (I1).
