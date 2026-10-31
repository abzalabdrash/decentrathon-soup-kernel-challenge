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

## What can be cut, and why it was not done before the deadline

- Safe, small: the forward drain could wait once on the newest copy instead of once per tensor in flight
  (copies on one side stream complete in order; ~28 waits per step), and an event pool could replace the
  111 `record_event()` constructions per step (events can be re-recorded once no future wait refers to
  their last record: after the handle's unpack, after the side stream waited for a read). Estimate 2-4 ms
  per step of CPU, ~0.15-0.25 s on the public rows: below the GO.
- Larger: one copy per layer for the MLP matrices instead of three (contiguous gate|up|down slots and host
  ranges in the type ring) would cut ~2/3 of `_load`'s calls (~8-10 ms per step), but it changes the type
  ring's slot layout and early release's per-matrix timing; not something to land in the last hours.
- Outside these files: Triton's per-launch overhead (launch paths: fewer launches, the launcher's
  current_stream) and the harness's per-launch timing are the larger CPU items (I1).
