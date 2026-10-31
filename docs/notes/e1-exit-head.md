# E1: the process exit and the head's bytes (branch claude/e1-exit-head-8fb01b)

## Report

**A, exit (`solution/exit_release.py`).** On the T4 our process spent 0.81-0.88 s after its threads
were gone (kernel teardown) against the base's 0.32 s: ~3.6 GB of pinned host memory handed back
page by page with nothing to overlap. `SOUP_SOLUTION_EXIT_RELEASE=1` starts a non-daemon thread at
the last step's end that frees the offload slabs (cudaFreeHost) and unregisters the weights' and
head's buffers (values stay, pageable), under the checkpoint/adapter save. `SOUP_SOLUTION_EXIT_NOGC=1`
disables the collector there (Py_FinalizeEx's 44 ms full collection is skipped). Train end -> process
end: 1.84 -> 1.27 s (-0.57) -> 1.22 s (-0.61) with NOGC (soup-e1-exit2); `=all` (also the allocator's
cached GPU blocks) another -0.035 s (soup-e1-exit3, 2 runs). No os._exit, nothing skipped.
**Recommend on: EXIT_RELEASE=all, EXIT_NOGC=1.** `solution/__init__.py`: one `if` wiring it.

**B, head (`solution/head_pack.py`, `SOUP_SOLUTION_HEAD_PACK=1`, own kernel `head_unpack_kernel`).**
Lossless 12-bit form of the bf16-origin fp16 weight: exactly 75.00 % (333.8 of 445.1 MiB), 15
escapes, every chunk verified bit for bit at build. T4: copy 1.06 -> 0.82 ms per 4,096-row chunk,
decode 0.15 ms; isolated head -9.3 ms at <= 650 tokens, +2..3 ms at >= 1,100 (hence
`SOUP_SOLUTION_HEAD_PACK_MAX_ROWS=800`). In training the head's copy wait falls 9 -> 2 ms/step, but
the power-limited T4 returns only -0.07..-0.24 s per run (per-step analysis, soup-e1-packv2/v3), and
the extra CUDA module costs +2 MiB nvidia-smi (401 -> 403). **Recommend off.**

Full run with all four on (soup-e1-full1): all 10 checks, 46.4 s / 403 MiB vs base 80.5 s.
Question for the organizers: below. Branch: `claude/e1-exit-head-8fb01b`.

## Part A: the process exit

Tool: `solution/bench_exit.py` + `solution/exit_probe.py` (`--mode bench`). Runs the harness's own
command (`harness/_train.py`, run_task's config and environment) through a wrapper script that stamps
every atexit callback, the finalization's collections and the end of the module teardown on
CLOCK_MONOTONIC, while the parent stamps Popen, each output line, and polls `/proc` (threads, state)
until wait() returns.

### Where the time goes (T4, soup-e1-exit1, 2 runs each, commit aa6d57e)

| stage (s) | base | ours |
|---|---:|---:|
| Trainer.train end -> main() end (checkpoint/adapter save, Soup's summary, harness metrics) | 0.36 | 0.47 |
| atexit callbacks (ours: weakref.finalize unregistering the registered buffers 69 ms) | 0.03 | 0.09 |
| Python finalization (its full gc: base 0.36, ours 0.05) | 0.84 | 0.13 |
| C-level exit (static destructors, CUDA runtime) | 0.83 | 0.26 |
| kernel teardown of the process (threads gone -> wait() returns) | 0.32 | **0.81** |
| train end -> process end | 2.39 | 1.75 |

Our `[solution] phase exit` mark is 0.86 s before the process end because everything after the
atexit callbacks (finalization, C exit, the kernel's teardown) comes after it; the 0.81 s kernel part
is ours: ~3.6 GB of page-locked host memory (offload slabs 2,560 MiB from cudaHostAlloc, the layers'
NF4 weights 645 MiB and the fp16 head 445 MiB registered with cudaHostRegister) handed back page by page
after the process is gone, with nothing to overlap.

### Change (SOUP_SOLUTION_EXIT_RELEASE=1, SOUP_SOLUTION_EXIT_NOGC=1; off by default)

`solution/exit_release.py`: at the last step's on_step_end a non-daemon thread synchronizes, frees
the offload slabs (`offload.release_host_memory`: cudaFreeHost through torch's host allocator) and
unregisters every registered buffer (`weight_stream.unpin_all`: the values stay as pageable memory,
so a later eval/generation is still correct, only slower), under the checkpoint save, the adapter
save and Soup's summary. EXIT_NOGC disables the collector at the same point (Py_FinalizeEx's own full
collection is then skipped; the module teardown runs as always). `solution/__init__.py`: one `if`
wiring the switch (orchestrator: please note).

Results (T4, soup-e1-exit2, commit 10e9308, ABBA 2 runs each + a warm run; seconds):

| stage | defaults | EXIT_RELEASE=1 | + EXIT_NOGC=1 |
|---|---:|---:|---:|
| Trainer.train end -> main() end | 0.48 | 0.47 | 0.46 |
| atexit callbacks | 0.09 | 0.07 | 0.07 |
| Python finalization (of it gc) | 0.13 (0.044) | 0.12 (0.044) | 0.10 (0) |
| C-level exit | 0.26 | 0.25 | 0.26 |
| kernel teardown | 0.88 | 0.36 | 0.34 |
| **train end -> process end** | **1.84** | **1.27 (-0.57)** | **1.22 (-0.61)** |

The release thread takes 0.50-0.53 s (2,560 MiB of slabs freed, 1,090 MiB unregistered) and is done
before main() returns (the interpreter's join of it waits 0.0 ms); the main thread's own tail is not
slower (0.48 -> 0.46 s). Our kernel teardown is now at the base's (0.32 s in soup-e1-exit1). Wall in
that session: 47.7 / 48.6 / 48.5 s, i.e. inside the session's loop noise (loop 28.5-30.5 s); the
exit stages are what this switch changes and they are measured directly. What is left above the base:
~30 ms in torch.library's own finalizers (the base has the same 234-246 `_del_library` callbacks, 12 ms
there; left alone), and the post-train work (+0.1 s: the harness reads back ~41k CUDA event pairs of
our kernels' launches). `SOUP_SOLUTION_EXIT_RELEASE=all` also returns the caching allocator's free
GPU blocks on the thread (136 MiB on the T4): soup-e1-exit3 (db6a66f, 2 runs each, both with NOGC):
train end -> process end 1.116 s (=1) against 1.081 s (=all), C-level exit 0.242 -> 0.225, kernel
0.301 -> 0.278; the release thread 0.45-0.47 s either way.

## Part B: the head's bytes (SOUP_SOLUTION_HEAD_PACK=1, off by default)

Offline on the real checkpoint (233,373,696 values of the tied embedding, bf16 -> fp16):
- exponent fields 0..13; entropy 2.58 bits; ideal sign + 7 mantissa bits + H(exp) = 10.58 bits (66.1 %);
- every normal value has its 3 low fp16 mantissa bits zero (bf16 origin); all 482,053 subnormals and
  zeros fall into 4 exact shifted classes (m < 128; 128..255; 256..511 even; 512..1023 multiple of 4);
- so 17 classes in all, each an affine map `bits = sign << 15 | ADD + (M << SH)`; a 4-bit code into a
  per-chunk table of the 16 most frequent classes covers all but 15 values (exponent field 13),
  which are escapes (max 4 per 4,096-row chunk, 5 per 3,072-row chunk).
- Ratio: exactly 75.00 % (333.8 MiB instead of 445.1 MiB), chunk sizes uniform (1.5 bytes per value);
  DFloat11-style Huffman would reach ~66 % but needs a bit-serial decoder.

Format and kernel: `solution/head_pack.py`. Per chunk an A plane (sign | 7 bits, 1 byte per value) and a
B plane (two 4-bit codes per byte); `head_unpack_kernel` (own Triton kernel) decodes pairs of values:
one 16-bit load of two A bytes, one B byte, two table loads, one 32-bit store of two fp16; escapes are
patched in registers. Built once per run in a thread at peft's kbit preparation: the fp16 host copy
goes through the GPU in chunks (64K-entry bit tables for the A byte and the class), each chunk is
decoded with the training kernel and compared bit for bit with the fp16 chunk before its bytes go to
the pinned host buffer; any mismatch or overflow leaves the pack unbuilt. The head's ring then holds
packed chunks (copy stream) and each chunk is decoded into one fp16 buffer right before its GEMMs;
the packed slot is released as soon as its decode is queued. SOUP_SOLUTION_HEAD_PACK_MAX_ROWS limits
it to heads of at most that many rows (long steps are compute-bound: the decode would only add).

T4 microbench (`solution/bench_head_pack.py`, soup-e1-pack1, the real checkpoint; the head called
right after its prefetch, so nothing hides the first copies; medians of 6):

| chunk rows | copy fp16 / packed per chunk | decode per chunk | head N=300 | N=650 | N=830 | N=1100 | N=1400 | N=1970 |
|---|---|---|---|---|---|---|---|---|
| 4096 | 1.06 / 0.82 ms (11.9 / 11.6 GB/s) | 0.14-0.16 ms | 40.2 -> 30.9 | 40.3 -> 31.1 | 40.8 -> 41.9 | 48.3 -> 50.4 | 65.2 -> 67.4 | 85.2 -> 88.1 |
| 3072 | 0.82 / 0.61 ms | 0.12-0.14 ms | 41.4 -> 29.9 | 41.8 -> 30.4 | 42.0 -> 38.6 | 50.8 -> 53.8 | 68.1 -> 71.7 | 82.1 -> 87.5 |
| 2048 | 0.55 / 0.42 ms | 0.09-0.11 ms | 40.2 -> 30.4 | 40.6 -> 34.9 | 40.8 -> 37.0 | 53.1 -> 55.8 | 65.5 -> 68.9 | 84.9 -> 87.7 |

(head ms, fp16 ring -> packed ring; every packed head bit-identical to the fp16 head.) Copy-bound
heads (up to ~800 rows) are 9-11.5 ms shorter; compute-bound heads pay the decode (38 x 0.15 ms ~
6 ms of GPU, +2..5 ms net), hence SOUP_SOLUTION_HEAD_PACK_MAX_ROWS. The decode kernel's block/warps
sweep moves it by +-10 % (1,024 pairs x 8 warps kept); it runs at ~150 GB/s read+write on the T4
(a laptop GPU with a 32 MB L2 measures it above its own D2D copy rate; a 256-entry pair table and
4-values-per-lane variants were not faster). Head memory above its start: 4,096 rows +6..7 MiB
packed (two 9 MiB packed slots + one 12 MiB decode buffer against two 12 MiB fp16 slots); 3,072 rows
packed is below the 4,096-row fp16 ring at every N (26.1 vs 28.2 MiB at N=300, 40.1 vs 44.3 at 1,400).
With MAX_ROWS the run's largest steps (which set the peak) keep the fp16 ring. Build in the thread:
0.7 s on the T4 (334 MiB pinned, 38 chunks encoded and verified), 75 MiB transient GPU memory.

In training, soup-e1-packv2 (namedesign, 59402d0, variants ABBA 2 runs each + warm, a hot session:
median step 0.43 s; pack started at peft's kbit preparation):

| variant | wall | steps 3+ | step 0 | first step mark | head copy wait/step | nvidia-smi |
|---|---:|---:|---:|---:|---:|---:|
| defaults | 48.31 | 28.47 | 0.565 | +16.75 s | 6.4-7.9 ms | 401 |
| HEAD_PACK=1 (all heads) | 48.32 | 28.12 (-0.35) | 0.483 | +17.11 s | 1.7-1.9 ms | 403 |
| HEAD_PACK=1, ROWS=3072 | 48.40 | 28.21 (-0.26) | 0.507 | +17.07 s | 0.7-1.0 ms | 403 |
| HEAD_PACK=1, MAX_ROWS=800 | 48.29 | 28.02 (-0.45) | 0.480 | +17.21 s | 1.8-1.9 ms | 403 |

The steps gain what the microbench says, but the build delayed the first step by 0.3-0.45 s
(setup_transformers +0.25..0.4 s: started at prepare, its cudaHostRegister of 334 MiB holds up the
setup's own CUDA calls), so the wall did not move. torch's reserved/allocated are identical (272 /
234.6 MiB); nvidia-smi +2 MiB outside the allocator (one more CUDA module). The build now starts at
the end of Soup's setup (SOUP_SOLUTION_HEAD_PACK_AT=setup, default; under the trainer build, CPU
work, where the offload prepins for the same reason): soup-e1-packv3 measures it.

Full run with the candidate switches (soup-e1-full1, peagverr, 59402d0, `--env EXIT_RELEASE=1
EXIT_NOGC=1 HEAD_PACK=1 HEAD_PACK_MAX_ROWS=800`): **all 10 checks pass**; base 80.5 s / 14,345 MiB, ours
46.4 s / 403 MiB (s 1.735 against a fast base; m 35.6), eval -0.015 %, loss curve 0.01 % / 0.05 %,
own kernels 14.0 % (head_unpack_kernel listed under check 3); 33 of 64 heads packed, the pack ready
0.2 s before train begin; the release: 2,560 MiB freed and 1,424 MiB unregistered in 0.45 s.

soup-e1-packv3 (namedesign, c0c7be2, every variant with EXIT_RELEASE=1 + EXIT_NOGC=1, ABBA 2 runs,
a cooler session): steps 3+ / wall / first step minus setup begin: no pack 25.46 / 43.62 / 5.87 s;
pack at setup end, MAX_ROWS=800: 25.94 / 44.34 / 5.87 s; pack at prepare: 26.39 / 45.60 / 6.30 s;
pack for every head: 26.20 / 44.70 / 5.88 s, and one of its two runs crossed a 20 MiB page (423 MiB,
reserved 292: the packed ring is +6 MiB at the head of the largest steps). Starting the build at the
end of the setup removes the startup cost; the step totals swing with the session's drift (the first
positions are faster) more than the pack moves them. Normalizing by the unpacked large steps of the
same runs (packed only for steps of <= 800 tokens: 31 of 64), the packed steps are 0.6 % (packv2) and
2.2 % (packv3) shorter: -0.07..-0.24 s per run. The head's copy wait itself falls from 6.4-9.7 to
1.7-2.5 ms per step (-0.45 s per run); the T4 at its 70 W limit spends the idle it loses on lower
clocks (RESEARCH: removed GPU idle never returns to the wall in full). Verdict: works, bit-exact,
ready behind its switch; not worth +2 MiB at the measured gain.

## Question for the organizers (exact wording)

> Замороженный общий вес `embed_tokens`/`lm_head` мы, по вашему разрешению, держим в pinned-памяти
> хоста и стримим чанками на GPU каждый шаг. Можно ли держать на хосте этот вес в сжатом без потерь
> виде (12 бит на значение вместо 16: байт «знак + 7 бит мантиссы» и 4-битный код класса порядка по
> таблице чанка, плюс список из 15 исключений) и распаковывать каждый чанк на GPU своим Triton-ядром
> обратно в fp16 прямо перед GEMM? Значения и dtype, подаваемые в операции, бит-в-бит те же, что у
> базы (fp16-вес): при загрузке каждый чанк распаковывается тем же ядром и сравнивается с fp16-чанком,
> при любом расхождении сжатие выключается. Сжатый вид строится заново в каждом запуске из чекпойнта
> (между запусками ничего не кэшируется), его построение и распаковка входят в замер.

English: The frozen tied embed_tokens/lm_head weight is, as you allowed, kept in pinned host memory
and streamed to the GPU in chunks every step. May we keep it on the host in a lossless compressed
form (12 bits per value instead of 16: a sign + 7 mantissa bits byte and a 4-bit exponent-class code
into a per-chunk table, plus a list of 15 exceptions) and decompress each chunk on the GPU with our
own Triton kernel back to fp16 right before the GEMMs? The values and dtypes fed to the ops are
bit-identical to the base's fp16 weight: at load time every chunk is decoded with the same kernel
and compared with the fp16 chunk, and any mismatch turns the compression off. The compressed form is
rebuilt from the checkpoint in every run (nothing is cached between runs); building and decoding it
are inside the measured time.

The exit release needs no question in our reading (no os._exit, no skipped finalization, no
background process; memory that training no longer uses is handed back on a thread while Soup
saves); if the orchestrator prefers to ask: «Можно ли после последнего шага обучения освобождать
pinned-буферы оффлоада и снимать регистрацию pinned-памяти весов в потоке, пока Soup сохраняет
адаптер (значения весов остаются в обычной памяти, все выходы пишутся как раньше), и выключать
сборщик мусора после последнего шага?»

## State at hand-over (31 Oct, ~11:00 Astana)

- Branch head after this commit; main 0b47c87 (B1 + M1) merged in cleanly (only a syntax check after the
  merge: re-run the targeted tests and one T4 run on the merged code).
- On by default: nothing new (all E1 switches are off in code). Recommended on: SOUP_SOLUTION_EXIT_RELEASE=all,
  SOUP_SOLUTION_EXIT_NOGC=1 (-0.57..-0.65 s, measured). Recommended off: SOUP_SOLUTION_HEAD_PACK (+2 MiB).
- In flight: no Kaggle or Modal run (all E1 sessions finished and were downloaded before the WSL restart).
- Left: find the +2 MiB outside the allocator with the pack on (host registration's GPU page tables vs the
  new kernel module): `python tools/modal_run.py --profile glofcast-prod --mode bench --host any --bench
  "solution.bench_pack_mem" --slug e1-pmem --commit <head>` (prints nvidia-smi memory.used after each step).
  If it is the registration, put the packed bytes in already-registered memory; then the pack is a small net
  win (-0.07..-0.24 s, 0 MiB) and can be enabled with SOUP_SOLUTION_HEAD_PACK_MAX_ROWS=800.

## Follow-up on main ce27a54 (31 Oct, branch claude/e1-exit-attn)

- Tests on the merged main (the orchestrator merged head_stream.py by hand with C1's slot views):
  test_exit_release + test_head_pack 13/13 (test_streamed_head_packed_equals_fp16: the packed ring gives
  the fp16 ring's bits), test_head_stream 15/15, Soup's 79 with apply() 79/79 (laptop). The pack branch
  of `_load` / `_acquire` / the head loop reads the same as on the E1 branch.
- ce27a54's commit message says the exit release is on by default, but `solution/__init__.py` still
  reads `SOUP_SOLUTION_EXIT_RELEASE` with default "0" and no setdefault: it is off unless the
  environment sets it (told the orchestrator).
- The +2 MiB with HEAD_PACK=1 (T4, peagverr soup-e1-pmem, ce27a54, `solution.bench_pack_mem`,
  nvidia-smi memory.used after each step): context 107 MiB; register 334 MiB of host memory +0;
  register 1,090 MiB more +2; 10 pinned slabs of 256 MiB (cudaHostAlloc) +6; small tensors +16 (the
  first allocations and kernels of the process); load `head_unpack_kernel`'s module +0; another
  Triton module +0; unregister the 334 MiB +0. So it is the GPU-side mapping of page-locked host
  memory (about one 2 MiB step per ~1 GB pinned or registered), not the kernel module: the pack's
  334 MiB move our 645 + 445 + 2,560 MiB over the next step. Already-pinned memory with room for it is
  only the offload's spare slabs (7 of 10 used on the public data, all 10 at 2,048-token rows, where
  two slabs would be pinned again on demand, 130 ms each, and the step comes back). Verdict unchanged:
  HEAD_PACK stays off (its gain is within the session noise and the organizers' answer on the packed
  weight is pending); by the formula +2 MiB costs nothing while we have the best memory.
