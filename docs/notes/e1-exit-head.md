# E1: the process exit and the head's bytes (branch claude/e1-exit-head-8fb01b)

Draft, filled in as the T4 measurements come in.

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
GPU blocks on the thread (148 MiB locally, aimed at the 0.25 s C-level exit); not measured on the T4.

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
