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

Results: TODO (soup-e1-exit2: sol / rel / relgc).

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

Results: TODO (soup-e1-pack1 microbench, soup-e1-packv1 variants).

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
