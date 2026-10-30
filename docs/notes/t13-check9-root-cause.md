# T13: what failed check 9 on d9f60bb (from the existing run files, no new Kaggle run)

Full run 20261030-185502 (d9f60bb), solution: torch_max_allocated_mib 2512.6, torch_max_reserved_mib 2592,
nvidia-smi peak 2385. The 2512.6 is exactly the setup peak printed by the diag run
("phase setup_transformers end ... peak 2513 MiB"); every training step peaks at 982-1874 MiB allocated.
So the reserved peak 2592 is a setup transient that the 100 ms nvidia-smi sampler missed, not the
training loop's allocator.

Where: solution/embed16.py, `_halve`/loop body:
    half = weight.data.to(torch.float16)                       # +445 MiB
    if not torch.equal(half.to(torch.float32), weight.data):   # +890 MiB fp32 copy, +222 MiB bool of eq
At that moment the fp32 embedding (890 MiB, cast by prepare_model_for_kbit_training) is alive: 946 + 445 +
890 + 222 = 2503 MiB, measured 2513. Then weight.data = half and empty_cache drop it to 592 reserved.

Fix (embed16.py, owned by T14): compare in row chunks, e.g.
    ok = all(torch.equal(half[i:i + 8192].float(), weight.data[i:i + 8192]) for i in range(0, rows, 8192))
(transient 50 MiB + 12 MiB instead of 1112 MiB), or avoid the fp32 cast by peft in the first place.
Setup peak then ~1391 MiB (inside prepare_model_for_kbit_training: fp16 + fp32 embedding).

Training loop (diag run, peaks reset per step): reserved 592 at train begin, 1364 after step 1, 2012 at
train end, nvidia-smi 2191: reserved only grows during training, nvidia-smi ~ reserved + 160-180 MiB.
