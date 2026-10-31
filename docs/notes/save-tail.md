# Save tail: the tokenizer written once (branch claude/e1-exit-attn, R3 plan item 3)

## What changed

`solution/save_tail.py`, switch `SOUP_SOLUTION_SAVE_ONCE=1` (off by default), wired by one `if` in
`solution/__init__.py` after the exit release. Only in the `soup train` process
(`exit_release._soup_train_process`): pytest and Soup's 79 tests keep the original.

After the last step the run serializes the same tokenizer three times: `Trainer._save_checkpoint` ->
`save_model` -> `_save` -> `processing_class.save_pretrained(checkpoint-64)`, then Soup's
`trainer.save_model(output_dir)` (the same chain into the output directory) and Soup's own
`tokenizer.save_pretrained(output_dir)` (soup_cli/trainer/sft.py:2577, the same directory again).
Each call writes tokenizer_config.json, chat_template.jinja and the 11.4 MB tokenizer.json from the
Rust serializer (laptop: 61-95 ms per call; `backend_tokenizer.to_str()` alone 58 ms).

The patch of `PreTrainedTokenizerBase.save_pretrained` runs the original the first time on a
tokenizer object and remembers: the files it returned (taken only when exactly those files changed in
the directory during the call), their bytes, the returned tuple (relative names), the arguments
(directory type, legacy_format, filename_prefix, other kwargs) and a fingerprint of everything that
call serializes: type, the backend tokenizer's identity, size (added tokens), truncation and padding,
init kwargs and the attributes they name, special and added tokens, chat template, padding and
truncation sides, model_max_length, response_template, init_inputs, auto class/map, processor class,
name_or_path, vocab file names. A later call on the same object (weakref) with the same arguments,
the same fingerprint, push_to_hub off and a directory that is not a file writes the same bytes
(makedirs, then each file) and returns the tuple joined with its own directory, with the original's
two info log lines; anything else runs the original and becomes the new reference.

Bitwise: two saves of an unchanged tokenizer write the same bytes (checked on the real Qwen2.5
tokenizer: tokenizer.json 11,421,892 B, tokenizer_config.json 693 B, chat_template.jinja 2,507 B
equal), so the replayed files are what the original would write.

## Tests (`solution/test_save_tail.py`, the real tokenizer from the HF cache, skipped without it)

- the tail's three saves (checkpoint-64, output, output again): the six files and the three return
  tuples equal the original path's, 1 written + 2 replayed;
- another `filename_prefix`, an added token, another tokenizer object: the original (and the bytes
  after a change are the changed tokenizer's); a save right after the changed one replays it;
- backend truncation and the chat template are in the fingerprint;
- outside `soup train` nothing is patched. Soup's 79 tests pass with the switch on.

## T4

**soup-e1-tail1** (account 3 slerfas, e4b9f3c, variants ABBA 2 rounds + warm; order warm, base, once,
half, half, once, base). Save tail = last step end -> `phase train end` (SFTTrainerWrapper.train's
return: checkpoint save, Soup's save_model and tokenizer save):

| run | wall s | steps 3+ s | save tail s | train end -> exit s | exit line |
|---|---:|---:|---:|---:|---|
| base_1 | 41.53 | 23.44 | 0.99 | 0.17 | |
| once_1 | 42.54 | 24.63 | 0.92 | 0.17 | tokenizer written 1x in 166 ms, replayed 2x in 101 ms; adapter saved 2x in 122 ms |
| half_1 | 44.50 | 25.83 | 0.99 | 0.17 | |
| half_2 | 44.86 | 26.28 | 0.97 | 0.17 | |
| once_2 | 45.00 | 27.18 | 0.94 | 0.17 | tokenizer written 1x in 170 ms, replayed 2x in 118 ms; adapter saved 2x in 123 ms |
| base_2 | 45.04 | 26.87 | 1.02 | 0.18 | |

All 371 MiB nvidia-smi, reserved 250 / allocated 225.5 MiB in every run, checks 3, 4, 9 pass.
SAVE_ONCE (first version): save tail -0.075 s (0.99/1.02 -> 0.92/0.94). The wall cannot show it: this
T4 drifted +3.4 s in steps 3+ over the session (base_1 23.44 -> base_2 26.87), far above the effect.
The replay itself cost 50-59 ms per call (11.4 MB written on Kaggle's disk), so the second version
leaves the third save's files in place when they are exactly the bytes the second wrote (same inode,
mtime, size): expected about -0.13 s. A hard-link variant (-0.05 s more) was tried and dropped: an
in-place writer of either file would change both (a test showed it).

**soup-e1-save2** (account 3, d79ad23 = the second version, base vs once, ABBA 2 rounds + warm; order
warm, base, once, once, base):

| run | wall s | steps 3+ s | save tail s | train end -> exit s | exit line |
|---|---:|---:|---:|---:|---|
| base_1 | 40.67 | 23.66 | 0.91 | 0.16 | |
| once_1 | 43.02 | 26.21 | 0.87 | 0.13 | tokenizer written 1x in 158 ms, replayed 2x in 89 ms (files: 3 written, 3 already in place); adapter saved 2x in 123 ms |
| once_2 | 42.32 | 25.74 | 0.86 | 0.13 | tokenizer written 1x in 153 ms, replayed 2x in 90 ms (same) |
| base_2 | 42.13 | 25.40 | 0.90 | 0.16 | |

371 MiB, reserved 250, allocated 225.5 in all runs; checks 3, 4, 9 pass. Save tail -0.04 s, train end ->
exit -0.03 s (0.01 s resolution of the marks): about -0.07 s in all, the first session -0.075 s. The
wall again follows the session's drift (steps 3+ of once 26.2/25.7 against base 23.7/25.4 although the
switch changes nothing in the steps). Why so little: on Kaggle's disk writing the 11.4 MB tokenizer.json
costs ~80 ms by itself (the second save's replay: 85 ms of the 89; the third save, left in place,
~5 ms), and the original's second and third saves were cheaper than its first (warm serializer and
page cache), so the two saves the switch removes were worth ~0.12 s and their replacement costs 0.09 s.

**Verdict: below the merge rule (0.1 s); SAVE_ONCE stays off.** The R3 estimate (-0.15..-0.3 s) assumed
the copy is nearly free; on the T4 machines the write of the file dominates.

## State and what is left

- Off by default, in the branch only (save_tail.py, its tests, one `if` in apply()); nothing to merge
  unless a later measurement finds the write cheaper. To turn it on: `os.environ.setdefault(
  "SOUP_SOLUTION_SAVE_ONCE", "1")` next to the exit release in `apply()`.
- What the tail is made of on the T4 (last step end -> SFTTrainerWrapper.train return, 0.86-1.02 s in
  these sessions): the checkpoint (adapter ~60 ms, tokenizer ~155 ms, optimizer.pt, scheduler, scaler,
  RNG, trainer_state.json), Soup's `_assert_finite_training_state` (112 `.item()`), the final adapter
  (~60 ms) and tokenizer saves, training_args.bin twice. A per-call profile of this tail on the T4 would
  show what else is there; the tokenizer's share is ~0.35 s of it, the file writes most of that.
- Rejected: hard links instead of copies for the second directory (-0.05 s more): an in-place writer of
  either file would change both.
