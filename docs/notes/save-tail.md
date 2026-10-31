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

**soup-e1-save2** (account 3, d79ad23, base vs once, ABBA 2 rounds + warm): in flight; fetch with
`bash tools/kaggle_win.sh 3 --attach --slug soup-e1-save2 --mode variants` and read the save tail as
above (`tail_stats`: train end - (first step begin + metrics.json train_loop_s)).

## State and what is left

- Merge rule (orchestrator): ABBA in one session >= 0.1 s, peak unchanged (371 MiB), bitwise. If it
  passes: `os.environ.setdefault("SOUP_SOLUTION_SAVE_ONCE", "1")` next to the exit release in
  `apply()`, then full + jury.
- Not done: the adapter's second save (`PeftModel.save_pretrained`, checkpoint and output directory)
  is not replayed: its README.md (peft's model card) is merged with whatever README the target
  directory holds, so equal bytes are not guaranteed without a test per directory; decide from the
  timed `adapter saved 2x in N ms` of the T4 run (worth it only if N/2 >= 0.1 s).
