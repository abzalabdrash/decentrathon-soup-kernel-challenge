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

(filled below)
