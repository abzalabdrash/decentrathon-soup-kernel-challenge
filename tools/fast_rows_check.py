"""SOUP_SOLUTION_FAST_ROWS against Soup's own row formatting on hidden rows of the dataset.

    python tools/fast_rows_check.py . 51689 0 0     # every non-public row (~3 min, CPU only)
    python tools/fast_rows_check.py . 300 100 100   # a quick sample

For each row: Soup's alpaca conversion, then build_assistant_only_labels(messages, tokenizer, 512)
with the original loss_mask helpers and with trainer_build's replacements installed; input ids,
labels and attention mask must be equal. Public rows (task/prepare_data.py) are skipped."""
import random
import sys
import time

repo = sys.argv[1]
n_random, n_long, n_short = (int(x) for x in sys.argv[2:5])
sys.path.insert(0, repo)
sys.path.insert(0, repo + "/task")

from datasets import load_dataset  # noqa: E402
from prepare_data import DATASET, EVAL_ROWS, REVISION, TRAIN_ROWS  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from soup_cli.data import loss_mask  # noqa: E402
from soup_cli.data.formats import format_to_messages  # noqa: E402
from solution import trainer_build  # noqa: E402

ds = load_dataset(DATASET, revision=REVISION, split="train")
public = set(TRAIN_ROWS) | set(EVAL_ROWS)
lengths = [len(a) + len(b) + len(c) for a, b, c in zip(ds["instruction"], ds["input"], ds["output"])]
order = [i for i in sorted(range(len(ds)), key=lambda i: (lengths[i], i)) if i not in public]
rng = random.Random(83)
picked = sorted(set(rng.sample(order, n_random)) | set(order[-n_long:]) | set(order[:n_short]))
print(f"dataset rows {len(ds)}, non-public {len(order)}, checking {len(picked)}", flush=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
assert trainer_build._flag("SOUP_SOLUTION_FAST_ROWS", "1")

convs = []
for i in picked:
    row = {k: ds[i][k] for k in ("instruction", "input", "output")}
    convs.append((i, format_to_messages(row, "alpaca")))


def build(messages):
    try:
        return loss_mask.build_assistant_only_labels(messages, tok, 512)
    except Exception as exc:  # noqa: BLE001 - the same exception both ways is equal behaviour
        return ("raised", type(exc).__name__, str(exc))


start = time.perf_counter()
ref = [build(c["messages"]) if c else None for _, c in convs]
t_ref = time.perf_counter() - start
undo = trainer_build.install()
assert loss_mask._tokenize_only.__name__ == "_tokenize_only" and hasattr(loss_mask._tokenize_only, "_soup_original")
start = time.perf_counter()
fast = [build(c["messages"]) if c else None for _, c in convs]
t_fast = time.perf_counter() - start
undo()
assert not hasattr(loss_mask._tokenize_only, "_soup_original")

diff = [i for (i, _), a, b in zip(convs, ref, fast) if a != b]
raised = sum(1 for a in ref if isinstance(a, tuple))
capped = sum(1 for a in ref if isinstance(a, dict) and len(a["input_ids"]) >= 512)
print(f"rows {len(convs)}: different {len(diff)} {diff[:10]}; raised in both {raised}; at the 512 cap {capped}; "
      f"original {t_ref:.1f} s, fast {t_fast:.1f} s", flush=True)
sys.exit(1 if diff else 0)
