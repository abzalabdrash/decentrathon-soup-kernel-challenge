"""Alternative train / eval files from other rows of the task's dataset, to emulate the jury's run.

    python tools/jury_data.py --out data_jury --train-rows 10000:10256 --eval-rows 50000:50256
    python tools/jury_data.py --out data_jury --select longest     # the 256 longest rows train, the next 256 eval
    python tools/jury_data.py --out data_jury --select extremes    # shortest and longest rows interleaved

The jury trains and evaluates on hidden rows of AmanMussa/kazakh-instruction-v2 with another seed
(harness/run_task.py --train-file --eval-file --seed --repeats 2). Memory and the padded token count
scale with the longest batch, and the packed path has edge cases in very short rows, so besides
plain row ranges this picks the longest rows (every sequence near max_length 512: batches of ~2048
tokens, against 1,427 at most in the public rows) or the shortest and longest interleaved (batches
mixing one-token answers with full rows). The public rows (task/prepare_data.py) are never picked.

Writes train.jsonl and eval.jsonl in task/prepare_data.py's format, and jury_data.log: the rows
picked and their estimated token counts (Qwen's chat template over a user turn of instruction and
input and an assistant turn of output, truncated at 512; Soup's own formatting differs by a few
tokens, the runs' rewind.jsonl has the exact counts), per batch of 4 in file order.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "task"))
from prepare_data import DATASET, EVAL_ROWS, REVISION, TRAIN_ROWS, write_jsonl  # noqa: E402

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LENGTH = 512
MAX_PROMPT = 448  # Soup refuses a row whose answer is truncated away at max_length: keep 64 of it
COUNT = 256
PUBLIC = set(TRAIN_ROWS) | set(EVAL_ROWS)


def span(text):
    start, _, end = text.partition(":")
    return range(int(start), int(end))


def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL)


def _ids(tokenizer, messages, **kwargs):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    return ids["input_ids"] if hasattr(ids, "keys") else ids  # transformers 5: an encoding


def _user(row):
    return row["instruction"] + (("\n\n" + row["input"]) if row["input"] else "")


def _trainable(dataset, ranked, needed):
    """The first `needed` rows of `ranked` whose prompt leaves at least MAX_LENGTH - MAX_PROMPT
    tokens of the answer ("no causal-loss target remains after truncation" otherwise: Soup stops)."""
    tokenizer = _tokenizer()
    kept = []
    for i in ranked:
        prompt = _ids(tokenizer, [{"role": "user", "content": _user(dataset[i])}],
                      add_generation_prompt=True)
        if len(prompt) <= MAX_PROMPT:
            kept.append(i)
            if len(kept) == needed:
                break
    return kept


def pick(dataset, select, count):
    """(train rows, eval rows) by character length of instruction + input + output."""
    lengths = [len(a) + len(b) + len(c) for a, b, c in
               zip(dataset["instruction"], dataset["input"], dataset["output"])]
    order = [i for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)) if i not in PUBLIC]
    if select == "longest":
        ranked = _trainable(dataset, order[::-1], 2 * count)
        return ranked[:count], ranked[count:2 * count]
    if select == "shortest":
        return order[:count], order[count:2 * count]
    if select == "extremes":
        half = count // 2
        short, long_ = order[:count], _trainable(dataset, order[::-1], count)
        train = [row for pair in zip(short[:half], long_[:half]) for row in pair]
        evals = [row for pair in zip(short[half:], long_[half:]) for row in pair]
        return train, evals
    raise SystemExit(f"--select must be longest, shortest or extremes, got {select!r}")


def token_counts(rows):
    try:
        tokenizer = _tokenizer()
    except Exception as exc:  # noqa: BLE001 - the counts are informative only
        return None, f"no tokenizer ({exc.__class__.__name__}): counts not estimated"
    counts = [min(len(_ids(tokenizer, [{"role": "user", "content": _user(row)},
                                       {"role": "assistant", "content": row["output"]}])),
                  MAX_LENGTH) for row in rows]
    return counts, ""


def describe(name, indices, rows):
    counts, note = token_counts(rows)
    lines = [f"{name}: {len(indices)} rows, first {indices[:8]}"]
    if counts is None:
        return lines + [f"  {note}"]
    batches = [sum(counts[i:i + 4]) for i in range(0, len(counts), 4)]
    lines.append(f"  estimated tokens per row: min {min(counts)}, median {sorted(counts)[len(counts) // 2]}, "
                 f"max {max(counts)}; rows at the 512 cap {sum(c >= MAX_LENGTH for c in counts)}; "
                 f"rows under 8 tokens {sum(c < 8 for c in counts)}")
    lines.append(f"  per batch of 4 in file order (the trainer shuffles with the seed): max {max(batches)}, "
                 f"padded to the longest row up to {max(4 * max(counts[i:i + 4]) for i in range(0, len(counts), 4))}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(REPO / "data_jury"))
    parser.add_argument("--train-rows", type=span, help="START:END of the dataset rows for training")
    parser.add_argument("--eval-rows", type=span, help="START:END of the dataset rows for eval")
    parser.add_argument("--select", choices=("longest", "shortest", "extremes"))
    parser.add_argument("--count", type=int, default=COUNT)
    args = parser.parse_args()
    if bool(args.select) == bool(args.train_rows or args.eval_rows):
        parser.error("give --select, or --train-rows and --eval-rows")

    from datasets import load_dataset

    dataset = load_dataset(DATASET, split="train", revision=REVISION)
    if args.select:
        train, evals = pick(dataset, args.select, args.count)
    else:
        train, evals = list(args.train_rows), list(args.eval_rows)
        clash = (set(train) | set(evals)) & PUBLIC
        if clash or set(train) & set(evals):
            parser.error(f"rows overlap the public rows or each other: {sorted(clash)[:5]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_rows, eval_rows = [dataset[i] for i in train], [dataset[i] for i in evals]
    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "eval.jsonl", eval_rows)
    lines = [f"{DATASET}@{REVISION[:8]}: {len(dataset)} rows; " + " ".join(sys.argv[1:])]
    lines += describe("train", train, train_rows) + describe("eval", evals, eval_rows)
    (out / "jury_data.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps({"train": str(out / "train.jsonl"), "eval": str(out / "eval.jsonl")}))


if __name__ == "__main__":
    main()
