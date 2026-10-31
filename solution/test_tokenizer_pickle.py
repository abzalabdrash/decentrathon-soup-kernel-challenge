"""SOUP_SOLUTION_TOKENIZER_PICKLE (tokenizer_pickle.py): datasets' pickle of a tokenizer, and so every
fingerprint that hashes one, is the same byte for byte with and without the same-bytes writer."""

import pytest

transformers = pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from datasets.fingerprint import Hasher  # noqa: E402
from datasets.utils import _dill  # noqa: E402

from solution import tokenizer_pickle  # noqa: E402


@pytest.fixture(autouse=True)
def datasets_own_reducers():
    """Every tokenizer type in datasets' dispatch with datasets' own reducer during a test (another
    test file of the same process may leave startup.py's DATASET_HASH reducer there)."""
    original = _dill._save_transformersPreTrainedTokenizerBase
    dispatch = _dill.Pickler.dispatch
    saved = {}
    for kind, fn in list(dispatch.items()):
        if isinstance(kind, type) and issubclass(kind, transformers.PreTrainedTokenizerBase) and fn is not original:
            saved[kind] = fn
            dispatch[kind] = original
    yield
    for kind, fn in saved.items():
        dispatch[kind] = fn


@pytest.fixture(scope="module")
def small_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["Сәлем, қалайсың? Жақсы, рахмет.", "Translate the sentence into Kazakh.",
              "The quick brown fox jumps over the lazy dog."] * 30
    trainer = trainers.BpeTrainer(vocab_size=900, special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator(corpus, trainer)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>")
    # the Python copies a tokenizer loaded from files keeps (Qwen2Tokenizer: `_vocab`, `_merges`)
    import json

    merges = json.loads(backend.to_str())["model"]["merges"]
    tok.__dict__["_vocab"] = dict(backend.get_vocab())
    tok.__dict__["_merges"] = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges]
    return tok


@pytest.fixture(scope="module")
def qwen_tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001 - not downloaded here
        pytest.skip("Qwen2.5 tokenizer not in the local cache")


def _streams(tok):
    """Objects whose pickles hold the tokenizer at different places of the stream: alone, twice, after
    payloads of several sizes (another frame alignment), its vocabulary and merges referenced before
    and after it, inside a closure as Soup's map holds it."""
    vocab = tok.__dict__.get("_vocab")
    merges = tok.__dict__.get("_merges")

    def format_row(example):
        return tok(example["text"])

    def checked(example, index):
        return format_row(example)

    out = [tok, [tok, tok], format_row, checked]
    out += [[b"x" * size, tok] for size in (1, 4093, 65521, 65536, 70001)]
    if merges:
        out += [[merges, tok, merges[3]], [merges[10], tok]]
    if vocab is not None:
        out += [[vocab, tok, vocab]]
    return out


def _dumps_both(objects):
    plain = [_dill.dumps(obj) for obj in objects]
    undo = tokenizer_pickle.install()
    assert undo is not None, tokenizer_pickle.LAST
    try:
        before = tokenizer_pickle.LAST["fast"]
        fast = [_dill.dumps(obj) for obj in objects]
        taken = tokenizer_pickle.LAST["fast"] - before
    finally:
        undo()
    return plain, fast, taken


def test_self_check():
    assert tokenizer_pickle._self_check()


def test_same_bytes_small_tokenizer(small_tokenizer):
    objects = _streams(small_tokenizer)
    plain, fast, taken = _dumps_both(objects)
    assert taken > 0  # the writer was used
    for i, (a, b) in enumerate(zip(plain, fast)):
        assert a == b, f"stream {i} differs"


def test_same_bytes_and_fingerprint_qwen(qwen_tokenizer):
    tok = qwen_tokenizer
    objects = [tok, [b"y" * 777, tok], [tok.__dict__["_merges"][5], tok, tok]]
    plain, fast, taken = _dumps_both(objects)
    assert taken == len(objects) and tokenizer_pickle.LAST["fallback"] is None
    assert plain == fast

    def checked_format_row(example, row_index):  # the shape of Soup's map function: a closure over it
        return tok(example["text"])

    expected = Hasher.hash(checked_format_row)
    undo = tokenizer_pickle.install()
    try:
        got = Hasher.hash(checked_format_row)
    finally:
        undo()
    assert got == expected


def test_soup_map_fingerprint_qwen(qwen_tokenizer):
    """Soup's own tokenization map of the task's rows: the same fingerprint and the same rows."""
    import json
    from pathlib import Path

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.data import sft_format
    from soup_cli.trainer import sft as soup_sft

    rows_file = Path(__file__).resolve().parents[1] / "data" / "train.jsonl"
    if not rows_file.exists():
        pytest.skip("data/train.jsonl not prepared")
    cfg = load_config_from_string(TASK_YAML)
    raw = [json.loads(line) for line in rows_file.read_text(encoding="utf-8").splitlines()[:16] if line.strip()]
    rows = [{"messages": [{"role": "user", "content": (r.get("instruction", "") + "\n" + r.get("input", "")).strip()},
                          {"role": "assistant", "content": r.get("output", "")}]} for r in raw]

    def run():
        fr = sft_format.build_format_row(tokenizer=qwen_tokenizer, data_cfg=cfg.data, console=soup_sft.console,
                                         training_cfg=cfg.training)
        return soup_sft._map_text_sft_rows(rows, format_row=fr, split="train", max_length=cfg.data.max_length)

    expected = run()
    undo = tokenizer_pickle.install()
    try:
        before = tokenizer_pickle.LAST["fast"]
        got = run()
        assert tokenizer_pickle.LAST["fast"] > before
    finally:
        undo()
    assert got._fingerprint == expected._fingerprint
    assert got.to_list() == expected.to_list()


def test_ineligible_containers_fall_back(small_tokenizer):
    """A vocabulary with a value the writer does not reproduce: datasets' own reducer, same bytes."""
    tok = small_tokenizer
    vocab = tok.__dict__.get("_vocab")
    if vocab is None:
        pytest.skip("this tokenizer keeps no Python vocabulary")
    tok.__dict__["_vocab"] = dict(vocab, odd=2 ** 40)
    try:
        plain, fast, _ = _dumps_both([tok])
    finally:
        tok.__dict__["_vocab"] = vocab
    assert plain == fast


def test_install_undo_restores_dispatch(small_tokenizer):
    _dill.dumps(small_tokenizer)  # datasets registers its reducer for the type
    original = _dill._save_transformersPreTrainedTokenizerBase
    kind = type(small_tokenizer)
    assert _dill.Pickler.dispatch.get(kind) is original
    undo = tokenizer_pickle.install()
    assert _dill.Pickler.dispatch.get(kind) is not original
    assert tokenizer_pickle.install() is None  # once
    undo()
    assert _dill._save_transformersPreTrainedTokenizerBase is original
    assert _dill.Pickler.dispatch.get(kind) is original
    assert tokenizer_pickle._Marker not in _dill.Pickler.dispatch


def test_trainer_build_scopes_it_to_setup(monkeypatch, small_tokenizer):
    """trainer_build.py installs it for Soup's setup only, when the switch is on."""
    from solution import trainer_build

    monkeypatch.setenv("SOUP_SOLUTION_TOKENIZER_PICKLE", "1")
    assert tokenizer_pickle.enabled()
    monkeypatch.setenv("SOUP_SOLUTION_TOKENIZER_PICKLE", "0")
    assert not tokenizer_pickle.enabled()
    monkeypatch.delenv("SOUP_SOLUTION_TOKENIZER_PICKLE")
    assert not tokenizer_pickle.enabled()  # off by default
    assert callable(trainer_build.patch_soup_trainer_build)


TASK_YAML = """
base: Qwen/Qwen2.5-1.5B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.0
  max_length: 512
training:
  epochs: 1
  batch_size: 4
  lora:
    r: 16
    alpha: 32
  quantization: 4bit
output: ./runs/out
"""
