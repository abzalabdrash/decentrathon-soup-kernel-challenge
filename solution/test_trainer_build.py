"""trainer_build.py: the fast row formatting gives Soup's rows, the bytecode scan gives dill's globals.

CPU only; the tokenizer is a small byte-level BPE built here (no download) with a ChatML template
without a `{% generation %}` block, as Qwen2.5's."""

import importlib
import sys

import pytest

loss_mask = pytest.importorskip("soup_cli.data.loss_mask")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from solution import trainer_build  # noqa: E402

CHATML = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    backend = Tokenizer(models.BPE(unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = ["Сәлем, қалайсың? Жақсы, рахмет.", "Translate the sentence into Kazakh.",
              "system user assistant", "The quick brown fox jumps over the lazy dog."] * 20
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    backend.train_from_iterator(corpus, trainer)
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
                                               pad_token="<|endoftext|>")
    tok.chat_template = CHATML
    return tok


ROWS = [
    [{"role": "user", "content": "Translate the sentence into Kazakh."},
     {"role": "assistant", "content": "Сәлем, қалайсың?"}],
    [{"role": "system", "content": "system"}, {"role": "user", "content": "quick brown fox"},
     {"role": "assistant", "content": "Жақсы, рахмет."}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
     {"role": "user", "content": "c"}, {"role": "assistant", "content": "the lazy dog"}],
]


def _labels(tok, max_length=512):
    return [loss_mask.build_assistant_only_labels(row, tok, max_length=max_length) for row in ROWS]


def test_fast_rows_same_labels(tokenizer, monkeypatch):
    calls = []
    original = loss_mask._tokenize_only
    monkeypatch.setattr(loss_mask, "_tokenize_only", lambda t, m: calls.append(len(m)) or original(t, m))
    expected = _labels(tokenizer)
    plain_calls = len(calls)
    expected_short = _labels(tokenizer, max_length=7)
    calls.clear()
    monkeypatch.setenv("SOUP_SOLUTION_FAST_ROWS", "1")
    undo = trainer_build.install()
    try:
        assert getattr(loss_mask._apply_template_with_mask, "_soup_original", None) is not None
        assert _labels(tokenizer) == expected
        # the whole conversation is tokenized once per row, not twice
        assert len(calls) == plain_calls - len(ROWS)
        # the same rows again: equal answers (from the memo or not)
        assert _labels(tokenizer, max_length=7) == expected_short
        # a changed message is not answered from the memo
        changed = [dict(m) for m in ROWS[0]]
        first = loss_mask._tokenize_only(tokenizer, changed)
        changed[1]["content"] = "the lazy dog"
        assert loss_mask._tokenize_only(tokenizer, changed) != first
    finally:
        undo()
    assert getattr(loss_mask._apply_template_with_mask, "_soup_original", None) is None
    assert getattr(loss_mask._coerce_int_list, "_soup_original", None) is None


def test_fast_mask_only_without_generation_block(tokenizer, monkeypatch):
    monkeypatch.setenv("SOUP_SOLUTION_FAST_ROWS", "1")
    undo = trainer_build.install()
    try:
        assert loss_mask._apply_template_with_mask(tokenizer, ROWS[0]) is None
        # no assistant message: the original runs (and returns the all-zero mask)
        prompt_only = [{"role": "user", "content": "a"}]
        ids, mask = loss_mask._apply_template_with_mask(tokenizer, prompt_only)
        assert ids and not any(mask)
        # a template with a generation block: the original runs and finds the assistant tokens
        generation = CHATML.replace("+ message['content'] +", "+ '' }}{% generation %}{{ message['content'] }}"
                                    "{% endgeneration %}{{ ''+")
        assert trainer_build._GENERATION.search(generation)
        saved = tokenizer.chat_template
        tokenizer.chat_template = generation
        try:
            out = loss_mask._apply_template_with_mask(tokenizer, ROWS[0])
            assert out is not None and any(out[1])
        finally:
            tokenizer.chat_template = saved

        class Fake:  # a stand-in tokenizer (Soup's tests use them): the original
            chat_template = CHATML

            def apply_chat_template(self, messages, **kwargs):
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 1]}

        assert loss_mask._apply_template_with_mask(Fake(), ROWS[0]) == ([1, 2, 3], [0, 1, 1])
    finally:
        undo()


def test_fast_int_list_matches_loop(monkeypatch):
    original = loss_mask._coerce_int_list
    monkeypatch.setenv("SOUP_SOLUTION_FAST_ROWS", "1")
    undo = trainer_build.install()
    try:
        fast = loss_mask._coerce_int_list
        assert fast is not original
        for values, allow_bool in (([3, 1, 4], False), ([], False), ((5, 6), False), ([True, 2], True),
                                   ([7], True)):
            assert fast(values, field="x", allow_bool=allow_bool) == original(values, field="x",
                                                                            allow_bool=allow_bool)
        source = [1, 2]
        assert fast(source, field="x") is not source
        for bad, allow_bool in (([1, True], False), ([1.5], False), (None, False), (["a"], False)):
            with pytest.raises(ValueError):
                original(bad, field="x", allow_bool=allow_bool)
            with pytest.raises(ValueError):
                fast(bad, field="x", allow_bool=allow_bool)
    finally:
        undo()


def _functions():
    names = ("soup_cli.data.loss_mask", "soup_cli.data.sft_format", "soup_cli.trainer.sft",
             "transformers.tokenization_utils_base", "trl.trainer.sft_trainer", "datasets.arrow_dataset",
             "datasets.fingerprint", "json", "inspect")
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - an optional module that does not import here
            continue
        for value in list(vars(module).values()):
            candidates = [value]
            if isinstance(value, type):
                candidates = [v for v in vars(value).values() if callable(v)]
            for fn in candidates:
                if getattr(fn, "__code__", None) is not None:
                    yield fn


def test_global_names_match_dill():
    detect = pytest.importorskip("dill.detect")
    compared = 0
    for fn in _functions():
        # dill parses the text of dis.dis: a quoted string constant that contains "_GLOBAL" (e.g.
        # inspect.getclosurevars' 'LOAD_GLOBAL') comes out as a "name" too, which is never a key of a
        # module's globals; the names that can be globals are the same
        found = {name for name in detect.nestedglobals(fn) if name.isidentifier()}
        assert trainer_build.global_names(fn.__code__) == found, fn
        compared += 1
    assert compared > 200


def test_fast_fingerprint_same_hash(monkeypatch):
    fingerprint = pytest.importorskip("datasets.fingerprint")
    import json

    def transform(example, row_index):
        return {"text": json.dumps(example) + str(row_index) + str(sys.maxsize)}

    expected = fingerprint.Hasher.hash(transform)
    monkeypatch.setenv("SOUP_SOLUTION_FAST_FINGERPRINT", "1")
    undo = trainer_build.install()
    try:
        import dill.detect

        assert getattr(dill.detect.nestedglobals, "_soup_original", None) is not None
        assert fingerprint.Hasher.hash(transform) == expected
    finally:
        undo()


TASK_YAML = """
base: Qwen/Qwen2.5-1.5B-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.0
  max_length: 64
training:
  epochs: 1
  batch_size: 4
  lora:
    r: 16
    alpha: 32
  quantization: 4bit
output: ./runs/out
"""


def test_early_rows_same_dataset(tokenizer, tmp_path, monkeypatch):
    """SOUP_SOLUTION_EARLY_ROWS: the tokenizer load inside Soup's setup starts the thread, and Soup's
    build_format_row / _map_text_sft_rows calls with the same objects get its results: the same rows
    as Soup's own tokenization."""
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.data import sft_format
    from soup_cli.trainer import sft as soup_sft

    tokenizer.save_pretrained(tmp_path)
    cfg = load_config_from_string(TASK_YAML)
    rows = [{"messages": list(row)} for row in ROWS] * 3

    class Wrapper:
        config = cfg

    reference_tok = transformers.AutoTokenizer.from_pretrained(tmp_path)
    reference_fr = sft_format.build_format_row(tokenizer=reference_tok, data_cfg=cfg.data,
                                               console=soup_sft.console, training_cfg=cfg.training)
    expected = soup_sft._map_text_sft_rows(rows, format_row=reference_fr, split="train",
                                           max_length=cfg.data.max_length).to_list()

    undo = trainer_build._install_early(Wrapper(), ({"train": rows},), {})
    assert undo is not None
    try:
        early = trainer_build._EARLY["current"]
        tok = transformers.AutoTokenizer.from_pretrained(tmp_path)
        assert early.thread is not None  # started by the tokenizer load
        fr = sft_format.build_format_row(tokenizer=tok, data_cfg=cfg.data, console=soup_sft.console,
                                         training_cfg=cfg.training)
        assert fr is early.format_row
        got = soup_sft._map_text_sft_rows(rows, format_row=fr, split="train", max_length=cfg.data.max_length)
        assert got is early.result
        assert got.to_list() == expected
        masked = soup_sft._ensure_assistant_masks(got)
        assert masked is early.masked
        assert masked.to_list() == soup_sft._ensure_assistant_masks._soup_original(got).to_list()
        # another row list or another format_row: Soup's own call
        other = soup_sft._map_text_sft_rows(rows[:2], format_row=fr, split="train",
                                            max_length=cfg.data.max_length)
        assert other is not early.result and other.to_list() == expected[:2]
    finally:
        undo()
    assert trainer_build._EARLY["current"] is None
    assert getattr(sft_format.build_format_row, "_soup_original", None) is None
    assert getattr(soup_sft._map_text_sft_rows, "_soup_original", None) is None
    assert "from_pretrained" in transformers.AutoTokenizer.__dict__
    assert getattr(transformers.AutoTokenizer.__dict__["from_pretrained"].__func__, "_soup_original", None) is None


def test_early_rows_not_for_changed_tokenizer(tokenizer, tmp_path):
    """A tokenizer changed after the thread started (a token added): Soup tokenizes itself."""
    from soup_cli.config.loader import load_config_from_string
    from soup_cli.data import sft_format
    from soup_cli.trainer import sft as soup_sft

    tokenizer.save_pretrained(tmp_path)
    cfg = load_config_from_string(TASK_YAML)
    rows = [{"messages": list(row)} for row in ROWS]

    class Wrapper:
        config = cfg

    undo = trainer_build._install_early(Wrapper(), ({"train": rows},), {})
    try:
        early = trainer_build._EARLY["current"]
        tok = transformers.AutoTokenizer.from_pretrained(tmp_path)
        early.thread.join()
        tok.add_tokens(["<new_token>"])
        fr = sft_format.build_format_row(tokenizer=tok, data_cfg=cfg.data, console=soup_sft.console,
                                         training_cfg=cfg.training)
        assert fr is not early.format_row
    finally:
        undo()


def test_early_probes_serve_same_answers(monkeypatch):
    """SOUP_SOLUTION_EARLY_PROBES: entry_points() and accelerate's P2P check answered from the probe
    thread give what the calls give, and only while a run's probes exist and sys.path is unchanged."""
    import importlib.metadata as metadata

    accelerate_state = pytest.importorskip("accelerate.state")
    saved_ep, saved_p2p = metadata.entry_points, accelerate_state.check_cuda_p2p_ib_support
    saved_flag = trainer_build._PROBES["patched"]
    trainer_build._PROBES["patched"] = False
    try:
        trainer_build._patch_probes()
        assert metadata.entry_points is not saved_ep
        expected_all = saved_ep()
        expected_group = saved_ep(group="console_scripts")
        # no probes: the originals
        assert trainer_build._PROBES["current"] is None
        assert sorted(metadata.entry_points(group="console_scripts")) == sorted(expected_group)
        probes = trainer_build._Probes()
        probes._work()
        trainer_build._PROBES["current"] = probes
        assert probes.entry_points == expected_all
        assert metadata.entry_points(group="console_scripts") == expected_group
        assert accelerate_state.check_cuda_p2p_ib_support() == probes.p2p
        monkeypatch.setattr(sys, "path", sys.path + ["/nonexistent-soup-test-dir"])
        assert metadata.entry_points(group="console_scripts") == expected_group  # the original again
    finally:
        trainer_build._PROBES["current"] = None
        metadata.entry_points = saved_ep
        accelerate_state.check_cuda_p2p_ib_support = saved_p2p
        trainer_build._PROBES["patched"] = saved_flag


def test_cheap_fingerprint_deterministic(tokenizer, monkeypatch):
    """SOUP_SOLUTION_CHEAP_FINGERPRINT: the map's rows are the same, its fingerprint is the same for
    the same inputs (no RNG), differs for another function, and datasets' own hashing is not used."""
    datasets = pytest.importorskip("datasets")
    import random

    import datasets.fingerprint as fingerprint

    rows = [{"messages": list(row)} for row in ROWS]

    def first(example, index):
        return {"n": len(example["messages"]) + index}

    def second(example, index):
        return {"n": len(example["messages"]) * 2 + index}

    plain = datasets.Dataset.from_list(rows).map(first, with_indices=True, remove_columns=["messages"])
    monkeypatch.setenv("SOUP_SOLUTION_CHEAP_FINGERPRINT", "1")
    undo = trainer_build.install()
    try:
        import dill.detect

        calls = []
        globalvars = dill.detect.globalvars
        monkeypatch.setattr(dill.detect, "globalvars", lambda *a, **k: calls.append(a) or globalvars(*a, **k))
        state = random.getstate()
        a = datasets.Dataset.from_list(rows)
        a = a.map(first, with_indices=True, remove_columns=["messages"])
        b = datasets.Dataset.from_list(rows).map(first, with_indices=True, remove_columns=["messages"])
        c = datasets.Dataset.from_list(rows).map(second, with_indices=True, remove_columns=["messages"])
        assert random.getstate() == state
        assert a.to_list() == plain.to_list()
        assert a._fingerprint == b._fingerprint != c._fingerprint
        # the map functions were not pickled with their globals (datasets still hashes its tables)
        assert not [c for c in calls if c and getattr(c[0], "__name__", "") in ("first", "second")]
    finally:
        undo()
    assert getattr(fingerprint.update_fingerprint, "_soup_original", None) is None


def test_probes_alone_do_not_tokenize_early(tokenizer, tmp_path):
    """SOUP_SOLUTION_EARLY_PROBES without SOUP_SOLUTION_EARLY_ROWS: the tokenizer load starts no
    early tokenization (a list of rows once shadowed the flag)."""
    from soup_cli.config.loader import load_config_from_string

    tokenizer.save_pretrained(tmp_path)
    cfg = load_config_from_string(TASK_YAML)

    class Wrapper:
        config = cfg

    undo = trainer_build._install_early(Wrapper(), ({"train": [{"messages": list(ROWS[0])}]},), {},
                                        rows=False, probes=False)
    try:
        early = trainer_build._EARLY["current"]
        transformers.AutoTokenizer.from_pretrained(tmp_path)
        assert early.thread is None
    finally:
        undo()
