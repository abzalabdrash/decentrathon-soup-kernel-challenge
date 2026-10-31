"""SOUP_SOLUTION_SAVE_ONCE (save_tail.py): the save tail's three tokenizer saves (a checkpoint
directory, then twice the output directory) give the same files, bit for bit, and the same return
values as the original; a changed tokenizer, other arguments or another object run the original;
outside the `soup train` process nothing is patched."""

import os

import pytest

from solution import save_tail


def _tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the task's tokenizer is not in the local cache: {exc!r}")


def _files(directory):
    out = {}
    for root, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(root, name)
            with open(path, "rb") as f:
                out[os.path.relpath(path, directory)] = f.read()
    return out


def _tail(tok, root):
    """The three saves of the tail: checkpoint, final output, final output again."""
    ckpt, final = os.path.join(root, "out", "checkpoint-64"), os.path.join(root, "out")
    return [tok.save_pretrained(ckpt), tok.save_pretrained(final), tok.save_pretrained(final)]


@pytest.fixture
def patched():
    save_tail.unpatch_tokenizer_save()
    save_tail._STATE.update(written=0, replayed=0)
    yield
    save_tail.unpatch_tokenizer_save()


def _relative(results, root):
    return [tuple(os.path.relpath(p, root) for p in r) for r in results]


def test_tail_same_files_and_results(tmp_path, patched):
    tok = _tokenizer()
    want = _tail(tok, str(tmp_path / "orig"))
    save_tail.patch_tokenizer_save()
    got = _tail(tok, str(tmp_path / "once"))
    assert save_tail._STATE["written"] == 1 and save_tail._STATE["replayed"] == 2
    assert _relative(got, str(tmp_path / "once")) == _relative(want, str(tmp_path / "orig"))
    assert all(isinstance(p, str) for r in got for p in r)
    orig, once = _files(tmp_path / "orig"), _files(tmp_path / "once")
    assert orig.keys() == once.keys() and len(orig) == 6
    for name in orig:
        assert orig[name] == once[name], name


def test_changed_tokenizer_or_arguments_take_the_original(tmp_path, patched):
    tok = _tokenizer()
    save_tail.patch_tokenizer_save()
    tok.save_pretrained(str(tmp_path / "a"))
    tok.save_pretrained(str(tmp_path / "b"), filename_prefix="x")  # other arguments
    assert save_tail._STATE["replayed"] == 0 and save_tail._STATE["written"] == 2
    tok.add_tokens(["<soup_new_token>"])  # a changed state
    tok.save_pretrained(str(tmp_path / "c"))
    assert save_tail._STATE["replayed"] == 0 and save_tail._STATE["written"] == 3
    assert b"<soup_new_token>" in (tmp_path / "c" / "tokenizer.json").read_bytes()
    tok.save_pretrained(str(tmp_path / "d"))  # unchanged since c: replayed, the same bytes as c
    assert save_tail._STATE["replayed"] == 1
    assert _files(tmp_path / "c") == _files(tmp_path / "d")
    other = _tokenizer()  # another object with the first state: its own first save
    other.save_pretrained(str(tmp_path / "e"))
    assert save_tail._STATE["written"] == 4
    assert _files(tmp_path / "e") == _files(tmp_path / "a")


def test_padding_state_and_template_are_in_the_fingerprint(tmp_path, patched):
    tok = _tokenizer()
    save_tail.patch_tokenizer_save()
    tok.save_pretrained(str(tmp_path / "a"))
    tok.backend_tokenizer.enable_truncation(128)  # serialized in tokenizer.json
    tok.save_pretrained(str(tmp_path / "b"))
    tok.chat_template = "{{ messages }}"
    tok.save_pretrained(str(tmp_path / "c"))
    assert save_tail._STATE["replayed"] == 0 and save_tail._STATE["written"] == 3
    assert (tmp_path / "a" / "tokenizer.json").read_bytes() != (tmp_path / "b" / "tokenizer.json").read_bytes()
    assert (tmp_path / "c" / "chat_template.jinja").read_text() == "{{ messages }}"


def test_only_in_the_soup_train_process(patched):
    save_tail.patch_soup_save_tail()  # pytest's argv: nothing patched
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    assert not getattr(PreTrainedTokenizerBase, "_soup_save_once", False)
