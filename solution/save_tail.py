"""The tokenizer written once in the save tail of the run (SOUP_SOLUTION_SAVE_ONCE=1; R3 plan, item 3).

After the last step the same tokenizer is serialized three times: the Trainer's checkpoint
(`Trainer._save_checkpoint` -> `save_model` -> `_save` -> `processing_class.save_pretrained`, into
`checkpoint-64/`), Soup's final `trainer.save_model(output_dir)` (the same chain, into the output
directory) and Soup's own `tokenizer.save_pretrained(output_dir)` (the same directory again,
soup_cli/trainer/sft.py). Each call rebuilds tokenizer_config.json, chat_template.jinja and the
11.4 MB tokenizer.json (the Rust serializer: 60-95 ms on the laptop, most of a call).

Here `PreTrainedTokenizerBase.save_pretrained` runs as before the first time on a tokenizer and
remembers what it wrote: the files it returned (all of them written by that call, nothing else in
the directory changed), their bytes, the returned tuple, the arguments and a fingerprint of every
tokenizer state that call serialized (the init kwargs and the attributes they name, the special
and added tokens, the chat template, the backend's identity, size, truncation and padding, ...).
A later call on the same tokenizer object with the same arguments and the same fingerprint writes
the same bytes into its directory (makedirs, then each file, as the original's) and returns the same
tuple with its directory, the original's two info log lines included. A file that already holds
these bytes because this module (or the first save) wrote them there and nobody touched it since
(same inode, mtime and size: Soup's third save, into the directory of the second) is left as it is; anything else (another
object, a changed state, other arguments, push_to_hub, a file path as the directory) runs the
original. Every file is bit for bit what the original would write: two saves of one unchanged
tokenizer give the same bytes (tested, also on the real Qwen2.5 tokenizer).

Only in the `soup train` process (as exit_release.py's collector switch): pytest and Soup's own
tests keep the original. A log line at exit says how many saves were written and how many replayed.
"""

import atexit
import os
import time
import weakref

_STATE = {"patched": False, "entry": None, "written": 0, "replayed": 0, "write_s": 0.0,
          "replay_s": 0.0, "reported": False, "adapter": [0, 0.0], "files": [0, 0],
          "own": {}}  # path -> its stat key right after a replay wrote it
CONFIG_SUFFIX = "tokenizer_config.json"


def _fingerprint(tok):
    """Everything of `tok` that save_pretrained serializes, cheaply: a changed value means another
    save. The backend tokenizer is taken by identity plus the state that can change on it (added
    tokens through its size, truncation and padding)."""
    backend = getattr(tok, "_tokenizer", None)
    init = getattr(tok, "init_kwargs", {})
    named = [(k, repr(getattr(tok, k, None))) for k in sorted(init)]
    return (
        type(tok), id(backend), repr(sorted(init.items())), repr(named), len(tok),
        getattr(tok, "vocab_size", None), repr(getattr(tok, "chat_template", None)),
        repr(sorted(tok.added_tokens_decoder.items())), repr(tok.special_tokens_map),
        repr(getattr(tok, "_extra_special_tokens", None)), getattr(tok, "padding_side", None),
        getattr(tok, "truncation_side", None), getattr(tok, "model_max_length", None),
        repr(getattr(tok, "response_template", None)), repr(getattr(tok, "init_inputs", None)),
        repr(getattr(tok, "_auto_class", None)), repr(getattr(tok, "_auto_map", None)),
        repr(getattr(tok, "_processor_class", None)), repr(getattr(tok, "name_or_path", None)),
        repr(getattr(backend, "truncation", None)), repr(getattr(backend, "padding", None)),
        repr(getattr(tok, "vocab_files_names", None)),
    )


def _listing(directory):
    """{relative path: (mtime_ns, size)} of the files under `directory` (empty when it is missing)."""
    found = {}
    if not os.path.isdir(directory):
        return found
    for root, _, names in os.walk(directory):
        for name in names:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            found[os.path.relpath(path, directory)] = (st.st_mtime_ns, st.st_size)
    return found


def _stat_key(path):
    """(inode, device, mtime_ns, size) of `path`, None when it is missing: an equal key means nobody
    wrote the file in between."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_ino, st.st_dev, st.st_mtime_ns, st.st_size)


def _args_key(save_directory, legacy_format, filename_prefix, kwargs):
    return (type(save_directory), legacy_format, filename_prefix, repr(sorted(kwargs.items())))


def _remember(tok, save_directory, key, result, before):
    """The entry for later calls when this call wrote exactly the files it returned, else None."""
    if not isinstance(result, tuple) or not result or not all(isinstance(p, str) for p in result):
        return None
    directory = os.fspath(save_directory)
    rels = []
    for path in result:
        rel = os.path.relpath(path, directory)
        if rel.startswith(os.pardir) or os.path.isabs(rel) or os.path.join(save_directory, rel) != path:
            return None
        rels.append(rel)
    after = _listing(directory)
    changed = {rel for rel, stat in after.items() if before.get(rel) != stat}
    if changed != set(rels) or len(set(rels)) != len(rels):
        return None
    contents = []
    for rel in rels:
        with open(os.path.join(directory, rel), "rb") as f:
            contents.append(f.read())
    return {"ref": weakref.ref(tok), "fingerprint": _fingerprint(tok), "key": key, "dir": directory,
            "rels": tuple(rels), "contents": tuple(contents),
            "stats": tuple(_stat_key(os.path.join(directory, rel)) for rel in rels)}


def _replay(entry, save_directory):
    from transformers.tokenization_utils_base import logger

    os.makedirs(save_directory, exist_ok=True)
    paths = []
    counts = _STATE["files"]  # written, already in place
    for rel, data, first_key in zip(entry["rels"], entry["contents"], entry["stats"]):
        path = os.path.join(save_directory, rel)
        now = _stat_key(path)
        first = os.path.abspath(path) == os.path.abspath(os.path.join(entry["dir"], rel))
        if now is not None and (now == _STATE["own"].get(path) or (first and now == first_key)):
            counts[1] += 1  # these very bytes are there, untouched since they were written
        else:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            counts[0] += 1
            _STATE["own"][path] = _stat_key(path)
        if rel.endswith(CONFIG_SUFFIX):
            logger.info(f"tokenizer config file saved in {path}")
        elif rel.endswith(".jinja"):
            logger.info(f"chat template saved in {path}")
        paths.append(path)
    return tuple(paths)


def _matches(entry, tok, save_directory, key, push_to_hub):
    return (entry is not None and not push_to_hub and entry["ref"]() is tok and entry["key"] == key
            and not os.path.isfile(save_directory) and entry["fingerprint"] == _fingerprint(tok))


def patch_tokenizer_save():
    """PreTrainedTokenizerBase.save_pretrained writing an unchanged tokenizer's files from the first
    save's bytes (see the module doc). Idempotent."""
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    cls = PreTrainedTokenizerBase
    if getattr(cls, "_soup_save_once", False):
        return
    original = cls.save_pretrained

    def save_pretrained(self, save_directory, legacy_format=None, filename_prefix=None,
                        push_to_hub=False, **kwargs):
        started = time.perf_counter()
        try:
            key = _args_key(save_directory, legacy_format, filename_prefix, kwargs)
            entry = _STATE["entry"]
            if _matches(entry, self, save_directory, key, push_to_hub):
                result = _replay(entry, save_directory)
                _STATE["replayed"] += 1
                _STATE["replay_s"] += time.perf_counter() - started
                return result
            before = _listing(os.fspath(save_directory)) if not push_to_hub else None
        except Exception:  # noqa: BLE001 - anything unexpected: the original, untouched
            return original(self, save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
        result = original(self, save_directory, legacy_format, filename_prefix, push_to_hub, **kwargs)
        if not push_to_hub:
            try:
                _STATE["entry"] = _remember(self, save_directory, key, result, before)
            except Exception:  # noqa: BLE001
                _STATE["entry"] = None
        _STATE["written"] += 1
        _STATE["write_s"] += time.perf_counter() - started
        return result

    save_pretrained.__doc__ = original.__doc__
    cls.save_pretrained = save_pretrained
    cls._soup_original_save_pretrained = original
    cls._soup_save_once = True
    _STATE["patched"] = True
    if not _STATE["reported"]:
        _STATE["reported"] = True
        atexit.register(_report)


def unpatch_tokenizer_save():
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    cls = PreTrainedTokenizerBase
    if getattr(cls, "_soup_save_once", False):
        cls.save_pretrained = cls._soup_original_save_pretrained
        del cls._soup_original_save_pretrained
        cls._soup_save_once = False
    _STATE.update(patched=False, entry=None, own={}, files=[0, 0])


def _report():
    if _STATE["written"] or _STATE["replayed"]:
        calls, seconds = _STATE["adapter"]
        written, kept = _STATE["files"]
        print(f"[solution] save tail: tokenizer written {_STATE['written']}x in "
              f"{_STATE['write_s'] * 1e3:.0f} ms, replayed {_STATE['replayed']}x in "
              f"{_STATE['replay_s'] * 1e3:.0f} ms (files: {written} written, {kept} already in place); "
              f"adapter saved {calls}x in {seconds * 1e3:.0f} ms", flush=True)


def _time_adapter_saves():
    """PeftModel.save_pretrained timed for the exit line (the same call, nothing else changes)."""
    try:
        from peft import PeftModel
    except ImportError:
        return
    if getattr(PeftModel, "_soup_save_timed", False):
        return
    original = PeftModel.save_pretrained

    def save_pretrained(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            _STATE["adapter"][0] += 1
            _STATE["adapter"][1] += time.perf_counter() - started

    save_pretrained.__doc__ = original.__doc__
    PeftModel.save_pretrained = save_pretrained
    PeftModel._soup_save_timed = True


def patch_soup_save_tail():
    """apply(): with SOUP_SOLUTION_SAVE_ONCE=1, in the `soup train` process only."""
    from .exit_release import _soup_train_process

    if _soup_train_process():
        patch_tokenizer_save()
        _time_adapter_saves()
