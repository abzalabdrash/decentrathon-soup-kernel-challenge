"""Soup's trainer build (the rest of SFTTrainerWrapper.setup after the model): the tokenization of the
training rows, with the same rows and datasets' own fingerprint (the organizers' answers 9, 13 and 53:
how the fingerprint is computed and what it hashes stay Soup's). Each part has its own switch, off by
default until measured on the T4; every replacement lives only while Soup's `setup` runs and takes the
original code for anything it does not recognize.

SOUP_SOLUTION_EARLY_ROWS=1: the tokenization of the training rows needs only the tokenizer and the
rows, and the tokenizer is ready before the model load starts (soup_cli/trainer/sft.py:1661-1666,
then from_pretrained at :1719). So when Soup's setup has loaded the tokenizer, a thread of this
process builds Soup's own format_row (`build_format_row`, the same arguments Soup passes at
sft.py:846-851) and runs Soup's own `_map_text_sft_rows` on the same rows (sft.py:852-857) while
the main thread loads the model; when Soup then asks for them with the same tokenizer, config and row
list, it gets that format_row and that dataset (and the dataset with Soup's assistant_masks column,
`_ensure_assistant_masks`, sft.py:880). The work is the same and happens once; only its time moves
under the model load, where the main thread waits on the GPU and the disk part of the time.
Taken only for Soup's plain text SFT path (no RAFT, vision, audio, layer streaming, curriculum,
chat-template override, prompt strategy or vocabulary expansion, a tokenizer with a pad token, so
Soup changes nothing in it after the load), and the tokenizer's vocabulary size, template and special
tokens must be unchanged when the result is used; otherwise, or if the thread failed, Soup's call runs
as usual (and raises what it raises).

SOUP_SOLUTION_HELD_ENCODE=1: Soup's row tokenization calls the tokenizer once per
text, and transformers' `TokenizersBackend._encode_plus` (tokenization_utils_tokenizers.py:1027)
hands even a single text to the Rust tokenizer's `encode_batch`, which releases the GIL around its
work (~0.1 ms of Rust per row) and must take it back afterwards. While the warm-up threads compile
kernels, every such retake waits until a worker drops the GIL at its next switch interval (1 ms
while the warm-up runs): on the T4 (soup-s1-diag0, startup profile) the 256 rows' `encode_batch`
calls took 1.17 s of a 1.66 s tokenization window in which the main thread got 23 % of a CPU. On a
4-CPU WSL box with one busy Python thread, 512 calls: `encode_batch([text])` 0.71 s, `encode(text)`
0.15 s, 0.06 s alone. While Soup's setup runs, `_encode_plus` sees the tokenizer through a stand-in
whose `encode_batch` of one plain text (no pretokenized input, no padding configured on the Rust
tokenizer) returns `[encode(text)]`: tokenizers' `encode_batch` is `encode` mapped over the inputs
plus a batch padding that is off here, so the encodings are the same (ids, type ids, masks, offsets,
tokens, word ids, overflow: checked on the 256 training rows); anything else, and every other
attribute, goes to the tokenizer itself. The stand-in lives only for that one call.
"""

import os
import threading
import time



def _flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


# ---------------------------------------------------------------- SOUP_SOLUTION_EARLY_ROWS

EARLY_TIMEOUT = 300.0  # seconds Soup's setup waits for the thread before it tokenizes itself
_EARLY = {"current": None}


def _signature(tokenizer):
    """What must not change between the thread's tokenization and its use."""
    return (len(tokenizer), getattr(tokenizer, "chat_template", None), tokenizer.pad_token,
            tokenizer.eos_token, tokenizer.bos_token, tokenizer.unk_token,
            tuple(map(str, getattr(tokenizer, "all_special_tokens", ()))))


def _plain_text_sft(wrapper, tokenizer, rows):
    try:
        from transformers import PreTrainedTokenizerBase

        cfg = wrapper.config
        tcfg, data = cfg.training, cfg.data
        return (isinstance(tokenizer, PreTrainedTokenizerBase) and tokenizer.pad_token is not None
                and getattr(cfg, "task", None) == "sft" and getattr(cfg, "modality", "text") == "text"
                and getattr(cfg, "backend", None) == "transformers"
                and getattr(data, "format", None) != "raft"
                and not getattr(tcfg, "stream_layers", False) and not getattr(tcfg, "curriculum", False)
                and not getattr(data, "chat_template", None) and not getattr(data, "prompt_strategy", None)
                and not getattr(data, "add_new_tokens", None)
                and not getattr(data, "new_special_tokens", None)
                and isinstance(getattr(data, "max_length", None), int)
                and isinstance(rows, list) and rows and all(isinstance(row, dict) for row in rows))
    except Exception:  # noqa: BLE001
        return False


class _Early:
    """One early tokenization of a Soup setup (see the module docstring)."""

    def __init__(self, wrapper, rows, build_format_row, map_rows, masks):
        self.wrapper, self.rows = wrapper, rows
        self.build_format_row, self.map_rows, self.masks = build_format_row, map_rows, masks
        self.thread = self.tokenizer = self.signature = self.format_row = None
        self.result = self.masked = self.error = self.seconds = None
        self.used = False
        self.ready = threading.Event()  # set once format_row is built (or failed)

    def start(self, tokenizer):
        if self.thread is not None or not _plain_text_sft(self.wrapper, tokenizer, self.rows):
            return
        from soup_cli.trainer import sft as soup_sft

        cfg = self.wrapper.config
        self.tokenizer, self.signature = tokenizer, _signature(tokenizer)
        self.data_cfg, self.tcfg, self.max_length = cfg.data, cfg.training, cfg.data.max_length
        self.console = getattr(soup_sft, "console", None)
        self.thread = threading.Thread(target=self._work, name="solution-early-rows", daemon=True)
        self.thread.start()

    def _work(self):
        begin = time.perf_counter()
        try:
            self.format_row = self.build_format_row(tokenizer=self.tokenizer, data_cfg=self.data_cfg,
                                                    console=self.console, training_cfg=self.tcfg)
        except BaseException as exc:  # noqa: BLE001 - Soup's own call then
            self.error = exc
        finally:
            self.ready.set()
        if self.error is None:
            try:
                self.result = self.map_rows(self.rows, format_row=self.format_row, split="train",
                                            max_length=self.max_length)
                # Soup's next pass over the same dataset (sft.py:880): the assistant_masks column
                self.masked = self.masks(self.result)
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
        self.seconds = time.perf_counter() - begin

    def format_row_for(self, tokenizer, data_cfg, console, training_cfg):
        if (self.thread is None or tokenizer is not self.tokenizer or data_cfg is not self.data_cfg
                or training_cfg is not self.tcfg or console is not self.console):
            return None
        if not self.ready.wait(EARLY_TIMEOUT) or self.error is not None:
            return None
        return self.format_row if _signature(tokenizer) == self.signature else None

    def dataset_for(self, rows, format_row, split, max_length):
        if (self.thread is None or self.format_row is None or rows is not self.rows
                or format_row is not self.format_row or split != "train" or max_length != self.max_length):
            return None
        waited = time.perf_counter()
        self.thread.join(EARLY_TIMEOUT)
        waited = time.perf_counter() - waited
        if self.thread.is_alive() or self.error is not None or self.result is None:
            return None
        if _signature(self.tokenizer) != self.signature:
            return None
        print(f"[solution] training rows tokenized during the model load ({self.seconds:.2f} s in a "
              f"thread; the trainer build waited {waited:.2f} s for it)", flush=True)
        self.used = True
        return self.result

    def masked_for(self, dataset):
        if not self.used or dataset is not self.result or self.masked is None:
            return None
        return self.masked


def _early_build_format_row(original):
    def build_format_row(*args, **kwargs):
        early = _EARLY["current"]
        if early is not None:
            try:
                names = ("tokenizer", "data_cfg", "console", "training_cfg")
                values = dict(zip(names, args))
                if len(args) <= len(names) and set(kwargs) <= set(names) - set(values):
                    values.update(kwargs)
                    found = early.format_row_for(values.get("tokenizer"), values.get("data_cfg"),
                                                 values.get("console"), values.get("training_cfg"))
                    if found is not None:
                        return found
            except Exception:  # noqa: BLE001
                pass
        return original(*args, **kwargs)

    build_format_row._soup_original = original
    return build_format_row


def _early_map_rows(original):
    def _map_text_sft_rows(rows, *args, **kwargs):
        early = _EARLY["current"]
        if early is not None and not args and set(kwargs) == {"format_row", "split", "max_length"}:
            try:
                found = early.dataset_for(rows, kwargs["format_row"], kwargs["split"], kwargs["max_length"])
                if found is not None:
                    return found
            except Exception:  # noqa: BLE001
                pass
        return original(rows, *args, **kwargs)

    _map_text_sft_rows._soup_original = original
    return _map_text_sft_rows


def _early_masks(original):
    def _ensure_assistant_masks(dataset, *args, **kwargs):
        early = _EARLY["current"]
        if early is not None and not args and not kwargs:
            found = early.masked_for(dataset)
            if found is not None:
                return found
        return original(dataset, *args, **kwargs)

    _ensure_assistant_masks._soup_original = original
    return _ensure_assistant_masks


def _install_early(wrapper, args, kwargs):
    """For one Soup setup: the hook that starts the rows' thread once the tokenizer is loaded, and
    Soup's calls that take the rows' results. Returns the undo function (None when not installed)."""
    try:
        import transformers
        from soup_cli.data import sft_format
        from soup_cli.trainer import sft as soup_sft
    except Exception:  # noqa: BLE001
        return None
    dataset = args[0] if args else kwargs.get("dataset")
    train_rows = dataset.get("train") if isinstance(dataset, dict) else None
    auto = getattr(transformers, "AutoTokenizer", None)
    loader = getattr(auto, "__dict__", {}).get("from_pretrained")
    build, map_rows = getattr(sft_format, "build_format_row", None), getattr(soup_sft, "_map_text_sft_rows", None)
    masks = getattr(soup_sft, "_ensure_assistant_masks", None)
    if (not isinstance(loader, classmethod) or not callable(build) or not callable(map_rows)
            or not callable(masks) or any(getattr(f, "_soup_original", None) is not None
                                          for f in (build, map_rows, masks))):
        return None
    early = _Early(wrapper, train_rows, build, map_rows, masks)

    def from_pretrained(cls, *a, **k):
        tokenizer = loader.__func__(cls, *a, **k)
        if _EARLY["current"] is early:
            try:
                early.start(tokenizer)
            except Exception:  # noqa: BLE001 - Soup tokenizes itself then
                pass
        return tokenizer

    from_pretrained._soup_original = loader
    hook = classmethod(from_pretrained)
    new_build, new_map, new_masks = _early_build_format_row(build), _early_map_rows(map_rows), _early_masks(masks)
    auto.from_pretrained = hook
    sft_format.build_format_row = new_build
    soup_sft._map_text_sft_rows = new_map
    soup_sft._ensure_assistant_masks = new_masks
    _EARLY["current"] = early

    def undo():
        if _EARLY["current"] is early:
            _EARLY["current"] = None
        if auto.__dict__.get("from_pretrained") is hook:
            auto.from_pretrained = loader
        if getattr(sft_format, "build_format_row", None) is new_build:
            sft_format.build_format_row = build
        if getattr(soup_sft, "_map_text_sft_rows", None) is new_map:
            soup_sft._map_text_sft_rows = map_rows
        if getattr(soup_sft, "_ensure_assistant_masks", None) is new_masks:
            soup_sft._ensure_assistant_masks = masks

    return undo


# ---------------------------------------------------------------- SOUP_SOLUTION_HELD_ENCODE

class _SingleEncode:
    """Stands for a tokenizers.Tokenizer inside one `_encode_plus` call: a batch of one text is
    encoded with `encode`, which keeps the GIL, instead of `encode_batch`, which releases it (see
    the module docstring); everything else goes to the tokenizer itself."""

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)

    def encode_batch(self, input, is_pretokenized=False, add_special_tokens=True):  # noqa: A002 - tokenizers' name
        inner = self._inner
        if (type(input) is list and len(input) == 1 and type(input[0]) is str and not is_pretokenized
                and inner.padding is None):
            return [inner.encode(input[0], is_pretokenized=False, add_special_tokens=add_special_tokens)]
        return inner.encode_batch(input, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens)


def _held_encode_plus(original):
    try:
        from tokenizers import Tokenizer
    except Exception:  # noqa: BLE001
        Tokenizer = None

    def _encode_plus(self, *args, **kwargs):
        state = getattr(self, "__dict__", None)
        inner = state.get("_tokenizer") if isinstance(state, dict) else None
        if Tokenizer is None or type(inner) is not Tokenizer:
            return original(self, *args, **kwargs)
        state["_tokenizer"] = _SingleEncode(inner)
        try:
            return original(self, *args, **kwargs)
        finally:
            if type(state.get("_tokenizer")) is _SingleEncode:
                state["_tokenizer"] = inner

    _encode_plus._soup_original = original
    return _encode_plus


# ---------------------------------------------------------------- scoping

def _replacements():
    """[(owner, attribute, factory)] for the switches that are on."""
    out = []
    if _flag("SOUP_SOLUTION_HELD_ENCODE"):
        try:
            from transformers.tokenization_utils_tokenizers import TokenizersBackend

            if callable(TokenizersBackend.__dict__.get("_encode_plus")):
                out.append((TokenizersBackend, "_encode_plus", _held_encode_plus))
        except Exception:  # noqa: BLE001
            pass
    return out


def install():
    """Puts the replacements in place; returns the function that takes them out again (each one
    only if it is still ours)."""
    done = []
    for owner, attribute, factory in _replacements():
        original = getattr(owner, attribute, None)
        if original is None or getattr(original, "_soup_original", None) is not None:
            continue
        replacement = factory(original)
        setattr(owner, attribute, replacement)
        done.append((owner, attribute, original, replacement))

    def undo():
        while done:
            owner, attribute, original, replacement = done.pop()
            if getattr(owner, attribute, None) is replacement:
                setattr(owner, attribute, original)

    return undo


def patch_soup_trainer_build():
    """SFTTrainerWrapper.setup runs with the replacements of the switches that are on and the early
    tokenization (SOUP_SOLUTION_EARLY_ROWS); with none of them nothing is wrapped."""
    early = _flag("SOUP_SOLUTION_EARLY_ROWS")
    if not _replacements() and not early:
        return
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_trainer_build", False):
        return
    original_setup = cls.setup

    def setup(self, *args, **kwargs):
        undo = install()
        undo_early = _install_early(self, args, kwargs) if early else None
        try:
            return original_setup(self, *args, **kwargs)
        finally:
            if undo_early is not None:
                undo_early()
            undo()

    cls.setup = setup
    cls._soup_trainer_build = True
