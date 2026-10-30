"""Soup's trainer build (the rest of SFTTrainerWrapper.setup after the model): the tokenization of the
training rows and datasets' fingerprint of it, with the same results and less work. Each part has
its own switch, off by default until measured on the T4; every replacement lives only while Soup's
`setup` runs and takes the original code for anything it does not recognize.

SOUP_SOLUTION_FAST_ROWS=1: Soup formats every row with `build_assistant_only_labels`
(soup_cli/data/loss_mask.py:273). It first asks the chat template for an assistant mask
(`_apply_template_with_mask`: a render with `return_assistant_tokens_mask=True` and a tokenization),
and Qwen2.5's template has no `{% generation %}` block, so transformers returns an all-zero mask
(transformers/utils/chat_template_utils.py:508, tokenization_utils_base.py:3133-3154: no generation
indices, `current_mask = [0] * len(ids)`) and Soup rejects it whenever the row has an assistant
message (loss_mask.py:146-147), then renders and tokenizes the conversation again for its incremental
fallback. Here, for a real transformers tokenizer whose template has no `{% generation %}` block
(transformers' own regex) and a conversation with an assistant message, `_apply_template_with_mask`
returns None at once: the value Soup computes, without the first render (one of the four renders
and tokenizations of each row). Every ids list then goes through `_coerce_int_list` (loss_mask.py:48),
which checks and converts each token in a Python loop; a list whose items are all exactly `int` is
returned as a copy (what the loop returns for it: `index(x)` and `int(x)` of an int are x), anything
else takes the loop. The rows are the same (test_trainer_build.py; on the 256 training rows, input
ids and labels compared one by one); datasets' fingerprint of the map changes, because the functions
it hashes through the module's globals are these two (a fingerprint only names cache files of
datasets backed by files, and this one is built in memory; the organizers allow even a random one).

SOUP_SOLUTION_FAST_FINGERPRINT=1: datasets fingerprints each map by pickling the function with dill
and `recurse=True` (datasets/utils/_dill.py), and dill finds the globals a function uses with
`dill.detect.nestedglobals`, which prints `dis.dis` of the code to a captured stdout and parses the
lines that name a `*_GLOBAL*` instruction (dill/detect.py:138-160). Here the same names come from a
scan of the raw bytecode (the same instructions, their name operands, the nested code objects too):
the same names that can be globals (dill's text parsing also returns quoted string constants that
contain "_GLOBAL", which no module has as a global), so the same fingerprint (checked against dill
over the functions of soup_cli, trl, transformers and datasets in test_trainer_build.py), cheaper.

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

SOUP_SOLUTION_EARLY_PROBES=1: two lookups on the way to the first step that wait without working,
moved under the model load on a thread of this process, with the same result handed over: accelerate's
check for old RTX 4000 cards without P2P (`check_cuda_p2p_ib_support`, accelerate/utils/environment.py,
called by its first PartialState while Soup builds TrainingArguments), which runs `nvidia-smi
--query-gpu=count,name` and waits for it (~75 ms on the T4 with ~5% CPU in the main thread); and
`importlib.metadata.entry_points()`, which Soup's plugin discovery calls at the start of train()
(soup_cli/plugins/__init__.py:193): a read of every installed distribution's entry points (~70 ms on
Kaggle). Started with the tokenizer load inside Soup's setup (as EARLY_ROWS); the entry points are
served (a select on the same immutable EntryPoints object) only while sys.path is the one they were
read with, until Soup's train() returns. Nothing is installed during a run (the rules forbid it; the
harness compares `pip freeze`), and the GPUs do not change, so the answers are the ones the calls
would give later; the organizers allow the same for packages_distributions().
"""

import dis
import os
import re
import sys
import threading
import time
import types

_GENERATION = re.compile(r"\{\%-?\s*generation\s*-?\%\}")  # transformers' check (chat_template_utils.py:508)


def _flag(name):
    return os.environ.get(name, "0").strip().lower() not in ("0", "false", "no", "")


# ---------------------------------------------------------------- SOUP_SOLUTION_FAST_ROWS

def _template_without_generation(tokenizer):
    """The tokenizer's single chat template when it is a real transformers tokenizer with the
    standard apply_chat_template and its template has no `{% generation %}` block, else None."""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        return None
    if getattr(type(tokenizer), "apply_chat_template", None) is not PreTrainedTokenizerBase.apply_chat_template:
        return None
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template or _GENERATION.search(template):
        return None
    return template


def _fast_mask(original):
    def _apply_template_with_mask(tokenizer, messages, *args, **kwargs):
        try:
            if (not args and not kwargs and _template_without_generation(tokenizer) is not None
                    and isinstance(messages, (list, tuple)) and messages
                    and all(isinstance(m, dict) for m in messages)
                    and any(m.get("role") == "assistant" for m in messages)):
                return None  # the all-zero mask Soup rejects (see the module docstring)
        except Exception:  # noqa: BLE001
            pass
        return original(tokenizer, messages, *args, **kwargs)

    _apply_template_with_mask._soup_original = original
    return _apply_template_with_mask


def _fast_int_list(original):
    def _coerce_int_list(values, *args, **kwargs):
        if type(values) is list and values:
            kinds = set(map(type, values))
            if len(kinds) == 1 and int in kinds:
                return values.copy()
        return original(values, *args, **kwargs)

    _coerce_int_list._soup_original = original
    return _coerce_int_list


# ---------------------------------------------------------------- SOUP_SOLUTION_FAST_FINGERPRINT

_GLOBAL_OPS = frozenset(op for name, op in dis.opmap.items() if "_GLOBAL" in name)
_LOAD_GLOBAL = dis.opmap.get("LOAD_GLOBAL")
_EXTENDED_ARG = dis.EXTENDED_ARG
_LOAD_GLOBAL_SHIFT = sys.version_info >= (3, 11)  # LOAD_GLOBAL's oparg is namei << 1 | push_null


def global_names(code):
    """The names of the `*_GLOBAL*` instructions of `code` and of every code object nested in its
    constants: what dill.detect.nestedglobals reads from `dis.dis` (see the module docstring)."""
    names = set()
    stack = [code]
    while stack:
        co = stack.pop()
        raw = co.co_code
        table = co.co_names
        ext = 0
        for i in range(0, len(raw), 2):
            op = raw[i]
            arg = raw[i + 1] | ext
            if op == _EXTENDED_ARG:
                ext = arg << 8
                continue
            ext = 0
            if op in _GLOBAL_OPS:
                names.add(table[arg >> 1 if (op == _LOAD_GLOBAL and _LOAD_GLOBAL_SHIFT) else arg])
        for const in co.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
    return names


def _fast_nestedglobals(original):
    def nestedglobals(func, recurse=True):
        if not recurse:
            return original(func, recurse)
        try:
            from dill.detect import code as code_of

            code = code_of(func)
            if code is None:
                return list()
            return list(global_names(code))
        except Exception:  # noqa: BLE001 - dill's own way then
            return original(func, recurse)

    nestedglobals._soup_original = original
    return nestedglobals


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
            from . import _on

            if _on("SOUP_SOLUTION_DATASET_HASH"):
                from .startup import install_fast_tokenizer_reducer

                install_fast_tokenizer_reducer(self.tokenizer)  # removed when Soup's setup returns
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


def _install_early(wrapper, args, kwargs, rows=True, probes=False):
    """For one Soup setup: the hook that starts the threads once the tokenizer is loaded (the rows
    with `rows`, the probes with `probes`), and Soup's calls that take the rows' results. Returns
    the undo function (None when not installed)."""
    try:
        import transformers
        from soup_cli.data import sft_format
        from soup_cli.trainer import sft as soup_sft
    except Exception:  # noqa: BLE001
        return None
    dataset = args[0] if args else kwargs.get("dataset")
    rows = dataset.get("train") if isinstance(dataset, dict) else None
    auto = getattr(transformers, "AutoTokenizer", None)
    loader = getattr(auto, "__dict__", {}).get("from_pretrained")
    build, map_rows = getattr(sft_format, "build_format_row", None), getattr(soup_sft, "_map_text_sft_rows", None)
    masks = getattr(soup_sft, "_ensure_assistant_masks", None)
    if (not isinstance(loader, classmethod) or not callable(build) or not callable(map_rows)
            or not callable(masks) or any(getattr(f, "_soup_original", None) is not None
                                          for f in (build, map_rows, masks))):
        return None
    early = _Early(wrapper, rows, build, map_rows, masks)

    def from_pretrained(cls, *a, **k):
        tokenizer = loader.__func__(cls, *a, **k)
        if _EARLY["current"] is early:
            if probes:
                _start_probes(tokenizer)
            if rows:
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


# ---------------------------------------------------------------- SOUP_SOLUTION_EARLY_PROBES

_PROBES = {"current": None, "patched": False}


class _Probes:
    def __init__(self):
        self.entry_points = self.sys_path = self.p2p = None
        self.p2p_function = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._work, name="solution-early-probes", daemon=True)

    def _work(self):
        try:
            import importlib.metadata as metadata

            self.sys_path = tuple(sys.path)
            self.entry_points = getattr(metadata.entry_points, "_soup_original", metadata.entry_points)()
        except Exception:  # noqa: BLE001
            self.entry_points = None
        try:
            from accelerate.utils import environment

            self.p2p = bool(environment.check_cuda_p2p_ib_support())
        except Exception:  # noqa: BLE001
            self.p2p = None
        self.done.set()


def _probe_entry_points(original):
    def entry_points(**params):
        probes = _PROBES["current"]
        if probes is not None and probes.done.wait(5.0) and probes.entry_points is not None \
                and probes.sys_path == tuple(sys.path):
            try:
                return probes.entry_points.select(**params)
            except Exception:  # noqa: BLE001
                pass
        return original(**params)

    entry_points._soup_original = original
    return entry_points


def _probe_p2p(original):
    def check_cuda_p2p_ib_support():
        probes = _PROBES["current"]
        if probes is not None and probes.done.wait(5.0) and probes.p2p is not None:
            return probes.p2p
        return original()

    check_cuda_p2p_ib_support._soup_original = original
    return check_cuda_p2p_ib_support


def _patch_probes():
    """The two calls, served from the probes while a run has them (otherwise the originals)."""
    if _PROBES["patched"]:
        return
    try:
        import importlib.metadata as metadata

        import accelerate.state as accelerate_state
    except Exception:  # noqa: BLE001
        return
    if getattr(metadata.entry_points, "_soup_original", None) is None:
        metadata.entry_points = _probe_entry_points(metadata.entry_points)
    current = getattr(accelerate_state, "check_cuda_p2p_ib_support", None)
    if callable(current) and getattr(current, "_soup_original", None) is None:
        accelerate_state.check_cuda_p2p_ib_support = _probe_p2p(current)
    _PROBES["patched"] = True


def _start_probes(tokenizer):
    try:
        import torch
        from transformers import PreTrainedTokenizerBase

        if _PROBES["current"] is not None or not isinstance(tokenizer, PreTrainedTokenizerBase) \
                or not torch.cuda.is_available():
            return
    except Exception:  # noqa: BLE001
        return
    probes = _Probes()
    _PROBES["current"] = probes
    probes.thread.start()


# ---------------------------------------------------------------- scoping

def _replacements():
    """[(owner, attribute, factory)] for the switches that are on."""
    out = []
    if _flag("SOUP_SOLUTION_FAST_ROWS"):
        try:
            from soup_cli.data import loss_mask

            out.append((loss_mask, "_apply_template_with_mask", _fast_mask))
            out.append((loss_mask, "_coerce_int_list", _fast_int_list))
        except Exception:  # noqa: BLE001
            pass
    if _flag("SOUP_SOLUTION_FAST_FINGERPRINT"):
        try:
            import dill.detect

            out.append((dill.detect, "nestedglobals", _fast_nestedglobals))
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
    """SFTTrainerWrapper.setup runs with the replacements of the switches that are on, the early
    tokenization (SOUP_SOLUTION_EARLY_ROWS) and the early probes (SOUP_SOLUTION_EARLY_PROBES, which
    last until train() returns); none by default: then nothing is wrapped."""
    early = _flag("SOUP_SOLUTION_EARLY_ROWS")
    probes = _flag("SOUP_SOLUTION_EARLY_PROBES")
    if not _replacements() and not early and not probes:
        return
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_trainer_build", False):
        return
    original_setup = cls.setup
    original_train = cls.train
    if probes:
        _patch_probes()

    def setup(self, *args, **kwargs):
        undo = install()
        undo_early = _install_early(self, args, kwargs, rows=early, probes=probes) if (early or probes) else None
        try:
            return original_setup(self, *args, **kwargs)
        finally:
            if undo_early is not None:
                undo_early()
            undo()

    def train(self, *args, **kwargs):
        try:
            return original_train(self, *args, **kwargs)
        finally:
            _PROBES["current"] = None  # the probes serve this run's setup and the start of train only

    cls.setup = setup
    if probes:
        cls.train = train
    cls._soup_trainer_build = True
