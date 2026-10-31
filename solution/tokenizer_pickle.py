"""datasets' fingerprint of Soup's tokenization map, the same bytes written faster
(SOUP_SOLUTION_TOKENIZER_PICKLE=1, off by default until the organizers answer).

Soup tokenizes the training rows with `Dataset.from_list(rows).map(checked_format_row, ...)`
(soup_cli/trainer/sft.py:168), and datasets fingerprints the map by hashing the function with its
closure: `Hasher.hash(fn)` = xxhash of `datasets.utils._dill.dumps(fn)` (datasets/fingerprint.py:
212-214), a pickle written by dill's pure-Python pickler. The closure holds the tokenizer, whose state
holds Python copies of the vocabulary (`_vocab`, a dict of 151,643 str -> int) and the merges
(`_merges`, a list of 151,387 (str, str) tuples): about 0.9 of the tokenizer's 1.2 s on the WSL box,
one save() per string, int and tuple (~760 k calls). It happens once per run (the second map, Soup's
assistant masks, and TRL's truncation hash in milliseconds), on the main thread, as pure Python that
keeps the GIL for whole switch intervals, so it also stalls the warm-up threads while it runs.

Here, while Soup's `setup` runs (trainer_build.py), datasets' reducer for the tokenizer's type
(`_save_transformersPreTrainedTokenizerBase`, datasets/utils/_dill.py:246) is called on a stand-in
of the tokenizer whose state has those two containers wrapped in markers; the stand-in is memoized
as the tokenizer itself, and each marker writes exactly the bytes the pickler would write for its
container, with the pickler's own memo and framing:

- the container's memo check (a BINGET if it was pickled earlier in the stream), EMPTY_DICT or
  EMPTY_LIST, its MEMOIZE (`pickler.memoize`, the same call);
- the items in batches of `_BATCHSIZE` between MARK and SETITEMS / APPENDS (SETITEM / APPEND for a
  batch of one), the dict's items sorted as datasets' `_batch_setitems` sorts them (`sorted(items)`);
- each str as `save_str` writes it (SHORT_BINUNICODE or BINUNICODE; datasets' pickler never memoizes
  a str), each int as `save_long` (BININT1, BININT2 or BININT), each tuple as `save_tuple` (its two
  strings, TUPLE2, MEMOIZE), every tuple entered in the memo with the index `memoize` gives it;
- the frame committed (`framer.commit_frame()`) at every point where the pickler's `save()` would
  commit it: before each str, int and tuple, when the current frame holds 64 KiB or more.

The pickle, and so the fingerprint, is the same byte for byte (tests on the real Qwen2.5 tokenizer
and on Soup's map, and on a synthetic state that crosses frames, batches and every str and int form;
at run time only a cheap gate: the Python and dill versions those tests cover, datasets' pickler).
Anything the markers do not reproduce (another type inside the containers, a duplicate or already
memoized tuple, a string of 64 KiB or more, an int outside 32 bits, a dict that
does not sort, a pickler without framing, protocol < 4, a memo-less "fast" pickler, dill or pickle
dispatch other than the stock ones) takes datasets' own reducer for the whole tokenizer. Nothing else
in the stream changes: the rest of the state, the class, the backend tokenizer with datasets' own
truncation/padding toggling, the closure and the globals go through dill as before.
"""

import array
import itertools
import os
import pickle
import threading
import time

_BULKY = {"_vocab": "dict", "_merges": "pairs"}
LAST = {"seconds": 0.0, "fast": 0, "fallback": None, "check": None}  # for the log and the tests
_INSTALL_LOCK = threading.Lock()


def enabled():
    return os.environ.get("SOUP_SOLUTION_TOKENIZER_PICKLE", "0").strip().lower() not in ("0", "false", "no", "")


class _Marker:
    """One bulky container in the stand-in's state, written by `_save_marker`."""

    __slots__ = ("value", "kind", "plan")

    def __init__(self, value, kind, plan):
        self.value, self.kind, self.plan = value, kind, plan


# ---------------------------------------------------------------- the bytes of one container

_SHORT, _UNICODE = pickle.SHORT_BINUNICODE[0], pickle.BINUNICODE[0]
_INT1, _INT2, _INT4 = pickle.BININT1[0], pickle.BININT2[0], pickle.BININT[0]
_MARK, _TUPLE2, _MEMOIZE = pickle.MARK[0], pickle.TUPLE2[0], pickle.MEMOIZE[0]


_HEADERS = [bytes((_SHORT, n)) for n in range(256)]  # SHORT_BINUNICODE + length, by length
_INT_TABLE = []  # save_long's bytes by value, built up to the largest value asked for (see _int_bytes_of)
_MAX_TABLE = 1 << 20


def _int_bytes_of(values):
    """save_long's bytes of each value (all ints in [0, _MAX_TABLE)), from a table built once."""
    top = max(values) + 1
    if len(_INT_TABLE) < top:
        _INT_TABLE.extend(bytes((_INT1, v)) if v <= 0xFF else
                          (bytes((_INT2,)) + v.to_bytes(2, "little") if v <= 0xFFFF else
                           bytes((_INT4,)) + v.to_bytes(4, "little", signed=True))
                          for v in range(len(_INT_TABLE), top))
    return list(map(_INT_TABLE.__getitem__, values))


def _strings(texts, limit):
    """(save_str headers, encoded bytes, header + bytes lengths) of strings: SHORT_BINUNICODE up to
    255 bytes, BINUNICODE below `limit`; None when one would take save_str's large-bytes path."""
    encoded = [text.encode("utf-8", "surrogatepass") for text in texts]
    lengths = list(map(len, encoded))
    longest = max(lengths, default=0)
    if longest >= limit:
        return None
    if longest <= 0xFF:
        return list(map(_HEADERS.__getitem__, lengths)), encoded, [n + 2 for n in lengths]
    headers = [_HEADERS[n] if n <= 0xFF else bytes((_UNICODE,)) + n.to_bytes(4, "little") for n in lengths]
    return headers, encoded, [n + 2 if n <= 0xFF else n + 5 for n in lengths]


def _assemble(pieces, per_item, lengths, count, batch, many, one):
    """The batched stream of `count` items: each item is `per_item` consecutive byte strings of
    `pieces` and two frame checks, whose unit lengths are consecutive in `lengths` (the first unit
    starts at the item, the second where its second check is). Returns (data, check offsets)."""
    chunks, checks, pos = [], array.array("q"), 0
    for first in range(0, count, batch):
        last = min(first + batch, count)
        several = last - first > 1
        if several:
            chunks.append(pickle.MARK)
            pos += 1
        offsets = list(itertools.accumulate(lengths[2 * first:2 * last], initial=pos))
        checks.extend(offsets[:-1])
        chunks.append(b"".join(pieces[per_item * first:per_item * last]))
        pos = offsets[-1]
        chunks.append(many if several else one)
        pos += 1
    return b"".join(chunks), checks


def _plan_dict(d, limit, batch):
    """A str -> int dict's bytes after its MEMOIZE and the offsets in them where save() checks the
    frame (before every key and every value), or None. Built with comprehensions, maps and tables
    (no Python step per item on the usual path)."""
    if "__name__" in d and type(d["__name__"]) is str:
        return None  # dill's save_module_dict could take it for a module's namespace
    try:
        keys = sorted(d)  # the order of datasets' sorted(items): the keys are distinct
    except Exception:  # noqa: BLE001 - datasets' fallback order then
        return None
    if not keys:
        return {"opener": pickle.EMPTY_DICT, "data": b"", "checks": (), "ids": None}
    values = list(map(d.__getitem__, keys))
    if set(map(type, keys)) != {str} or set(map(type, values)) != {int}:
        return None
    if min(values) < 0 or max(values) >= _MAX_TABLE:
        return _plan_dict_items(keys, values, limit, batch)
    strings = _strings(keys, limit)
    if strings is None:
        return None
    headers, encoded, lengths = strings
    as_bytes = _int_bytes_of(values)
    count = len(keys)
    pieces = [None] * (3 * count)
    pieces[0::3], pieces[1::3], pieces[2::3] = headers, encoded, as_bytes
    units = [0] * (2 * count)
    units[0::2] = lengths
    units[1::2] = list(map(len, as_bytes))
    data, checks = _assemble(pieces, 3, units, count, batch, pickle.SETITEMS, pickle.SETITEM)
    return {"opener": pickle.EMPTY_DICT, "data": data, "checks": checks, "ids": None}


def _plan_dict_items(keys, values, limit, batch):
    """_plan_dict one item at a time (ints outside the table: negative or large), or None."""
    data, checks = bytearray(), array.array("q")
    put, mark = data.append, checks.append
    count = len(keys)
    for first in range(0, count, batch):
        last = min(first + batch, count)
        several = last - first > 1
        if several:
            put(_MARK)
        for key, value in zip(keys[first:last], values[first:last]):
            if type(key) is not str or type(value) is not int:
                return None
            mark(len(data))
            encoded = key.encode("utf-8", "surrogatepass")
            n = len(encoded)
            if n <= 0xFF:
                put(_SHORT)
                put(n)
            elif n < limit:
                put(_UNICODE)
                data += n.to_bytes(4, "little")
            else:
                return None  # save_str's _write_large_bytes: not reproduced
            data += encoded
            mark(len(data))
            if 0 <= value <= 0xFF:
                put(_INT1)
                put(value)
            elif 0 <= value <= 0xFFFF:
                put(_INT2)
                data += value.to_bytes(2, "little")
            elif -0x80000000 <= value <= 0x7FFFFFFF:
                put(_INT4)
                data += value.to_bytes(4, "little", signed=True)
            else:
                return None  # LONG1 / LONG4: not reproduced
        data += pickle.SETITEMS if several else pickle.SETITEM
    return {"opener": pickle.EMPTY_DICT, "data": data, "checks": checks, "ids": None}


_TAIL = bytes((_TUPLE2, _MEMOIZE))  # save_tuple after its two items: TUPLE2, then memoize's MEMOIZE


def _plan_pairs(lst, limit, batch):
    """A list of (str, str) tuples: its bytes after its MEMOIZE and the frame check offsets (before
    each tuple's two strings), with the tuples' ids in order (their memo keys), or None."""
    ids = list(map(id, lst))  # once: each id() is an audit event, and the run has Python audit hooks
    if len(set(ids)) != len(ids):
        return None  # a tuple twice: its second save would be a BINGET
    count = len(lst)
    if not count:
        return {"opener": pickle.EMPTY_LIST, "data": b"", "checks": (), "ids": ids}
    if set(map(type, lst)) != {tuple} or set(map(len, lst)) != {2}:
        return None
    firsts, seconds = zip(*lst)
    if set(map(type, firsts)) != {str} or set(map(type, seconds)) != {str}:
        return None
    a, b = _strings(firsts, limit), _strings(seconds, limit)
    if a is None or b is None:
        return None
    pieces = [None] * (5 * count)
    pieces[0::5], pieces[1::5], pieces[2::5], pieces[3::5] = a[0], a[1], b[0], b[1]
    pieces[4::5] = [_TAIL] * count
    units = [0] * (2 * count)
    units[0::2] = a[2]
    units[1::2] = [n + 2 for n in b[2]]  # the second string, then TUPLE2 and MEMOIZE
    data, checks = _assemble(pieces, 5, units, count, batch, pickle.APPENDS, pickle.APPEND)
    return {"opener": pickle.EMPTY_LIST, "data": data, "checks": checks, "ids": ids}


def _write(pickler, data, checks):
    """`data` into the pickler's frames, committed at each check offset where save() would commit
    (the current frame holds _FRAME_SIZE_TARGET bytes or more)."""
    framer = pickler.framer
    target = framer._FRAME_SIZE_TARGET
    frame = framer.current_frame
    view = memoryview(data)
    flushed = 0
    shift = frame.tell()  # the frame's size at offset `pos` of data is pos + shift
    for pos in checks:
        if pos + shift >= target:
            frame.write(view[flushed:pos])
            framer.commit_frame()
            frame = framer.current_frame
            flushed = pos
            shift = frame.tell() - pos
    frame.write(view[flushed:])


def _save_marker(pickler, marker):
    """The dispatch entry of _Marker: the container's bytes as save() would write them (see the
    module docstring); save() already made the frame check for it."""
    value, plan = marker.value, marker.plan
    memo = pickler.memo
    found = memo.get(id(value))
    if found is not None:
        pickler.write(pickler.get(found[0]))
        return
    pickler.write(plan["opener"])
    pickler.memoize(value)
    ids = plan["ids"]
    if ids:  # each tuple memoized in order: index len(memo) at its save (strings are never memoized)
        start = len(memo)
        memo.update(zip(ids, zip(range(start, start + len(ids)), value)))
    _write(pickler, plan["data"], plan["checks"])


# ---------------------------------------------------------------- the tokenizer's reducer

def _usable(pickler):
    """Whether this pickler writes what the markers reproduce."""
    from datasets.utils import _dill

    dispatch = type(pickler).dispatch
    stock = pickle._Pickler.dispatch
    framer = getattr(pickler, "framer", None)
    return (isinstance(pickler, _dill.Pickler) and getattr(pickler, "proto", 0) >= 4 and getattr(pickler, "bin", False)
            and not getattr(pickler, "fast", False) and framer is not None
            and getattr(framer, "current_frame", None) is not None
            and getattr(pickler, "reducer_override", None) is None
            and pickler.put(0) == pickle.MEMOIZE
            and all(dispatch.get(t) is stock.get(t) for t in (list, tuple, str, int))
            and getattr(dispatch.get(dict), "__name__", "") == "save_module_dict"
            and type(pickler)._batch_setitems.__module__ == "datasets.utils._dill")


def _fast_reducer(original, checked=lambda: LAST["check"], counts=LAST):
    """datasets' tokenizer reducer `original`, given a stand-in whose bulky containers are markers."""

    def save(pickler, obj):
        started = time.perf_counter()
        try:
            state = obj.__dict__
            if not checked() or not _usable(pickler):
                raise LookupError("pickler")
            limit = pickler.framer._FRAME_SIZE_TARGET
            batch = pickler._BATCHSIZE
            swapped = {}
            for key, kind in _BULKY.items():
                value = state.get(key)
                if kind == "dict" and type(value) is dict:
                    plan = _plan_dict(value, limit, batch)
                elif kind == "pairs" and type(value) is list:
                    plan = _plan_pairs(value, limit, batch)
                    if plan is not None and not pickler.memo.keys().isdisjoint(plan["ids"]):
                        plan = None  # a tuple pickled earlier in the stream: a BINGET there
                else:
                    plan = None
                if plan is not None:
                    swapped[key] = _Marker(value, kind, plan)
            if not swapped:
                raise LookupError("containers")
            view = object.__new__(type(obj))  # no __init__: a carrier of the state
            view.__dict__.update(state)
            view.__dict__.update(swapped)
        except Exception as exc:  # noqa: BLE001 - datasets' own reducer for the whole tokenizer
            counts["fallback"] = repr(exc)
            return original(pickler, obj)
        memoize = type(pickler).memoize

        def memoize_as_tokenizer(item):
            memoize(pickler, obj if item is view else item)

        pickler.memoize = memoize_as_tokenizer  # save_reduce memoizes the stand-in as the tokenizer
        try:
            return original(pickler, view)
        finally:
            del pickler.memoize
            counts["fast"] += 1
            counts["seconds"] += time.perf_counter() - started

    save._soup_tokenizer_pickle = True
    save._soup_original = original
    return save


# ---------------------------------------------------------------- the checks

_PYTHONS = ((3, 11), (3, 12), (3, 13))  # pickle.py's _Pickler as the markers reproduce it (tested: 3.12, 3.13)
_DILLS = ("0.3.8", "0.3.9", "0.4.0", "0.4.1")  # the dill releases datasets 5 supports (datasets/utils/_dill.py)


def _compatible():
    """The run-time gate (cheap): a Python whose pure-Python pickler the markers reproduce, a dill
    release datasets supports, datasets' own pickler and tokenizer reducer in place. The byte-for-byte
    equivalence with datasets' own pickle is checked by the tests (`_self_check`, and the real Qwen2.5
    tokenizer and Soup's map in test_tokenizer_pickle.py), not at run time. Per pickler, `_usable` checks
    its protocol, framing, memo and dispatch when the tokenizer is pickled."""
    try:
        import sys

        import dill
        from datasets.utils import _dill

        return (sys.version_info[:2] in _PYTHONS and dill.__version__ in _DILLS
                and callable(getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None))
                and issubclass(_dill.Pickler, pickle._Pickler))
    except Exception:  # noqa: BLE001 - datasets' own reducer then
        return False


def _self_check():
    """Pickles a synthetic tokenizer-like state with datasets' pickler, plainly and through the
    markers (frames crossed, batches of one and of BATCHSIZE, every str and int form, memo
    references before and after); True when the bytes are equal."""
    from datasets.utils import _dill

    class Carrier:
        pass

    # the table path: non-negative values of every int form, keys of both str forms
    vocab = {f"t{i:05d}" + "é" * (i % 7) + ("x" * 300 if i % 977 == 0 else ""): (i * 37) % 200003
             for i in range(6001)}
    vocab.update({"v255": 255, "v256": 256, "v65535": 65535, "v65536": 65536})
    # the item path: a negative value and the largest 32-bit one
    odd = {f"o{i}": (i * 7919) % 90001 - (5 if i % 501 == 0 else 0) for i in range(1501)}
    odd["big"] = 2 ** 31 - 1
    merges = [(f"a{i}", f"b{i % 13}" * (1 + i % 3)) for i in range(2001)]
    longer = [(f"c{i}" + ("y" * 280 if i % 97 == 0 else ""), f"d{i}") for i in range(1001)]  # batch of one at the end
    shared = ["shared", 1]

    def build():
        first, second = Carrier(), Carrier()
        first.__dict__.update({"head": shared, "_vocab": vocab, "_merges": merges, "tail": shared,
                               "again": [merges[5], vocab]})
        second.__dict__.update({"_vocab": odd, "_merges": longer, "zz": [longer[3], first]})
        return [first, second, merges[7], second]

    def reducer(pickler, obj):  # the shape of datasets' reducer: a reduce with the state
        pickler.save_reduce(type(obj), (), state=obj.__dict__.copy(), obj=obj)

    dispatch = _dill.Pickler.dispatch
    counts = {"fast": 0, "seconds": 0.0, "fallback": None}
    marker = dispatch.get(_Marker)
    try:
        dispatch[Carrier] = reducer
        plain = _dill.dumps(build())
        dispatch[Carrier] = _fast_reducer(reducer, checked=lambda: True, counts=counts)
        dispatch[_Marker] = _save_marker
        fast = _dill.dumps(build())
    finally:
        dispatch.pop(Carrier, None)
        if marker is None:
            dispatch.pop(_Marker, None)
    return plain == fast and counts["fast"] == 2 and counts["fallback"] is None


# ---------------------------------------------------------------- scoping (trainer_build.py)

def install():
    """While the returned function is not called: datasets' reducer for tokenizer types is the fast
    one (new registrations through the module global, existing dispatch entries replaced), the
    marker type has its dispatch entry. Returns the undo function, or None when nothing was done."""
    try:
        from datasets.utils import _dill
    except Exception:  # noqa: BLE001
        return None
    with _INSTALL_LOCK:
        original = getattr(_dill, "_save_transformersPreTrainedTokenizerBase", None)
        if original is None or getattr(original, "_soup_tokenizer_pickle", False):
            return None
        dispatch = _dill.Pickler.dispatch
        fast = _fast_reducer(original)
        replaced = [t for t, f in list(dispatch.items()) if f is original]
        dispatch[_Marker] = _save_marker
        if LAST["check"] is None:
            LAST["check"] = _compatible()
        if not LAST["check"]:
            dispatch.pop(_Marker, None)
            return None
        _dill._save_transformersPreTrainedTokenizerBase = fast
        for t in replaced:
            dispatch[t] = fast

    def undo():
        with _INSTALL_LOCK:
            if _dill._save_transformersPreTrainedTokenizerBase is fast:
                _dill._save_transformersPreTrainedTokenizerBase = original
            for t, f in list(dispatch.items()):
                if f is fast:
                    dispatch[t] = original  # what datasets registers for it
            dispatch.pop(_Marker, None)
            if LAST["fast"] or LAST["fallback"]:
                print(f"[solution] tokenizer pickle: {LAST['fast']} tokenizer(s) hashed through the same-bytes "
                      f"writer in {LAST['seconds']:.2f} s"
                      + (f"; datasets' own path taken: {LAST['fallback']}" if LAST["fallback"] else ""), flush=True)

    return undo

