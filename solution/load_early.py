"""The checkpoint's tensor bytes read into pinned host memory by a thread from the start of apply(),
while the main thread imports (SOUP_SOLUTION_LOAD_EARLY=1, off by default until measured; the
organizers allow reading the weight files in a thread during the imports when Soup then builds the
model as usual, answer 48).

On the T4 the model load's copies are staged by load_pinned.py's thread from the safetensors mmap
(page faults and a memmove per tensor into a ring of pinned buffers) while the main thread runs
transformers' loop: in soup-l1-p0 the staging took 1.1 s of a thread and pinning the load's buffers
1.0 s of another, both inside the model setup, where the warm-up's two workers also run, and the loop
still waited 0.3 s for staging. The imports before it (3 s of the main thread, GIL-bound) leave three
of the four vCPUs idle.

Here, first thing in apply():
- the checkpoint files are guessed as transformers will find them for Soup: the `base` of the config
  Soup runs with (the `--config` argument of `soup train` in sys.argv, read only as a hint of which
  files to read: an input-derived guess, not a mode switch; the same input gives the same behaviour in
  tests and in a measured run), a local directory, or its Hugging Face cache snapshot (`refs/main`),
  `model.safetensors` or the files of `model.safetensors.index.json`; without `--config`, a snapshot
  or a header that parses, nothing is read;
- a thread reads each file's data section (after the 8-byte length and the JSON header) with
  `preadv` into one host tensor per file, in chunks that end on tensor boundaries (~128 MiB), and
  registers each chunk with the driver (`cudaHostRegister`) once it is read; a chunk's event is set
  when it is pinned;
- the guess is checked against what Soup actually loads: only a file that transformers opens itself
  (load_stream.py records its `safe_open` calls) with the same real path, size and modification time
  as the file read ahead is served from memory; when the handled load opens none of them, or when
  Soup's model setup ends, the buffers are dropped and the load goes the usual way;
- load_pinned.py, for a tensor whose file range (LOAD_PREAD's mapping of the loader's slices to the
  header) lies in a chunk, takes a view of that chunk as the staged bytes once the chunk's event is
  set, and copies it to the GPU from there (the same bytes the slice holds; the cast on the GPU as
  before); load_stream.py's embedding cast reads its bytes from there too;
- when the load returns, after load_pinned.py's final synchronize, the chunks are unregistered and
  the memory freed by a thread.
Anything not covered (another file, a range across chunks, a failed read or registration) takes the
path it took before.
"""

import json
import os
import re
import struct
import sys
import threading
import time

import torch

CHUNK_BYTES = 128 << 20
_STATE = {"thread": None, "files": {}, "error": None, "seconds": None, "bytes": 0, "used": 0, "released": False}
_LOCK = threading.Lock()


def enabled():
    return os.environ.get("SOUP_SOLUTION_LOAD_EARLY", "0").strip().lower() not in ("0", "false", "no", "")


class _Chunk:
    __slots__ = ("start", "end", "ready", "registered", "order")

    def __init__(self, start, end):
        self.start, self.end = start, end  # byte range in the file's data section
        self.ready = threading.Event()
        self.registered = False
        self.order = ()  # the natural key of its first tensor in the loader's order (see _plan)


class _File:
    """One checkpoint file's data section in host memory."""

    def __init__(self, path, data_start, size, chunks, stamp):
        self.path, self.data_start, self.size, self.chunks = path, data_start, size, chunks
        self.stamp = stamp  # (size, mtime_ns) of the file when it was planned
        self.tensors = {}  # key -> (dtype name, shape, first, end): the header's entries (load_spec.py)
        self.buffer = None
        self.failed = False

    def chunk_of(self, offset, nbytes):
        """The chunk holding the file range [offset, offset + nbytes), or None."""
        begin = offset - self.data_start
        for chunk in self.chunks:
            if chunk.start <= begin and begin + nbytes <= chunk.end:
                return chunk
        return None


# ---------------------------------------------------------------- which files

def _config_base(argv=None):
    argv = sys.argv if argv is None else argv
    if "--config" not in argv or argv.index("--config") + 1 >= len(argv):
        return None
    with open(argv[argv.index("--config") + 1], encoding="utf-8") as f:
        match = re.search(r"^base:\s*['\"]?([^'\"\s#]+)", f.read(), re.M)
    return match.group(1) if match else None


def _snapshot(base):
    if os.path.isdir(base):
        return base
    if base.count("/") != 1:
        return None
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface"), "hub")
    folder = os.path.join(hub, "models--" + base.replace("/", "--"))
    try:
        with open(os.path.join(folder, "refs", "main"), encoding="utf-8") as f:
            revision = f.read().strip()
    except OSError:
        return None
    path = os.path.join(folder, "snapshots", revision)
    return path if os.path.isdir(path) else None


def checkpoint_files(base):
    """The safetensors files transformers loads for `base` (a local directory or a cached hub repo)."""
    folder = _snapshot(base)
    if folder is None:
        return []
    index = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index, encoding="utf-8") as f:
            names = sorted(set(json.load(f)["weight_map"].values()))
        return [os.path.join(folder, name) for name in names]
    single = os.path.join(folder, "model.safetensors")
    return [single] if os.path.exists(single) else []


def _plan(path):
    """_File for `path`: the data section's start and size, chunks ending on tensor boundaries."""
    with open(path, "rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
        stat = os.fstat(f.fileno())
        size = stat.st_size - 8 - length
    tensors = sorted((info["data_offsets"][0], info["data_offsets"][1], key) for key, info in header.items()
                     if key != "__metadata__")
    chunks, start = [], 0
    for _, end, _ in tensors:
        if end - start >= CHUNK_BYTES:
            chunks.append(_Chunk(start, end))
            start = end
    if start < size:
        chunks.append(_Chunk(start, size))
    # transformers' loop takes the tensors in natural key order (layers 0, 1, 2, ...: core_model_loading
    # dot_natural_key), the file holds them alphabetically (0, 1, 10, 11, ...): the thread reads the
    # chunk with the loop's earliest tensor first
    for chunk in chunks:
        keys = [_natural(key) for first, end, key in tensors if chunk.start <= first and end <= chunk.end]
        chunk.order = min(keys) if keys else ()
    entry = _File(os.path.realpath(path), 8 + length, size, chunks, (stat.st_size, stat.st_mtime_ns))
    entry.tensors = {key: (info["dtype"], tuple(info["shape"]), *info["data_offsets"]) for key, info in header.items()
                     if key != "__metadata__"}
    return entry


def _natural(key):
    """Sort key of a state-dict name with its numbers compared as numbers."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in key.split("."))


# ---------------------------------------------------------------- the reading thread

def _read(files, device):
    from .load_pinned import _read_into

    started = time.perf_counter()
    try:
        torch.cuda.set_device(device)
        cudart = torch.cuda.cudart()
        for entry in files:
            entry.buffer = torch.empty(entry.size, dtype=torch.uint8)
            fd = os.open(entry.path, os.O_RDONLY)
            try:
                for chunk in sorted(entry.chunks, key=lambda c: c.order):
                    view = entry.buffer[chunk.start:chunk.end]
                    _read_into(fd, entry.data_start + chunk.start, view, chunk.end - chunk.start)
                    ok = cudart.cudaHostRegister(view.data_ptr(), chunk.end - chunk.start, 0) == cudart.cudaError.success
                    chunk.registered = bool(ok)
                    if not ok:
                        raise RuntimeError("cudaHostRegister failed")
                    _STATE["bytes"] += chunk.end - chunk.start
                    chunk.ready.set()
            finally:
                os.close(fd)
    except Exception as exc:  # noqa: BLE001 - the chunks not ready take the loader's own staging
        _STATE["error"] = repr(exc)
        for entry in files:
            entry.failed = True
    finally:
        for entry in files:
            for chunk in entry.chunks:
                chunk.ready.set()  # nobody waits for a chunk that will not come (failed: not used)
        _STATE["seconds"] = round(time.perf_counter() - started, 3)


def start(argv=None):
    """Starts the reading thread (once) when the switch is on and the checkpoint is found. Returns the
    thread or None."""
    if not enabled() or not torch.cuda.is_available():
        return None
    with _LOCK:
        if _STATE["thread"] is not None:
            return _STATE["thread"]
        try:
            base = _config_base(argv)
            files = [_plan(path) for path in checkpoint_files(base)] if base else []
        except Exception as exc:  # noqa: BLE001 - nothing read ahead
            _STATE["error"] = f"plan: {exc!r}"
            return None
        if not files:
            return None
        _STATE["files"] = {entry.path: entry for entry in files}
        thread = threading.Thread(target=_read, args=(files, torch.cuda.current_device()),
                                  name="soup-load-early", daemon=True)
        _STATE["thread"] = thread
        thread.start()
        return thread


def view(path, offset, nbytes):
    """The uint8 host view of the file range [offset, offset + nbytes) of `path`, pinned, once its chunk
    is read (waits for it); None when it is not covered or the read failed."""
    entry = _covered(path)
    if entry is None:
        return None
    chunk = entry.chunk_of(offset, nbytes)
    if chunk is None:
        return None
    chunk.ready.wait()
    if entry.failed or not chunk.registered or entry.buffer is None:
        return None
    begin = offset - entry.data_start
    _STATE["used"] += 1
    return entry.buffer[begin:begin + nbytes]


def _covered(path):
    """The _File read ahead for `path`, a file transformers opened, when it is the same file (real path,
    size and modification time), else None. Resolved once per path (every tensor of a load asks)."""
    if not _STATE["files"] or _STATE["released"]:
        return None
    covered = _STATE.setdefault("covered", {})  # path as transformers opened it -> _File or None
    if path in covered:
        return covered[path]
    entry = _STATE["files"].get(os.path.realpath(path))
    if entry is not None:
        try:
            stat = os.stat(entry.path)
            if (stat.st_size, stat.st_mtime_ns) != entry.stamp:
                entry = None
        except OSError:
            entry = None
    covered[path] = entry
    return entry


def covers(path):
    """Whether `path`, a file transformers opened, is one read ahead: the same real path, size and
    modification time."""
    return _covered(path) is not None


def release(log=print):
    """After the load's copies are done (or when the load does not use them): unregisters the chunks
    and frees the memory, in a thread. Idempotent."""
    with _LOCK:
        if _STATE["released"] or not _STATE["files"]:
            return
        _STATE["released"] = True
        _STATE.setdefault("covered", {}).clear()
        files = list(_STATE["files"].values())
    thread = _STATE["thread"]

    def run():
        if thread is not None:
            thread.join()
        cudart = torch.cuda.cudart()
        for entry in files:
            if entry.buffer is None:
                continue
            for chunk in entry.chunks:
                if chunk.registered:
                    cudart.cudaHostUnregister(entry.buffer[chunk.start:chunk.end].data_ptr())
                    chunk.registered = False
            entry.buffer = None

    threading.Thread(target=run, name="soup-load-early-release", daemon=True).start()
    if log is not None:
        log(f"[solution] load early: {_STATE['bytes'] / 2**20:.0f} MiB of the checkpoint read into pinned host memory "
            f"from apply begin in {_STATE['seconds'] if _STATE['seconds'] is not None else float('nan'):.2f} s, "
            f"{_STATE['used']} tensors staged from it" + (f"; {_STATE['error']}" if _STATE["error"] else ""))
