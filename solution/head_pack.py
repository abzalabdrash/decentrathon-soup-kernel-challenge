"""The streamed head's weight in a lossless 12-bit form for the PCIe transfer, decoded into the chunk
slot by our own Triton kernel (SOUP_SOLUTION_HEAD_PACK=1; off by default).

head_stream.py moves the tied embedding / lm_head weight (151,936 x 1,536 fp16, 445 MiB) through the
PCIe bus once per step, and the head waits for those copies (~12 GB/s on the T4): only fewer bytes
shorten it. Qwen2.5-1.5B's checkpoint is bf16 and the run holds it in fp16, so every fp16 value of
the weight came from a bf16 value. Measured on the real checkpoint (233,373,696 values):
- normal values (exponent fields 1..13) carry 7 mantissa bits: their 3 low fp16 mantissa bits are
  always 0;
- the 482,053 fp16 subnormals and zeros fall into four exact classes by their 10-bit mantissa m:
  m < 128 (m itself), 128 <= m < 256 (m - 128), 256 <= m < 512 and even ((m - 256) / 2), 512 <= m
  and a multiple of 4 ((m - 512) / 4), i.e. bf16's 8 significant bits shifted into the subnormal
  range;
- so every value is sign + 7 stored bits + one of 17 classes, each class an affine map
  bits = (sign << 15) | (ADD + (M << SH)): normal exponent e: ADD = e << 10, SH = 3; the subnormal
  classes: (0, 0), (128, 0), (256, 1), (512, 2). The exponent entropy is 2.58 bits (an entropy coder
  would reach 66 %); a fixed 4-bit class code gives 75 % with a decode that is pure byte shuffling.

Format, per vocabulary chunk of R rows (n = rows x D values; all chunks but the last have R rows):
    A plane, n bytes: (sign << 7) | M
    B plane, n / 2 bytes: the class code of value 2j in the low nibble of byte j, of 2j + 1 in the
    high nibble; the code indexes the chunk's own table of 16 classes (the 16 most frequent in the
    chunk), entry = ADD | (SH << 16)
    escapes (at most MAX_ESC per chunk): values whose class is not in the chunk's table (here the 15
    values with exponent field 13, the 17th class), kept as (index, fp16 bits) on the GPU and patched
    by the decode kernel in registers before its store.
The host keeps 1.5 n bytes per chunk (334 MiB instead of 445; pinned), the copy stream moves them
into a uint8 slot, and `head_unpack_kernel` (own kernel, one program per BLOCK values) writes the
fp16 chunk into a reused buffer right before the chunk's GEMMs, on the compute stream.

Building (once, in a thread started at peft's kbit preparation, while Soup builds the trainer): the
fp16 weight goes through the GPU chunk by chunk (from its pinned host copy, on the default stream so
that every block it allocates is reused by training), the classes are counted, the chunk is encoded
with torch ops, decoded with the training kernel and compared with the fp16 chunk bit for bit, and
the packed chunk goes to the pinned host buffer. Any mismatch, a value no class fits more than
MAX_ESC times in a chunk, a dtype other than fp16, or an error leaves the pack unbuilt: the head
streams fp16 as before. Until the pack is ready (and on any head whose ring was made earlier) the
head streams fp16; the choice is made once per head, when its ring is allocated.
"""

import atexit
import threading
import time
import weakref

import torch
import triton
import triton.language as tl

MAX_ESC = 32  # escapes per chunk (int32 slots, 128 bytes: the per-chunk pointers stay 16-byte aligned)
TABLE = 16  # classes per chunk table (a 4-bit code)
# pairs of values per program of the decode kernel and its warps (an RTX 5060 laptop GPU, one chunk of
# 4,096 rows: 25-33 us against 104 us for one value per lane with a gathered nibble load)
BLOCK, WARPS = 1024, 8
N_CLASSES = 37  # class ids: 1..31 normal exponent fields, 32..35 the subnormal classes, 36 none
SUB_ROWS = 2048  # rows encoded at a time while building (an int32 index of 12 MiB)
_SUBNORMAL = ((0, 0), (128, 0), (256, 1), (512, 2))  # class id 32 + k: (ADD, SH)


@triton.jit(do_not_specialize=["n", "n_esc"])
def head_unpack_kernel(
    src_ptr,  # uint8: the packed chunk, A plane (n bytes) then B plane (n / 2 bytes)
    dst_ptr,  # int32: the fp16 chunk as n / 2 words of two values (little-endian: value 2j low)
    table_ptr,  # int32 (16,): ADD | (SH << 16) per class code
    esc_pos_ptr, esc_val_ptr,  # int32 (MAX_ESC,): value index and fp16 bits of the escapes
    n, n_esc,  # values (even), escapes
    BLOCK: tl.constexpr,  # pairs per program
):
    # pair j = values 2j and 2j + 1: A bytes 2j and 2j + 1 in one 16-bit load, both codes in B byte
    # j, both fp16 results in one 32-bit store (every access contiguous across the lanes)
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = j < n // 2
    b = tl.load(src_ptr + n + j, mask=mask, other=0).to(tl.int32)
    a = tl.load(src_ptr.to(tl.pointer_type(tl.int16)) + j, mask=mask, other=0).to(tl.int32) & 0xFFFF
    e0 = tl.load(table_ptr + (b & 15))
    e1 = tl.load(table_ptr + (b >> 4))
    a0 = a & 255
    a1 = a >> 8
    v0 = ((a0 >> 7) << 15) | ((e0 & 0xFFFF) + ((a0 & 127) << (e0 >> 16)))
    v1 = ((a1 >> 7) << 15) | ((e1 & 0xFFFF) + ((a1 & 127) << (e1 >> 16)))
    for k in range(n_esc):
        p = tl.load(esc_pos_ptr + k)
        v = tl.load(esc_val_ptr + k)
        v0 = tl.where(2 * j == p, v, v0)
        v1 = tl.where(2 * j + 1 == p, v, v1)
    tl.store(dst_ptr + j, v0 | (v1 << 16), mask=mask)


class PackSkip(Exception):
    """The weight does not fit the format; the head streams fp16."""


def _entry(cid):
    if cid < 32:
        return (cid << 10) | (3 << 16)
    add, sh = _SUBNORMAL[cid - 32]
    return add | (sh << 16)


def classify(x16):
    """(A plane bytes, class ids) of fp16 values (any shape, flattened): uint8 tensors on x16's device."""
    v = x16.reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    mag = v & 0x7FFF
    e = mag >> 10
    m = mag & 1023
    sub = e == 0
    # stored bits and class of a normal value (exact when its 3 low mantissa bits are 0)
    stored = m >> 3
    cid = torch.where((m & 7) == 0, e, N_CLASSES - 1)
    # subnormals and zeros: the class from the size of m, exact when its low bits fit the shift
    sub_k = (m >= 128).int() + (m >= 256).int() + (m >= 512).int()
    sub_sh = (sub_k - 1).clamp(min=0)
    sub_add = torch.where(sub_k == 0, 0, 64 << sub_k)
    sub_ok = ((m - sub_add) & ((1 << sub_sh) - 1)) == 0
    stored = torch.where(sub, (m - sub_add) >> sub_sh, stored)
    cid = torch.where(sub, torch.where(sub_ok, 32 + sub_k, N_CLASSES - 1), cid)
    a = ((v >> 15) << 7) | stored
    return a.to(torch.uint8), cid.to(torch.uint8)


def bit_tables(device):
    """(A byte, class id) of every 16-bit pattern: two uint8 tables of 65,536 entries on `device`, so
    that encoding a block of values is a mask and two gathers (few launches, little memory)."""
    every = torch.arange(1 << 16, dtype=torch.int32, device=device).to(torch.int16).view(torch.float16)
    return classify(every)


def decode_chunk(packed, out, table, esc_pos, esc_val, n_esc, block=None, warps=None):
    """Writes the fp16 values of a packed chunk into `out` (fp16, contiguous, an even number of
    values, 4-byte aligned) on the current stream."""
    n = out.numel()
    block = block or BLOCK
    head_unpack_kernel[(triton.cdiv(n // 2, block),)](
        packed, out.view(torch.int32), table, esc_pos, esc_val, n, n_esc, BLOCK=block,
        num_warps=warps or WARPS)


class HeadPack:
    """The packed form of a HeadStreamer's weight (built by `start()` in a thread)."""

    def __init__(self, streamer, log=print):
        if streamer.dtype != torch.float16:
            raise PackSkip(f"weight dtype {streamer.dtype}")
        if streamer.hidden % 2:
            raise PackSkip("odd hidden size")
        self.rows, self.hidden, self.vocab = streamer.rows, streamer.hidden, streamer.vocab
        device = torch.device(streamer.device)
        self.n_chunks = streamer.n_chunks
        self.device = device if device.index is not None else torch.device("cuda", torch.cuda.current_device())
        self.slot_bytes = self.rows * self.hidden * 3 // 2
        self.ready, self.error, self.seconds, self.escapes = False, None, 0.0, 0
        self.log = log
        self._source = streamer.host  # the fp16 host copy (pinned)
        self._stop = threading.Event()
        self._thread = None
        self.host = self._unpin = None
        self.tables = self.esc_pos = self.esc_val = None
        self.n_esc = [0] * self.n_chunks

    def chunk(self, c):
        """(first row, rows, first packed byte, packed bytes) of chunk c."""
        c0 = c * self.rows
        vc = min(self.rows, self.vocab - c0)
        return c0, vc, c * self.slot_bytes, vc * self.hidden * 3 // 2

    def start(self):
        self._thread = threading.Thread(target=self._build_safely, name="soup-head-pack", daemon=True)
        self._thread.start()
        # a process that ends while the thread works (a short test) waits for its current chunk: a
        # daemon thread inside a CUDA call while Python finalizes can abort the process
        atexit.register(_join_at_exit, weakref.ref(self))
        return self

    def join(self):
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self.ready = False
        self.tables = self.esc_pos = self.esc_val = None
        if self._unpin is not None:
            self._unpin()
            self._unpin = None
        self.host = None

    def _build_safely(self):
        started = time.perf_counter()
        try:
            self.build()
        except Exception as exc:  # noqa: BLE001 - an optimisation must not break the run
            self.error = repr(exc)
        self.seconds = time.perf_counter() - started
        if self.ready:
            self.log(f"[solution] head pack: {self.vocab} x {self.hidden} fp16 in {self.n_chunks} chunks of "
                     f"{self.rows} rows packed to 12 bits ({self.total_bytes / 2**20:.0f} MiB pinned, "
                     f"{self.escapes} escapes), verified bit for bit, {self.seconds:.2f} s")
        elif not self._stop.is_set():
            self.log(f"[solution] head pack off: {self.error}")

    def build(self):
        from .weight_stream import pinned_host_buffer

        torch.cuda.set_device(self.device)
        total = self.chunk(self.n_chunks - 1)[2] + self.chunk(self.n_chunks - 1)[3]
        host, unpin = pinned_host_buffer(total, self)
        self.total_bytes = total
        tables = torch.zeros(self.n_chunks, TABLE, dtype=torch.int32)
        esc_pos = torch.zeros(self.n_chunks, MAX_ESC, dtype=torch.int32)
        esc_val = torch.zeros(self.n_chunks, MAX_ESC, dtype=torch.int32)
        dev = self.device
        # the default stream: the blocks allocated here go back to the pool training allocates from
        stream = torch.cuda.default_stream(dev)
        with torch.cuda.stream(stream):
            g_tables = torch.empty(self.n_chunks, TABLE, dtype=torch.int32, device=dev)
            g_pos = torch.empty(self.n_chunks, MAX_ESC, dtype=torch.int32, device=dev)
            g_val = torch.empty(self.n_chunks, MAX_ESC, dtype=torch.int32, device=dev)
            a_of, cid_of = bit_tables(dev)
            for c in range(self.n_chunks):
                if self._stop.is_set():
                    unpin()
                    raise PackSkip("stopped")
                c0, vc, b0, nb = self.chunk(c)
                x = self._source[c0:c0 + vc].to(dev, non_blocking=True)
                n = x.numel()
                packed = torch.empty(nb, dtype=torch.uint8, device=dev)
                cid = torch.empty(n, dtype=torch.uint8, device=dev)
                hist = torch.zeros(N_CLASSES, dtype=torch.int64, device=dev)
                step = SUB_ROWS * self.hidden
                for i in range(0, n, step):  # A plane and class ids, a block of rows at a time
                    j = min(i + step, n)
                    bits = x.view(-1)[i:j].view(torch.int16).to(torch.int32)
                    bits &= 0xFFFF
                    torch.index_select(a_of, 0, bits, out=packed[i:j])
                    torch.index_select(cid_of, 0, bits, out=cid[i:j])
                    hist += torch.bincount(cid[i:j], minlength=N_CLASSES)
                hist = hist.tolist()
                present = sorted((k for k in range(1, N_CLASSES - 1) if hist[k]), key=lambda k: -hist[k])
                chosen = present[:TABLE]
                lut = torch.full((N_CLASSES,), TABLE, dtype=torch.uint8)
                for code, k in enumerate(chosen):
                    lut[k] = code
                    tables[c, code] = _entry(k)
                lut = lut.to(dev)
                code = torch.empty(n, dtype=torch.uint8, device=dev)
                for i in range(0, n, step):
                    j = min(i + step, n)
                    torch.index_select(lut, 0, cid[i:j].int(), out=code[i:j])
                del cid
                escaped = torch.nonzero(code == TABLE).flatten()
                k = escaped.numel()
                if k > MAX_ESC:
                    unpin()
                    raise PackSkip(f"chunk {c}: {k} values outside its 16 classes (at most {MAX_ESC})")
                if k:
                    code[escaped] = 0
                    esc_pos[c, :k] = escaped.int().cpu()
                    esc_val[c, :k] = x.reshape(-1).view(torch.int16)[escaped].int().cpu() & 0xFFFF
                self.n_esc[c] = k
                self.escapes += k
                pairs = code.view(-1, 2)
                torch.bitwise_or(pairs[:, 0], pairs[:, 1] << 4, out=packed[n:])
                g_tables[c].copy_(tables[c])
                g_pos[c].copy_(esc_pos[c])
                g_val[c].copy_(esc_val[c])
                out = torch.empty_like(x)
                decode_chunk(packed, out, g_tables[c], g_pos[c], g_val[c], k)
                if not torch.equal(out.view(torch.int16), x.view(torch.int16)):
                    unpin()
                    raise PackSkip(f"chunk {c}: the decoded chunk differs from the fp16 weight")
                host[b0:b0 + nb].copy_(packed)  # synchronous: the pinned bytes are final after it
            stream.synchronize()
        self.host, self._unpin = host, unpin
        self.tables, self.esc_pos, self.esc_val = g_tables, g_pos, g_val
        self.ready = True

    def host_chunk(self, c):
        _, _, b0, nb = self.chunk(c)
        return self.host[b0:b0 + nb]

    def decode(self, c, slot, out):
        """The fp16 chunk c from its packed bytes in `slot` into `out` (vc x D fp16)."""
        decode_chunk(slot, out, self.tables[c], self.esc_pos[c], self.esc_val[c], self.n_esc[c])


def _join_at_exit(ref):
    pack = ref()
    if pack is not None:
        pack._stop.set()
        pack.join()


def warm_kernels():
    from .warmup import compile_kernel

    compile_kernel(head_unpack_kernel, torch.uint8, torch.int32, torch.int32, torch.int32, torch.int32,
                   6_291_456, 0, BLOCK=BLOCK, num_warps=WARPS)
