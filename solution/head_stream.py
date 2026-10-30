"""The tied embedding / lm_head weight kept in pinned host memory for the whole run, and the output
head computed in one streamed pass over the vocabulary (own Triton kernels).

The 151,936 x 1536 weight of Qwen2.5-1.5B (445 MiB in fp16) was the largest static tensor of the
run, and the fused head's logits chunk (512 rows x V, 148 MiB) set the step's peak. Here neither
exists on the GPU: the weight lives in page-locked host memory and goes through the GPU once per
step in chunks of R vocabulary rows (8,192 by default: 24 MiB) on a ring of K slots (2 by default).

Setup (peft's `prepare_model_for_kbit_training` casts every fp16 parameter to fp32, the embedding
included: 933 MiB next to the fp16 copy, the setup's peak). Right before it is called inside Soup's
`SFTTrainerWrapper._setup_transformers`, the fp16 weight is copied to pinned host memory and the
Parameter (the same object for `embed_tokens.weight` and `lm_head.weight`) gets a placeholder as its
`.data`: one fp32 element on the GPU expanded to (V, D), 4 bytes. peft sees an fp32 parameter and
leaves it alone; `model.dtype`, `.device`, `.shape` and the parameter count stay what they are in the
base after the cast (fp32 on the GPU). Nothing reads the placeholder's values: the embedding and the
lm_head modules get instance-level `forward`s that read the host copy, and the fused head
(fused_ce.py) dispatches here when it is handed this Parameter. A `copy.deepcopy` of the Parameter
(peft's modules_to_save copies a module) yields the real fp32 weight. After `_setup_transformers`
returns, the setup is checked: if the head or the embedding became trainable, wrapped by peft or
replaced, the real weight goes back on the GPU in fp32 (the base's state) and every override is
removed. The saved adapter holds only the LoRA tensors, as before.

Embedding lookup: the ids come to the host (one synchronization; the packed path synchronizes right
before anyway), the rows are gathered from the host copy into a pinned staging buffer and copied to
the GPU (2.5 MB for 826 tokens), then cast to fp32: the base's lookup in its fp32 weight gives the
same values bit for bit (the fp16 -> fp32 round trip is exact). A training forward starts the copies
of the first K chunks of the head at the final norm (a pre-hook), after the last layer: its MLP sets
the forward's peak, and a ring allocated at the embedding lookup (more overlap, about 2 ms per 24 MiB
chunk) was alive there (Kaggle T4, step 6: +48 MiB on the step's peak).

The head, for N hidden rows (labelled rows first, as in fused_ce.py) and chunk c of W:
    L_c = h16 @ W_c^T                               fp16 cuBLAS GEMM, (N, R), in a reused buffer
    head_chunk_kernel (own, one program per row):   online max m, sum s = sum exp(x - m), entropy
        accumulator e = sum exp(x - m) * x, argmax (first occurrence, as torch), the target logit when
        the target falls in this chunk; for labelled rows: P_c = exp(L_c - m_new) written in place in
        fp16, acc *= exp(m_old - m_new) (skipped when the max did not move), and W[target] copied
        from the resident chunk when the target is in it
    acc += P_c @ W_c                                fp16 cuBLAS GEMM with an fp32 output, labelled rows
    head_finish_kernel (own): loss = lse - x_t, entropy = lse - e / s, correct = (argmax == target),
        dL/dh = (acc / s - W[target]) * grad_scale in place in acc (fp32)
W is read once per step: 445 MB over PCIe (about 37 ms at 12 GB/s), the copy of chunk c + K
overlapping the GEMMs of chunk c; the copy stream waits for the compute stream through events (the
ring is freed only after the compute stream has waited for every copy into it: no record_stream).

Numerics: the statistics are fp32 as before (loss and entropy agree with the resident head to fp32
rounding). The base rounds the fp32 softmax-minus-onehot to fp16 before its gradient GEMM and that
GEMM's output to fp16; here P is rounded to fp16 relative to the running max (never below the base's
normalized values), the GEMM accumulates and outputs fp32, and dL/dh stays fp32: the same precision
class, slightly closer to exact.

Guards: only a CUDA Qwen2ForCausalLM with the fused head patched, a tied plain nn.Embedding / nn.Linear
pair without bias, an fp16 or bf16 weight, and peft about to cast it (not GPTQ and the like), inside
Soup's setup. Eval and generation (the original forward) run through the lm_head override: logits in
full, chunk by chunk, with the base's dtypes. Switches (environment): SOUP_SOLUTION_HEAD_STREAM=0,
SOUP_SOLUTION_HEAD_ROWS (vocabulary rows per chunk), SOUP_SOLUTION_HEAD_SLOTS (ring slots, >= 2),
SOUP_SOLUTION_HEAD_PREFETCH=norm|embedding|head (where the first chunks start).
"""

import atexit
import collections
import contextlib
import functools
import os
import time
import weakref

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .fused_ce import IGNORE_INDEX, compute_dtype

DEFAULT_ROWS = 8192  # vocabulary rows per chunk: 8192 x 1536 x 2 bytes = 24 MiB per slot
DEFAULT_SLOTS = 2
# where a training forward starts copying the head's first chunks: "norm" (the final norm, after the
# last layer, whose MLP sets the forward's peak: the ring is not alive then), "embedding" (the start
# of the forward: more overlap, the ring alive through the forward), "head" (no early copies)
DEFAULT_PREFETCH = "norm"
MAX_ROWS = 32768  # one Triton block holds a row's whole chunk
BLOCK_D = 512
STAGING_ROWS = 4096  # pinned staging rows of the embedding lookup (grows on demand)
NOT_CAST = ("gptq", "aqlm", "eetq", "torchao", "hqq")  # peft leaves these models' parameters alone

_STREAMERS = {}  # id(Parameter) -> HeadStreamer (tensors compare elementwise: no weak-key dict)
_SETUP = {"depth": 0}  # > 0 while Soup's _setup_transformers runs


@triton.jit(do_not_specialize=["c0", "vc", "n_lab"])
def head_chunk_kernel(
    logits_ptr, stride_l,  # half (rows, vc): this chunk's logits; P written in place for labelled rows
    targets_ptr,  # int64 (rows,): target token id or IGNORE
    m_ptr, s_ptr, e_ptr, best_ptr, xt_ptr,  # fp32 (rows,): running max, sum, entropy sum, best, x_t
    idx_ptr,  # int32 (rows,): running argmax
    acc_ptr, stride_acc,  # fp32 (n_lab, D): rescaled here by exp(m_old - m_new)
    w_ptr, stride_w,  # half (vc, D): the chunk of W on the GPU
    wt_ptr, stride_wt,  # half (n_lab, D): W[target], written by the chunk that holds the target
    c0, vc, n_lab, D,  # first vocabulary id of the chunk, its rows, labelled rows, hidden size
    FIRST: tl.constexpr, IGNORE: tl.constexpr, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < vc
    base = logits_ptr + row.to(tl.int64) * stride_l
    x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
    target = tl.load(targets_ptr + row)
    t_local = target - c0
    hit = (target != IGNORE) & (t_local >= 0) & (t_local < vc)
    # the target's logit, read before the reductions below: their barriers order this load before
    # any warp's in-place store of P (a reduction over the block for it cost 20% of the kernel)
    x_hit = tl.load(base + tl.where(hit, t_local, 0), mask=hit, other=0.0).to(tl.float32)
    # one reduction for the max and its first occurrence inside the chunk (torch.argmax's tie rule)
    c_max, c_idx = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
    if FIRST:
        m_old = -float("inf")
        s = 0.0
        e = 0.0
        best = -float("inf")
        best_idx = 0
        x_t = 0.0
    else:
        m_old = tl.load(m_ptr + row)
        s = tl.load(s_ptr + row)
        e = tl.load(e_ptr + row)
        best = tl.load(best_ptr + row)
        best_idx = tl.load(idx_ptr + row)
        x_t = tl.load(xt_ptr + row)
    m_new = tl.maximum(m_old, c_max)
    alpha = tl.exp(m_old - m_new)
    p = tl.exp(x - m_new)
    s = s * alpha + tl.sum(p, axis=0)
    e = e * alpha + tl.sum(p * tl.where(mask, x, 0.0), axis=0)
    best_idx = tl.where(c_max > best, c0 + c_idx, best_idx)  # strictly greater: earlier chunks win ties
    best = tl.maximum(best, c_max)
    x_t = tl.where(hit, x_hit, x_t)
    tl.store(m_ptr + row, m_new)
    tl.store(s_ptr + row, s)
    tl.store(e_ptr + row, e)
    tl.store(best_ptr + row, best)
    tl.store(idx_ptr + row, best_idx)
    tl.store(xt_ptr + row, x_t)
    if row < n_lab:
        tl.store(base + offs, p.to(logits_ptr.dtype.element_ty), mask=mask)
        if not FIRST:
            if alpha < 1.0:
                acc_row = acc_ptr + row.to(tl.int64) * stride_acc
                for d0 in range(0, D, BLOCK_D):
                    od = d0 + tl.arange(0, BLOCK_D)
                    md = od < D
                    a = tl.load(acc_row + od, mask=md)
                    tl.store(acc_row + od, a * alpha, mask=md)
        if hit:
            w_row = w_ptr + t_local * stride_w
            wt_row = wt_ptr + row.to(tl.int64) * stride_wt
            for d0 in range(0, D, BLOCK_D):
                od = d0 + tl.arange(0, BLOCK_D)
                md = od < D
                tl.store(wt_row + od, tl.load(w_row + od, mask=md), mask=md)


@triton.jit(do_not_specialize=["n_lab"])
def head_finish_kernel(
    targets_ptr, m_ptr, s_ptr, e_ptr, xt_ptr, idx_ptr,  # the state left by head_chunk_kernel
    loss_ptr, entropy_ptr, correct_ptr,  # fp32, fp32, int32 (rows,)
    acc_ptr, stride_acc,  # fp32 (n_lab, D): sum_v exp(x_v - m) W_v in, dL/dh out
    wt_ptr, stride_wt,  # half (n_lab, D): W[target]
    grad_scale, n_lab, D,
    IGNORE: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    m = tl.load(m_ptr + row)
    s = tl.load(s_ptr + row)
    e = tl.load(e_ptr + row)
    lse = m + tl.log(s)
    target = tl.load(targets_ptr + row)
    labelled = target != IGNORE
    tl.store(loss_ptr + row, tl.where(labelled, lse - tl.load(xt_ptr + row), 0.0))
    tl.store(entropy_ptr + row, lse - e / s)
    tl.store(correct_ptr + row, (labelled & (tl.load(idx_ptr + row) == target)).to(tl.int32))
    if row < n_lab:
        scale = grad_scale / s
        acc_row = acc_ptr + row.to(tl.int64) * stride_acc
        wt_row = wt_ptr + row.to(tl.int64) * stride_wt
        for d0 in range(0, D, BLOCK_D):
            od = d0 + tl.arange(0, BLOCK_D)
            md = od < D
            a = tl.load(acc_row + od, mask=md)
            w = tl.load(wt_row + od, mask=md).to(tl.float32)
            tl.store(acc_row + od, a * scale - w * grad_scale, mask=md)


def _num_warps(block):
    # a row's chunk lives in registers: 16 values per thread from 4096 on keeps two programs of 16
    # warps per T4 SM (1024 threads); 32 values per thread halved the occupancy (4.7 ms per step
    # for the chunk kernel on the T4 with 8 warps against 2.2 ms for the resident head's two passes)
    return 4 if block <= 2048 else (8 if block <= 4096 else 16)


def _fp32_out_gemm_works(device):
    """torch.addmm with fp16 inputs and an fp32 output (`out_dtype`, torch >= 2.8) on this device."""
    try:
        a = torch.full((16, 16), 0.5, dtype=torch.float16, device=device)
        out = torch.full((16, 16), float("nan"), device=device)
        torch.addmm(out, a, a, beta=0, out_dtype=torch.float32, out=out)
        torch.addmm(out, a, a, out_dtype=torch.float32, out=out)
        return bool(torch.all(out == 8.0).item())
    except (RuntimeError, TypeError):
        return False


class HeadSkip(Exception):
    """The model is not one this module streams; it is left untouched."""


class HeadStreamer:
    """The weight of a tied embedding / lm_head pair in pinned host memory, with the overrides that
    read it. Construction does the move (the Parameter keeps a 4-byte placeholder of dtype
    `virtual_dtype` on the GPU); `restore()` puts the weight back on the GPU in `virtual_dtype`.

    Counters (see `stats()`): chunk copies, those issued only when the head needed them (not
    prefetched), copy time, and the time the compute stream waited for copies in the head."""

    def __init__(self, param, embedding=None, lm_head=None, rows=DEFAULT_ROWS, slots=DEFAULT_SLOTS,
                 virtual_dtype=torch.float32, prefetch_at=DEFAULT_PREFETCH, norm=None, source=None,
                 device=None, pinned=None):
        """`source`: the weight's values when they are not in `param` yet (the loader read them to
        the host, load_stream.py); then `device` is the GPU the weight belongs on."""
        data = param.data if source is None else source
        if not torch.cuda.is_available() or not (data.is_cuda if source is None else device is not None):
            raise HeadSkip("the weight is not on a CUDA device")
        if data.dtype not in (torch.float16, torch.bfloat16):
            raise HeadSkip(f"weight dtype {data.dtype}")
        if data.dim() != 2 or not data.is_contiguous() or data.numel() == 0:
            raise HeadSkip("not a contiguous 2-D weight")
        if _registered(param) is not None:
            raise HeadSkip("the weight is streamed already")
        from .weight_stream import pinned_host_buffer

        started = time.perf_counter()
        self.vocab, self.hidden = data.shape
        self.dtype, self.virtual_dtype = data.dtype, virtual_dtype
        self.device = data.device if source is None else torch.device(device)
        self._ring = None
        self.configure(rows, slots)
        nbytes = data.numel() * data.element_size()
        self._pinned = pinned if pinned is not None and pinned.buffer is not None and pinned.nbytes >= nbytes \
            else None
        if self._pinned is not None:  # pinned ahead while the loader read the checkpoint
            raw, self._unpin = pinned.buffer[:nbytes], pinned.unpin
        else:
            raw, self._unpin = pinned_host_buffer(nbytes, self)
        self.host = raw.view(self.dtype).view(self.vocab, self.hidden)
        self.host.copy_(data)
        torch.cuda.synchronize(self.device)
        self.fp32_out = None  # probed at the first head: the probe is the process's first cuBLAS call
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._staging, self._staging_done = None, None
        self._spans = collections.deque()
        self._spare_events = []
        self.loads = self.late_loads = self.heads = self.copies_timed = 0
        self.copy_ms = self.wait_ms = 0.0
        self.copy_bytes = 0
        self.param_ref = weakref.ref(param)
        self.embedding, self.lm_head = embedding, lm_head
        self._placeholder = torch.zeros(1, dtype=virtual_dtype, device=self.device).expand(
            self.vocab, self.hidden)
        param.data = self._placeholder  # the GPU copy goes with the last reference to it
        param.__deepcopy__ = functools.partial(_deepcopy_param, weakref.ref(self), weakref.ref(param))
        ref = weakref.ref(self)
        for module, fn in ((embedding, _embedding_forward), (lm_head, _lm_head_forward)):
            if module is not None:
                module.forward = functools.partial(fn, ref, module)
        self.prefetch_at = prefetch_at if prefetch_at in ("norm", "embedding", "head") else DEFAULT_PREFETCH
        if self.prefetch_at == "norm" and norm is None:
            self.prefetch_at = "head"
        self._hook = None
        if self.prefetch_at == "norm":
            self._hook = norm.register_forward_pre_hook(functools.partial(_prefetch_hook, ref))
        _STREAMERS[id(param)] = self
        weakref.finalize(param, _forget, id(param), ref)
        self.setup_seconds = time.perf_counter() - started

    def configure(self, rows=DEFAULT_ROWS, slots=DEFAULT_SLOTS):
        """Vocabulary rows per chunk and ring slots (between heads only)."""
        if self._ring is not None:
            raise RuntimeError("solution.head_stream: configure() while chunks are in flight")
        if not isinstance(slots, int) or slots < 2:
            raise HeadSkip(f"slots={slots!r}: at least 2")
        if not isinstance(rows, int) or not 16 <= rows <= MAX_ROWS:
            raise HeadSkip(f"rows={rows!r}: 16 .. {MAX_ROWS}")
        self.rows, self.slots = min(rows, self.vocab), slots
        self.n_chunks = -(-self.vocab // self.rows)
        self.block = triton.next_power_of_2(self.rows)
        self.num_warps = _num_warps(self.block)
        self._ready = [torch.cuda.Event() for _ in range(slots)]
        self._free = [torch.cuda.Event() for _ in range(slots)]
        self._owner = [-1] * slots

    @property
    def host_bytes(self):
        return self.host.numel() * self.host.element_size()

    def owns(self, param):
        """True while `param` still holds this streamer's placeholder."""
        return (self.host is not None and _registered(param) is self
                and param.data_ptr() == self._placeholder.data_ptr()
                and param.dtype == self.virtual_dtype and tuple(param.shape) == (self.vocab, self.hidden))

    def materialize(self, dtype=None, device=None):
        """The real weight as a new tensor (on the GPU in the virtual dtype by default)."""
        return self.host.to(device or self.device).to(dtype or self.virtual_dtype)

    def set_virtual_dtype(self, dtype):
        """The dtype the Parameter shows (its placeholder's): fp32 once peft would have cast it."""
        param = self.param_ref()
        if param is None or not self.owns(param) or dtype == self.virtual_dtype:
            return
        self._placeholder = torch.zeros(1, dtype=dtype, device=self.device).expand(self.vocab, self.hidden)
        self.virtual_dtype = dtype
        param.data = self._placeholder

    # ---- the ring --------------------------------------------------------------------------

    def _event(self):
        return self._spare_events.pop() if self._spare_events else torch.cuda.Event(enable_timing=True)

    def _harvest(self, block=False):
        while self._spans:
            kind, begin, end, nbytes = self._spans[0]
            if block:
                end.synchronize()
            elif not end.query():
                break
            ms = begin.elapsed_time(end)
            if kind == "copy":
                self.copy_ms += ms
                self.copy_bytes += nbytes
                self.copies_timed += 1
            else:
                self.wait_ms += ms
            self._spans.popleft()
            self._spare_events += (begin, end)

    def _chunk(self, c):
        c0 = c * self.rows
        return c0, min(self.rows, self.vocab - c0)

    def _load(self, c):
        """Starts the copy of chunk c into its slot on the copy stream, after the compute queued so
        far (which includes every use of the slot's previous chunk, and the ring's allocation)."""
        s = c % self.slots
        c0, vc = self._chunk(c)
        compute = torch.cuda.current_stream(self.device)
        self._free[s].record(compute)
        begin, end = self._event(), self._event()
        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(self._free[s])
            begin.record(self._copy_stream)
            self._ring[s, :vc].copy_(self.host[c0:c0 + vc], non_blocking=True)
            end.record(self._copy_stream)
            self._ready[s].record(self._copy_stream)
        self._spans.append(("copy", begin, end, vc * self.hidden * self.host.element_size()))
        self._owner[s] = c
        self.loads += 1

    def prefetch(self):
        """Allocates the ring (on the compute stream) and starts the copies of the first chunks."""
        if self._ring is None:
            self._ring = torch.empty(self.slots, self.rows, self.hidden, dtype=self.dtype,
                                     device=self.device)
            self._owner = [-1] * self.slots
        for c in range(min(self.slots, self.n_chunks)):
            if self._owner[c % self.slots] != c:
                self._load(c)

    def _acquire(self, c):
        """Chunk c on the GPU: the compute stream waits for its copy."""
        s = c % self.slots
        if self._owner[s] != c:
            self._load(c)
            self.late_loads += 1
        compute = torch.cuda.current_stream(self.device)
        begin, end = self._event(), self._event()
        begin.record(compute)
        compute.wait_event(self._ready[s])
        end.record(compute)
        self._spans.append(("wait", begin, end, 0))
        return self._ring[s, :self._chunk(c)[1]]

    def _release(self, c):
        """Chunk c's GEMMs are queued: its slot takes chunk c + K."""
        if c + self.slots < self.n_chunks:
            self._load(c + self.slots)

    # ---- the head --------------------------------------------------------------------------

    def head(self, hidden16, targets, n_lab, grad_scale):
        """Rows ordered labelled first: hidden (N, D) in the compute dtype, targets (N,) int64 with
        IGNORE_INDEX where there is no label, n_lab labelled rows. Returns per-row loss, entropy,
        correctness (N,) and dL/dh of the labelled rows, fp32 (max(n_lab, 1), D)."""
        n, d = hidden16.shape
        dev, dtype = hidden16.device, hidden16.dtype
        if d != self.hidden:
            raise ValueError(f"hidden size {d}, weight {self.hidden}")
        loss = torch.empty(n, dtype=torch.float32, device=dev)
        entropy = torch.empty(n, dtype=torch.float32, device=dev)
        correct = torch.empty(n, dtype=torch.int32, device=dev)
        acc = torch.empty(max(n_lab, 1), d, dtype=torch.float32, device=dev)
        if n == 0:
            return loss, entropy, correct, acc.zero_()
        stride = -(-n // 4) * 4  # 16-byte aligned rows of the state, whatever n
        state = torch.empty(5, stride, dtype=torch.float32, device=dev)
        m, s, e, best, x_t = state
        idx = torch.empty(n, dtype=torch.int32, device=dev)
        wt = torch.empty(max(n_lab, 1), d, dtype=dtype, device=dev)
        buffer = torch.empty(n * self.rows, dtype=dtype, device=dev)
        if self.fp32_out is None:
            self.fp32_out = _fp32_out_gemm_works(self.device)
        self.prefetch()  # a no-op for the chunks prefetched at the final norm (or the embedding)
        with torch.autocast("cuda", enabled=False):
            for c in range(self.n_chunks):
                c0, vc = self._chunk(c)
                w = self._acquire(c)
                if w.dtype != dtype:
                    w = w.to(dtype)
                logits = buffer[:n * vc].view(n, vc)
                torch.mm(hidden16, w.t(), out=logits)
                head_chunk_kernel[(n,)](
                    logits, logits.stride(0), targets, m, s, e, best, x_t, idx,
                    acc, acc.stride(0), w, w.stride(0), wt, wt.stride(0), c0, vc, n_lab, d,
                    FIRST=c == 0, IGNORE=IGNORE_INDEX, BLOCK=self.block, BLOCK_D=BLOCK_D,
                    num_warps=self.num_warps,
                )
                if n_lab > 0:
                    p, out = logits[:n_lab], acc[:n_lab]
                    if self.fp32_out:
                        torch.addmm(out, p, w, beta=0 if c == 0 else 1, out_dtype=torch.float32, out=out)
                    elif c == 0:
                        out.copy_(torch.mm(p, w))
                    else:
                        out.add_(torch.mm(p, w))
                self._release(c)
        head_finish_kernel[(n,)](
            targets, m, s, e, x_t, idx, loss, entropy, correct,
            acc, acc.stride(0), wt, wt.stride(0), grad_scale, n_lab, d,
            IGNORE=IGNORE_INDEX, BLOCK_D=BLOCK_D, num_warps=4,
        )
        if n_lab == 0:
            acc.zero_()
        # every copy into the ring has been waited for by the compute stream: freeing it on the
        # compute stream needs no record_stream
        self._ring, self._owner = None, [-1] * self.slots
        self.heads += 1
        self._harvest()
        return loss, entropy, correct, acc

    # ---- the module overrides --------------------------------------------------------------

    def _staging_rows(self, n):
        if self._staging_done is not None:
            self._staging_done.synchronize()  # the previous lookup's copy has left the buffer
        if self._staging is None or self._staging.shape[0] < n:
            rows = max(n, STAGING_ROWS)
            self._staging = torch.empty(rows, self.hidden, dtype=self.dtype, pin_memory=True)
        return self._staging[:n]

    def lookup(self, ids):
        """F.embedding(ids, weight) in the virtual dtype, from the host copy."""
        flat = ids.reshape(-1)
        staging = self._staging_rows(flat.numel())
        torch.index_select(self.host, 0, flat.to("cpu"), out=staging)
        device = ids.device if ids.is_cuda else self.device
        rows = torch.empty(staging.shape, dtype=self.dtype, device=device)
        rows.copy_(staging, non_blocking=True)
        if self._staging_done is None:
            self._staging_done = torch.cuda.Event()
        self._staging_done.record(torch.cuda.current_stream(device))
        return rows.to(self.virtual_dtype).view(*ids.shape, self.hidden)

    def linear(self, x, bias=None):
        """F.linear(x, weight, bias) with the base's dtypes (the weight in the virtual dtype, cast by
        autocast when it is on), chunk by chunk from the host copy."""
        autocast = x.is_cuda and torch.is_autocast_enabled("cuda")
        out = None
        for c in range(self.n_chunks):
            c0, vc = self._chunk(c)
            w = self.host[c0:c0 + vc].to(x.device, non_blocking=True)
            if not autocast:
                w = w.to(self.virtual_dtype)
            y = F.linear(x, w, None if bias is None else bias[c0:c0 + vc])
            if out is None:
                out = torch.empty(*y.shape[:-1], self.vocab, dtype=y.dtype, device=y.device)
            out[..., c0:c0 + vc] = y
        return out

    # ---- end of life -----------------------------------------------------------------------

    def restore(self):
        """The real weight back on the GPU in the virtual dtype, the overrides removed."""
        param = self.param_ref()
        if param is not None and _registered(param) is self:
            if param.data_ptr() == self._placeholder.data_ptr():
                param.data = self.materialize()
            _STREAMERS.pop(id(param), None)
            param.__dict__.pop("__deepcopy__", None)
        for module in (self.embedding, self.lm_head):
            if module is not None and isinstance(module.__dict__.get("forward"), functools.partial):
                del module.forward
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self.embedding = self.lm_head = None
        self._ring = self._staging = None
        torch.cuda.synchronize(self.device)
        if self._unpin is not None:
            self._unpin()
        self.host = None

    def stats(self):
        if self._spans:
            torch.cuda.synchronize(self.device)
            self._harvest(block=True)
        return {"heads": self.heads, "loads": self.loads, "late_loads": self.late_loads,
                "copies_timed": self.copies_timed, "copy_ms": self.copy_ms,
                "copy_gbps": self.copy_bytes / self.copy_ms / 1e6 if self.copy_ms > 0 else None,
                "wait_ms": self.wait_ms}


def _deepcopy_param(streamer_ref, param_ref, memo):
    """copy.deepcopy of the streamed Parameter: the real weight (as the base would copy it)."""
    streamer, param = streamer_ref(), param_ref()
    if streamer is None or param is None or not streamer.owns(param):
        raise RuntimeError("solution.head_stream: deepcopy of a weight that is no longer streamed")
    return torch.nn.Parameter(streamer.materialize(), requires_grad=param.requires_grad)


def _embedding_forward(streamer_ref, module, input):
    streamer = streamer_ref()
    if streamer is None or not streamer.owns(module.weight):
        return torch.nn.Embedding.forward(module, input)
    out = streamer.lookup(input)
    if streamer.prefetch_at == "embedding" and module.training and torch.is_grad_enabled():
        streamer.prefetch()  # a training forward ends in the fused head: start its first chunks
    return out


def _prefetch_hook(streamer_ref, module, args):
    """Forward pre-hook on the decoder's final norm: the last layer is done, the head comes next."""
    streamer = streamer_ref()
    if streamer is None or not (module.training and torch.is_grad_enabled()):
        return
    param = streamer.param_ref()
    if param is not None and streamer.owns(param):
        streamer.prefetch()


def _lm_head_forward(streamer_ref, module, input):
    streamer = streamer_ref()
    if streamer is None or not streamer.owns(module.weight):
        return torch.nn.Linear.forward(module, input)
    return streamer.linear(input, module.bias)


def streamer_for(weight):
    """The HeadStreamer that holds this Parameter's values, or None."""
    if not isinstance(weight, torch.Tensor):
        return None
    streamer = _registered(weight)
    return streamer if streamer is not None and streamer.owns(weight) else None


def _registered(param):
    streamer = _STREAMERS.get(id(param))
    return streamer if streamer is not None and streamer.param_ref() is param else None


def _forget(key, streamer_ref):
    if _STREAMERS.get(key) is streamer_ref():
        _STREAMERS.pop(key, None)


# ---- the loss --------------------------------------------------------------------------------


class StreamedLMHeadLoss(torch.autograd.Function):
    """fused_ce.FusedLMHeadLoss with the weight streamed from the host: same inputs, outputs, row
    order and gradient scale; dL/dh is kept in fp32 for the labelled rows only."""

    @staticmethod
    def forward(ctx, hidden2d, weight, targets, num_items, streamer):
        n, d = hidden2d.shape
        labelled = targets != IGNORE_INDEX
        order = torch.argsort(labelled.to(torch.int8), descending=True, stable=True)
        n_labelled = int(labelled.sum().item())
        if num_items is None:
            grad_scale = 1.0 / max(n_labelled, 1)
        else:
            grad_scale = 1.0 / float(num_items)
        hidden16 = hidden2d[order].to(compute_dtype())
        targets_sorted = targets[order].contiguous()
        loss_rows, entropy_rows, correct_rows, dh = streamer.head(
            hidden16, targets_sorted, n_labelled, grad_scale)
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(n, device=order.device)
        loss = loss_rows.sum() * grad_scale
        token_loss = loss_rows[inverse]
        token_entropy = entropy_rows[inverse]
        correct = correct_rows[inverse]
        ctx.save_for_backward(dh, order[:n_labelled])
        ctx.shape, ctx.in_dtype = (n, d), hidden2d.dtype
        ctx.mark_non_differentiable(token_loss, token_entropy, correct)
        return loss, token_loss, token_entropy, correct

    @staticmethod
    def backward(ctx, grad_loss, _g1, _g2, _g3):
        dh, rows = ctx.saved_tensors
        grad = torch.zeros(ctx.shape, dtype=ctx.in_dtype, device=dh.device)
        if rows.numel():
            g = dh[:rows.numel()]
            if grad_loss is not None:
                g = g * grad_loss.to(g.dtype)
            grad.index_copy_(0, rows, g.to(ctx.in_dtype))
        return grad, None, None, None, None


def streamed_lm_head_loss(hidden2d, weight, targets, num_items, streamer):
    return StreamedLMHeadLoss.apply(hidden2d, weight, targets, num_items, streamer)


# ---- setup -----------------------------------------------------------------------------------


def _base_model(model):
    get = getattr(model, "get_base_model", None)
    if callable(get):
        try:
            return get()
        except Exception:  # noqa: BLE001 - not a peft model after all
            return model
    return model


def _tied_pair(model):
    """(Parameter, embedding, lm_head) of a CUDA Qwen2ForCausalLM with the fused head patched and a
    tied plain nn.Embedding / nn.Linear pair; HeadSkip otherwise."""
    from transformers.models.qwen2 import modeling_qwen2

    base = _base_model(model)
    if not isinstance(base, modeling_qwen2.Qwen2ForCausalLM):
        raise HeadSkip(f"{type(base).__name__} is not a Qwen2ForCausalLM")
    if not getattr(type(base), "_soup_fused_ce", False):
        raise HeadSkip("the fused head is not patched")
    embedding, lm_head = base.get_input_embeddings(), base.get_output_embeddings()
    if type(embedding) is not torch.nn.Embedding or type(lm_head) is not torch.nn.Linear:
        raise HeadSkip("not a plain nn.Embedding / nn.Linear pair")
    param = embedding.weight
    if lm_head.weight is not param:
        raise HeadSkip("the embedding and the lm_head are not tied")
    if lm_head.bias is not None or embedding.max_norm is not None:
        raise HeadSkip("an lm_head bias or an embedding max_norm")
    if type(param) is not torch.nn.Parameter:
        raise HeadSkip(f"weight of type {type(param).__name__}")
    return param, embedding, lm_head


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _norm_of(model):
    norm = getattr(getattr(_base_model(model), "model", None), "norm", None)
    return norm if isinstance(norm, torch.nn.Module) else None


def load_embedding(model, value, device, log=print, pinned=None):
    """Called by load_stream.py with the input embedding's values, read to the host by the loader
    (`value`, a CPU tensor in the dtype the model loads in): they go to pinned host memory and the
    module gets a placeholder Parameter on `device`, with the lookup and lm_head overrides (the lm_head
    is tied to it when the load ends). Returns (what to set as the parameter, bytes moved); on any
    problem the values themselves on `device`, as the loader would have set them."""
    rows, slots, prefetch_at = _config()
    try:
        embedding, lm_head = model.get_input_embeddings(), model.get_output_embeddings()
        if type(embedding) is not torch.nn.Embedding or type(lm_head) is not torch.nn.Linear:
            raise HeadSkip("not a plain nn.Embedding / nn.Linear pair")
        if tuple(value.shape) != (embedding.num_embeddings, embedding.embedding_dim):
            raise HeadSkip(f"shape {tuple(value.shape)}")
        param = torch.nn.Parameter(torch.empty(0, dtype=value.dtype), requires_grad=value.is_floating_point())
        streamer = HeadStreamer(param, embedding, lm_head, rows=rows, slots=slots, virtual_dtype=value.dtype,
                                prefetch_at=prefetch_at, norm=_norm_of(model), source=value, device=device,
                                pinned=pinned)
    except HeadSkip as exc:
        log(f"[solution] embedding kept on the GPU at load: {exc}")
        return value.to(device), 0
    atexit.register(_report, weakref.ref(streamer), log)
    return param, streamer.host_bytes


def _config():
    rows = _env_int("SOUP_SOLUTION_HEAD_ROWS", DEFAULT_ROWS)
    slots = _env_int("SOUP_SOLUTION_HEAD_SLOTS", DEFAULT_SLOTS)
    prefetch_at = os.environ.get("SOUP_SOLUTION_HEAD_PREFETCH", DEFAULT_PREFETCH).strip().lower()
    return rows, slots, prefetch_at


def _restore_loaded_embedding(model, log=print):
    """A weight the loader put in host memory goes back to the GPU (streaming declined at prepare)."""
    try:
        param = _base_model(model).get_input_embeddings().weight
    except Exception:  # noqa: BLE001 - not a model with embeddings
        return
    streamer = _registered(param)
    if streamer is not None:
        streamer.restore()
        log("[solution] the embedding moved to host memory during the load is back on the GPU")


def stream_before_prepare(model, rows=None, slots=None, log=print):
    """Moves the tied weight of a real model to pinned host memory, with an fp32 placeholder (peft
    is about to cast it); a weight the loader moved already (load_embedding) only gets the fp32
    placeholder. Returns the HeadStreamer, or None with the model as it would be without us."""
    env_rows, env_slots, prefetch_at = _config()
    rows, slots = rows or env_rows, slots or env_slots
    try:
        if not isinstance(model, torch.nn.Module) or not torch.cuda.is_available():
            raise HeadSkip("not a CUDA model")
        if getattr(model, "quantization_method", None) in NOT_CAST or getattr(model, "hqq_quantized", False):
            raise HeadSkip("peft does not cast this model's parameters")
        param, embedding, lm_head = _tied_pair(model)
        loaded = _registered(param)
        if loaded is not None:
            if loaded.embedding is not embedding or loaded.lm_head is not lm_head or not loaded.owns(param):
                raise HeadSkip("the embedding loaded to the host was replaced")
            loaded.set_virtual_dtype(torch.float32)
            log(f"[solution] embedding / lm_head weight ({loaded.vocab} x {loaded.hidden} {loaded.dtype}) "
                f"in pinned host memory since the load: {loaded.host_bytes / 2**20:.0f} MiB; head in chunks "
                f"of {loaded.rows} rows through {loaded.slots} GPU slots, first chunks copied from the "
                f"{loaded.prefetch_at}")
            return loaded
        before = torch.cuda.memory_allocated()
        streamer = HeadStreamer(param, embedding, lm_head, rows=rows, slots=slots, prefetch_at=prefetch_at,
                                norm=_norm_of(model))
    except HeadSkip as exc:
        log(f"[solution] head streaming skipped: {exc}")
        _restore_loaded_embedding(model, log)
        return None
    freed = before - torch.cuda.memory_allocated()
    log(f"[solution] embedding / lm_head weight ({streamer.vocab} x {streamer.hidden} {streamer.dtype}) "
        f"in pinned host memory: {streamer.host_bytes / 2**20:.0f} MiB, {freed / 2**20:.0f} MiB freed on "
        f"the GPU, {streamer.setup_seconds:.2f} s; head in chunks of {streamer.rows} rows through "
        f"{streamer.slots} GPU slots ({streamer.slots * streamer.rows * streamer.hidden * 2 / 2**20:.0f} "
        f"MiB), first chunks copied from the {streamer.prefetch_at}")
    atexit.register(_report, weakref.ref(streamer), log)
    return streamer


def finish_setup(model, log=print):
    """After Soup's setup: keeps streaming when the weight is still the frozen tied weight of the
    model's own embedding and lm_head; otherwise puts it back on the GPU (fp32, the base's state).
    Returns the active HeadStreamer or None."""
    active = None
    for streamer in list(_STREAMERS.values()):
        param = streamer.param_ref()
        if param is None or streamer.host is None or not _contains(model, param):
            continue  # another model's weight
        try:
            found, embedding, lm_head = _tied_pair(model)
            if found is not param:
                reason = "the weight is no longer the model's tied embedding"
            elif embedding is not streamer.embedding or lm_head is not streamer.lm_head:
                reason = "the embedding or the lm_head was replaced"
            elif param.requires_grad:
                reason = "the weight is trainable"
            elif not streamer.owns(param):
                reason = "the weight's data was replaced"
            else:
                reason = None
        except HeadSkip as exc:
            reason = str(exc)
        if reason is None:
            active = streamer
            continue
        streamer.restore()
        log(f"[solution] head streaming undone ({reason}): the weight is back on the GPU")
    return active


def _contains(model, param):
    return isinstance(model, torch.nn.Module) and any(p is param for p in model.parameters())


def _report(ref, log):
    streamer = ref()
    if streamer is None or streamer.host is None or streamer.heads == 0:
        return
    try:
        st = streamer.stats()
    except Exception as exc:  # noqa: BLE001 - CUDA may be gone at exit
        log(f"[solution] head stream: {streamer.heads} heads ({exc!r})")
        return
    copy = (f"{st['copy_ms'] / st['copies_timed']:.2f} ms per chunk at {st['copy_gbps']:.1f} GB/s"
            if st["copies_timed"] and st["copy_gbps"] else "no timed copies")
    log(f"[solution] head stream: {st['heads']} heads, {st['loads']} chunk copies ({st['late_loads']} "
        f"not prefetched), {copy}; the head waited for copies {st['wait_ms'] / st['heads']:.1f} ms per step")


@contextlib.contextmanager
def setup_scope():
    """Marks Soup's setup: peft's prepare_model_for_kbit_training streams the head inside it."""
    _SETUP["depth"] += 1
    try:
        yield
    finally:
        _SETUP["depth"] -= 1


def _warm_kernels(rows):
    from .warmup import HIDDEN, compile_kernel, training_half_dtype

    half, f32 = training_half_dtype(), torch.float32
    block = triton.next_power_of_2(rows)
    # the training launches: stride_l = rows of the chunk (8192 and the last 4480: both divisible by
    # 16), hidden size and its strides 1536, c0 / vc / n_lab not specialized, every pointer aligned
    for first in (True, False):
        compile_kernel(head_chunk_kernel, half, rows, torch.int64, f32, f32, f32, f32, f32, torch.int32,
                       f32, HIDDEN, half, HIDDEN, half, HIDDEN, 0, rows, 387, HIDDEN,
                       FIRST=first, IGNORE=IGNORE_INDEX, BLOCK=block, BLOCK_D=BLOCK_D,
                       num_warps=_num_warps(block))
    compile_kernel(head_finish_kernel, torch.int64, f32, f32, f32, f32, torch.int32, f32, f32, torch.int32,
                   f32, HIDDEN, half, HIDDEN, 1.0, 387, HIDDEN, IGNORE=IGNORE_INDEX, BLOCK_D=BLOCK_D,
                   num_warps=4)


def patch_head_stream(rows=None, slots=None, load=None):
    """Streams the head during Soup's setup: peft's prepare_model_for_kbit_training (called inside
    SFTTrainerWrapper._setup_transformers) first moves the weight to the host, and the end of
    _setup_transformers checks the result (finish_setup). With `load` (default: SOUP_SOLUTION_LOAD_STREAM,
    on) the weight goes to the host while transformers loads it (load_stream.py) and never reaches the
    GPU. Fake modules in Soup's regression tests and calls outside Soup's SFT setup take the original
    paths."""
    import peft
    from soup_cli.trainer import sft as soup_sft

    if load is None:
        load = os.environ.get("SOUP_SOLUTION_LOAD_STREAM", "1") not in ("0", "false", "no")
    if load:
        from .load_stream import patch_loader

        patch_loader(embedding=True)

    rows = rows or _env_int("SOUP_SOLUTION_HEAD_ROWS", DEFAULT_ROWS)
    slots = slots or _env_int("SOUP_SOLUTION_HEAD_SLOTS", DEFAULT_SLOTS)
    cls = soup_sft.SFTTrainerWrapper
    if not getattr(cls, "_soup_head_stream", False):
        original_setup = cls._setup_transformers

        def _setup_transformers(self, *args, **kwargs):
            with setup_scope():
                result = original_setup(self, *args, **kwargs)
            model = getattr(self, "model", None)
            if isinstance(model, torch.nn.Module) and _STREAMERS:
                try:
                    finish_setup(model)
                except Exception as exc:  # noqa: BLE001 - an optimisation must not break setup
                    print(f"[solution] head streaming check failed: {exc!r}")
            return result

        cls._setup_transformers = _setup_transformers
        cls._soup_head_stream = True

    prepare = peft.prepare_model_for_kbit_training
    if not getattr(prepare, "_soup_head_stream", False):

        @functools.wraps(prepare)
        def prepare_model_for_kbit_training(model, *args, **kwargs):
            if _SETUP["depth"] > 0 and isinstance(model, torch.nn.Module):
                try:
                    stream_before_prepare(model, rows, slots)
                except Exception as exc:  # noqa: BLE001
                    print(f"[solution] head streaming skipped: {exc!r}")
                    _restore_loaded_embedding(model)
            return prepare(model, *args, **kwargs)

        prepare_model_for_kbit_training._soup_head_stream = True
        prepare_model_for_kbit_training._soup_original = prepare
        peft.prepare_model_for_kbit_training = prepare_model_for_kbit_training

    from .warmup import register

    register(functools.partial(_warm_kernels, rows), name="head_stream_kernels")
