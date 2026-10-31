"""The streamed output head in int8 (task H1, SOUP_SOLUTION_HEAD_INT8; the organizers' answer 29
allows int8 for the lm_head, forward and backward by the input).

head_stream.py streams the tied fp16 weight W (151,936 x 1,536, 445 MiB) from pinned host memory
through the GPU once per step: 38.9 ms of PCIe at 12 GB/s, plus about 0.58 TFLOP of fp16 GEMMs at
~830 tokens. Here a second host copy holds W in int8 with one scale per vocabulary row,

    Q[v, d] = round(W[v, d] c[d] / s_v[v]),   s_v[v] = max_d |W[v, d] c[d]| / 127,

with c the smoothing vector (below), built once per run from the fp16 copy in a thread (pinning,
the quantization on the GPU chunk by chunk, the kernels' compilation; inside the measured time).
The ring then carries 233 MB per step instead of 445 (each chunk's int8 rows followed by its fp32
scales), and both head GEMMs run on the T4's int8 tensor cores (`torch._int_mm`, cuBLASLt IMMA,
int32 accumulation). The embedding lookup still reads the exact fp16 copy (the answer covers the
lm_head only), and eval and generation (`HeadStreamer.linear`) stay fp16 as well.

Smoothing: the final hidden states h = gamma * x / rms(x) (the decoder's final RMSNorm) have outlier
channels (absmax / rms about 19 per token on the public rows), which per-token int8 rounds badly
(loss +0.5 %, accuracy -0.8 % against the fp16 head on real hidden states). The head quantizes h / c
and the copy holds W * c, c = |gamma|: the same products, the outliers moved into the weight's
per-row scales (loss +0.07 %, accuracy 0.0 %); dL/dh is divided by c at the end.

For N hidden rows (labelled first) quantized per token, h_q[n, d] = round(h[n, d] / (c[d] s_h[n])),
and chunk c of the vocabulary:
    L_c = h_q @ Q_c^T                              int32 (N, R), `_int_mm`, in blocks of FWD_ROWS
                                                   hidden rows (the int32 buffer stays the size of
                                                   the fp16 head's logits)
    head_i8_chunk_kernel (own, one program per row): x = L_c * s_h[n] * s_v[v] in fp32, then the
        online max / sum / entropy / argmax / target logit of head_stream.py; for labelled rows
        u = exp(x - m_new) * s_v[v] quantized per row and chunk, P_q = round(u * 127 / max u)
        (int8), its scale and exp(m_old - m_new) kept for the accumulation; the target's row of
        W_q copied
    Q_c^T: int8_transpose_kernel (own), into the int32 logits' buffer (dead by then): the backward
        contracts over the vocabulary, and IMMA wants both operands contiguous along the contraction
        ("TN"; the row-major chunk as is was 2x slower on the T4); s_v sits on that axis, so it goes
        into P before the quantization and one int8 copy of the weight serves both GEMMs
    G_c = P_q @ Q_c                                int32 (n_lab, D), `_int_mm`, labelled rows only
    head_i8_acc_kernel (own): acc = acc * exp(m_old - m_new) + G_c * scale_P (fp32)
    head_finish_kernel (head_stream.py): loss, entropy, correctness, dL/dh = (acc / s - W_q[t]) * g
The int32 logits never exist for more than one chunk; nothing of shape (N, V) is made.

Numerics: the forward sees W_q and h_q instead of W and h (per-row / per-token absmax int8, round
half up); the softmax statistics are fp32 as before. dL/dh is the gradient of that int8 head's
loss with P quantized per row and chunk (u below 1/254 of its chunk's largest value rounds to 0).
nan and inf are not hidden: a non-finite hidden row gives a non-finite scale, so its logits, loss
and dL/dh are not finite either and GradScaler skips the step as in the base.

Which heads take this path never depends on timing: the first START_HEADS heads of the run (the
kernels compile and the int8 copy is built meanwhile) and any head when the build failed take the
fp16 path; head START_HEADS + 1 waits for the thread if it is not done. Switches (environment):
SOUP_SOLUTION_HEAD_INT8 (on by default in the `soup train` process only, as int8_bwd.py; pytest
keeps the fp16 head), SOUP_SOLUTION_HEAD_INT8_START (START_HEADS), SOUP_SOLUTION_HEAD_INT8_AT
(where the build thread starts: "setup", the end of Soup's setup, or "head", the run's first head),
SOUP_SOLUTION_HEAD_INT8_SMOOTH / _ALPHA (smoothing()), SOUP_SOLUTION_HEAD_INT8_FWD_ROWS,
SOUP_SOLUTION_HEAD_INT8_WARPS.
"""

import atexit
import os
import threading
import time
import weakref

import torch
import triton
import triton.language as tl

from .fused_ce import IGNORE_INDEX
from .head_stream import BLOCK_D, head_finish_kernel

START_HEADS = int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_START", "4"))
BUILD_AT = os.environ.get("SOUP_SOLUTION_HEAD_INT8_AT", "setup").strip().lower()
# int8_transpose_kernel's square tile and warps: T4, one head's 38 transposes, 2.47 ms against 2.77 for 64 x 64
# with 4 warps (soup-h1-kern1)
T_TILE, T_WARPS = 128, 8
QUANT_ROWS = 1024  # vocabulary rows per quantization step of the build (transient GPU memory ~10 MiB)
ACC_BLOCK = 512
# warps of the chunk kernel: T4, one head at 830 tokens, 16 warps 5.8-6.0 ms against 8.5-9.8 for 8 and 8.0-9.5
# for 4 (soup-h1-kern1; head_stream's fp16 kernel is fastest with 8); 2-8 rows per program were slower
WARPS = int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_WARPS", "16"))
# hidden rows per forward GEMM: a chunk's int32 logits are twice the fp16 head's per row; in blocks of
# 512 rows the int8 head's memory stays 2-7 MiB below the fp16 head's at 512-2048 tokens (soup-h1-bench1)
FWD_ROWS = max(int(os.environ.get("SOUP_SOLUTION_HEAD_INT8_FWD_ROWS", "512")) // 8 * 8, 24)


def smoothing(streamer):
    """The smoothing vector for this streamer's copy (SOUP_SOLUTION_HEAD_INT8_SMOOTH): "gamma" (default)
    takes |gamma|^alpha of the decoder's final RMSNorm weight (frozen in this task; any positive vector
    keeps the products exact, so a later change of it would only change the rounding), alpha from
    SOUP_SOLUTION_HEAD_INT8_ALPHA (1: the norm weight folded into W whole: on real final hidden
    states of the public rows loss +0.07 % and accuracy +0.0 % against the fp16 head, against +0.48 % and -0.8 %
    unsmoothed and +0.40 % with alpha 0.5); "none": no smoothing."""
    mode = os.environ.get("SOUP_SOLUTION_HEAD_INT8_SMOOTH", "gamma").strip().lower()
    weight = getattr(getattr(streamer, "norm", None), "weight", None)
    if mode != "gamma" or not isinstance(weight, torch.Tensor) or weight.numel() != streamer.hidden:
        return None
    try:
        alpha = float(os.environ.get("SOUP_SOLUTION_HEAD_INT8_ALPHA", "1"))
    except ValueError:
        alpha = 1.0
    c = weight.detach().float().abs().clamp_min(1e-3) ** alpha
    return c / c.mean()


def enabled():
    from .int8_bwd import _soup_train_process

    default = "1" if _soup_train_process() else "0"
    return os.environ.get("SOUP_SOLUTION_HEAD_INT8", default).strip().lower() not in ("0", "false", "no", "")


@triton.jit(do_not_specialize=["row0", "first", "c0", "vc", "n_lab"])
def head_i8_chunk_kernel(
    l32_ptr, stride_l, row0,  # int32 (block rows, vc): this chunk's logits for rows row0 + i
    pq_ptr, stride_pq,  # int8 (n_lab, vc): P quantized, labelled rows
    sh_ptr, sv_ptr,  # fp32 (rows,) the tokens' scales, (vc,) the chunk's vocabulary row scales
    targets_ptr,  # int64 (rows,): target token id or IGNORE
    m_ptr, s_ptr, e_ptr, best_ptr, xt_ptr,  # fp32 (rows,): running max, sum, entropy sum, best, x_t
    idx_ptr,  # int32 (rows,): running argmax
    alpha_ptr, sp_ptr,  # fp32 (n_lab,): exp(m_old - m_new) and P's scale for this chunk
    q_ptr, stride_q,  # int8 (vc, D): the chunk of Q on the GPU
    wt_ptr, stride_wt,  # half (n_lab, D): W_q[target], written by the chunk that holds the target
    first, c0, vc, n_lab, D,  # first: 1 for the vocabulary's first chunk (a runtime flag: one compile)
    IGNORE: tl.constexpr, BLOCK: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = row0 + tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < vc
    base = l32_ptr + tl.program_id(0).to(tl.int64) * stride_l
    sh = tl.load(sh_ptr + row)
    sv = tl.load(sv_ptr + offs, mask=mask, other=0.0)
    xi = tl.load(base + offs, mask=mask, other=0)
    target = tl.load(targets_ptr + row)
    t_local = target - c0
    hit = (target != IGNORE) & (t_local >= 0) & (t_local < vc)
    t_safe = tl.where(hit, t_local, 0)
    # the target's logit and scale (one load each, as head_stream.head_chunk_kernel)
    sv_t = tl.load(sv_ptr + t_safe, mask=hit, other=0.0)
    x_hit = tl.load(base + t_safe, mask=hit, other=0).to(tl.float32) * (sh * sv_t)
    x = tl.where(mask, xi.to(tl.float32) * (sh * sv), -float("inf"))
    c_max, c_idx = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
    m_old = -float("inf")
    s = 0.0
    e = 0.0
    best = -float("inf")
    best_idx = 0
    x_t = 0.0
    if first == 0:
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
        u = p * sv  # s_v on the contraction axis of the backward GEMM goes into P
        u_max = tl.max(u, axis=0)
        inv = tl.where(u_max > 0.0, 127.0 / u_max, 0.0)
        pq = tl.minimum(tl.floor(u * inv + 0.5), 127.0).to(tl.int8)
        tl.store(pq_ptr + row.to(tl.int64) * stride_pq + offs, pq, mask=mask)
        tl.store(sp_ptr + row, u_max / 127.0)
        tl.store(alpha_ptr + row, alpha)
        if hit:
            q_row = q_ptr + t_local * stride_q
            wt_row = wt_ptr + row.to(tl.int64) * stride_wt
            for d0 in range(0, D, BLOCK_D):
                od = d0 + tl.arange(0, BLOCK_D)
                md = od < D
                w = tl.load(q_row + od, mask=md, other=0).to(tl.float32) * sv_t
                tl.store(wt_row + od, w.to(wt_ptr.dtype.element_ty), mask=md)


@triton.jit
def int8_transpose_kernel(src_ptr, dst_ptr, n_rows, n_cols, T: tl.constexpr):
    # src (n_rows, n_cols) row-major -> dst (n_cols, n_rows) row-major, one T x T tile per program
    r = tl.program_id(0) * T + tl.arange(0, T)
    c = tl.program_id(1) * T + tl.arange(0, T)
    live = (r < n_rows)[:, None] & (c < n_cols)[None, :]
    x = tl.load(src_ptr + r[:, None].to(tl.int64) * n_cols + c[None, :], mask=live, other=0)
    tl.store(dst_ptr + c[:, None].to(tl.int64) * n_rows + r[None, :], tl.trans(x), mask=tl.trans(live))


@triton.jit(do_not_specialize=["first"])
def head_i8_acc_kernel(acc_ptr, stride_acc, g_ptr, stride_g, alpha_ptr, sp_ptr, first, D, BLOCK: tl.constexpr):
    # acc[row] = acc[row] * exp(m_old - m_new) + G[row] * scale_P (fp32), labelled rows only
    row = tl.program_id(0)
    od = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    md = od < D
    g = tl.load(g_ptr + row.to(tl.int64) * stride_g + od, mask=md, other=0).to(tl.float32)
    g = g * tl.load(sp_ptr + row)
    at = acc_ptr + row.to(tl.int64) * stride_acc + od
    if first == 0:
        g = tl.load(at, mask=md, other=0.0) * tl.load(alpha_ptr + row) + g
    tl.store(at, g, mask=md)


def _pad_rows(n):
    """Rows of an `_int_mm` operand: more than 16 and a multiple of 8."""
    return max(-(-n // 8) * 8, 24)


def quant_rows(x, out, col=None):
    """Per-row absmax int8 of x (n, D), times `col` (D,) per column when given, into out[:n] (rows
    past n are left as they are): returns the fp32 scales absmax / 127. A zero row gets scale 0; a
    non-finite row a non-finite scale."""
    xf = x.float() if col is None else x.float() * col
    amax = xf.abs().amax(dim=1)
    inv = torch.where(amax > 0, 127.0 / amax.clamp_min(1e-30), torch.zeros_like(amax))
    q = torch.floor(xf.mul_(inv[:, None]).add_(0.5)).clamp_(-127, 127)
    out[:x.shape[0]].copy_(q)
    return amax / 127.0


class HeadSkipInt8(Exception):
    """This streamer's weight does not get an int8 copy."""


class HeadInt8:
    """The int8 form of a HeadStreamer's weight in pinned host memory (built by `start()` in a
    thread) and the head that streams it. Chunk c of the host buffer (and of a ring slot): its
    vc x D int8 rows, then its vc fp32 scales."""

    def __init__(self, streamer, log=print, smooth=None):
        """`smooth`: a positive (D,) vector c: the copy holds W * c (per hidden channel) and the head
        quantizes h / c, which moves the hidden states' outlier channels into the weight (the
        products are the same; only the rounding changes). None: c = 1."""
        if streamer.hidden % 64 or streamer.rows % 64 or streamer.vocab % 8:
            raise HeadSkipInt8(f"shape {streamer.vocab} x {streamer.hidden}, {streamer.rows} rows per chunk")
        self.rows, self.hidden, self.vocab = streamer.rows, streamer.hidden, streamer.vocab
        self.n_chunks = streamer.n_chunks
        device = torch.device(streamer.device)
        self.device = device if device.index is not None else torch.device("cuda", torch.cuda.current_device())
        self.slot_bytes = self.rows * (self.hidden + 4)
        self.block = triton.next_power_of_2(self.rows)
        from .head_stream import _num_warps

        self.num_warps = WARPS if WARPS in (1, 2, 4, 8, 16, 32) else _num_warps(self.block)
        self.ready, self.error, self.seconds, self.times = False, None, 0.0, {}
        self.calls = 0  # heads that asked (take())
        self.log = log
        self._source = streamer.host  # the fp16 host copy
        self._stop = threading.Event()
        self._thread = None
        self.host = self._unpin = None
        self._deferred = False
        self.total_bytes = self.vocab * (self.hidden + 4)
        self.col = self.inv_col = None
        if smooth is not None:
            c = smooth.detach().float().to(self.device).reshape(-1)
            if c.numel() != self.hidden or not bool(torch.isfinite(c).all()) or not bool((c > 0).all()):
                raise HeadSkipInt8("the smoothing vector is not a positive (D,) vector")
            self.col, self.inv_col = c, 1.0 / c

    def chunk(self, c):
        """(first row, rows, first byte, bytes) of chunk c."""
        c0 = c * self.rows
        vc = min(self.rows, self.vocab - c0)
        return c0, vc, c * self.slot_bytes, vc * (self.hidden + 4)

    def host_chunk(self, c):
        _, _, b0, nb = self.chunk(c)
        return self.host[b0:b0 + nb]

    # ---- the build ----------------------------------------------------------------------------

    def start(self, defer=False):
        """Starts the build thread now, or (`defer`) at the first take()."""
        self._deferred = defer
        if not defer:
            self._start_thread()
        return self

    def _start_thread(self):
        self._deferred = False
        self._thread = threading.Thread(target=self._build_safely, name="soup-head-int8", daemon=True)
        self._thread.start()
        atexit.register(_join_at_exit, weakref.ref(self))

    def join(self):
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def stop(self):
        self._stop.set()
        self.join()
        self.ready = False
        if self._unpin is not None:
            self._unpin()
            self._unpin = None
        self.host = None

    def _build_safely(self):
        started = time.perf_counter()
        try:
            self.build()
            began = time.perf_counter()
            warm_kernels(self)
            self.times["compile"] = time.perf_counter() - began
            self.ready = True
        except Exception as exc:  # noqa: BLE001 - an optimisation must not break the run
            self.error = f"{type(exc).__name__}: {exc}"
        self.seconds = time.perf_counter() - started
        if self.ready:
            t = self.times
            self.log(f"[solution] head int8: {self.vocab} x {self.hidden} in {self.n_chunks} chunks of "
                     f"{self.rows} rows ({self.total_bytes / 2**20:.0f} MiB pinned, per-row scales, smoothing "
                     f"{'on' if self.col is not None else 'off'}) in {self.seconds:.2f} s (pinning "
                     f"{t.get('pin', 0):.2f}, quantization {t.get('quant', 0):.2f}, compilation "
                     f"{t.get('compile', 0):.2f}), ready after {self.calls} heads; heads from number "
                     f"{START_HEADS + 1} on")
        elif not self._stop.is_set():
            self.log(f"[solution] head int8 off: {self.error}")

    def build(self):
        from .weight_stream import pinned_host_buffer

        torch.cuda.set_device(self.device)
        began = time.perf_counter()
        host, unpin = pinned_host_buffer(self.total_bytes, self)
        self.times["pin"] = time.perf_counter() - began
        began = time.perf_counter()
        dev = self.device
        # the default stream: the blocks allocated here go back to the pool training allocates from
        stream = torch.cuda.default_stream(dev)
        with torch.cuda.stream(stream), torch.no_grad():
            for c in range(self.n_chunks):
                if self._stop.is_set():
                    unpin()
                    raise HeadSkipInt8("stopped")
                c0, vc, b0, nb = self.chunk(c)
                out = torch.empty(nb, dtype=torch.uint8, device=dev)
                q = out[:vc * self.hidden].view(torch.int8).view(vc, self.hidden)
                scales = out[vc * self.hidden:].view(torch.float32)
                for r0 in range(0, vc, QUANT_ROWS):
                    r1 = min(r0 + QUANT_ROWS, vc)
                    w = self._source[c0 + r0:c0 + r1].to(dev, non_blocking=True)
                    scales[r0:r1] = quant_rows(w, q[r0:r1], self.col)
                if not bool(torch.isfinite(scales).all()):
                    unpin()
                    raise HeadSkipInt8(f"chunk {c}: a non-finite weight row")
                host[b0:b0 + nb].copy_(out)  # synchronous: the pinned bytes are final after it
            stream.synchronize()
        self.times["quant"] = time.perf_counter() - began
        self.host, self._unpin = host, unpin

    # ---- which heads ---------------------------------------------------------------------------

    def take(self):
        """Whether the head being prefetched streams the int8 copy (asked once per head)."""
        self.calls += 1
        if self._thread is None and self._deferred and not self._stop.is_set():
            self._start_thread()  # SOUP_SOLUTION_HEAD_INT8_AT=head
        if self.calls <= START_HEADS or self.error is not None:
            return False
        if not self.ready:
            self.join()
        return self.ready

    # ---- the head ------------------------------------------------------------------------------

    def head(self, st, hidden16, targets, n_lab, grad_scale):
        """HeadStreamer.head with this int8 copy in the ring (st.prefetch() chose it)."""
        n, d = hidden16.shape
        dev = hidden16.device
        loss = torch.empty(n, dtype=torch.float32, device=dev)
        entropy = torch.empty(n, dtype=torch.float32, device=dev)
        correct = torch.empty(n, dtype=torch.int32, device=dev)
        acc = torch.empty(max(n_lab, 1), d, dtype=torch.float32, device=dev)
        compute = st._compute()
        if n == 0:
            self._finish(st, compute)
            return loss, entropy, correct, acc.zero_()
        with torch.autocast("cuda", enabled=False):
            self._run(st, compute, hidden16, targets, n_lab, grad_scale, loss, entropy, correct, acc)
        self._finish(st, compute)
        return loss, entropy, correct, acc

    def _run(self, st, compute, hidden16, targets, n_lab, grad_scale, loss, entropy, correct, acc):
        n, d = hidden16.shape
        dev = hidden16.device
        blocks = [(b0, min(FWD_ROWS, n - b0)) for b0 in range(0, n, FWD_ROWS)]
        n_pad = max(b0 + _pad_rows(nb) for b0, nb in blocks)  # every block's GEMM rows: > 16, multiple of 8
        h_q = torch.empty(n_pad, d, dtype=torch.int8, device=dev)
        s_h = quant_rows(hidden16, h_q, self.inv_col)
        stride = -(-n // 4) * 4  # 16-byte aligned rows of the state, whatever n
        state = torch.empty(5, stride, dtype=torch.float32, device=dev)
        m, s, e, best, x_t = state
        idx = torch.empty(n, dtype=torch.int32, device=dev)
        lab = max(n_lab, 1)
        wt = torch.empty(lab, d, dtype=hidden16.dtype, device=dev)
        alpha_sp = torch.empty(2, -(-lab // 4) * 4, dtype=torch.float32, device=dev)
        alpha, sp = alpha_sp[0], alpha_sp[1]
        rows, vocab = self.rows, self.vocab
        # the int32 logits of a chunk, then (as int8) the chunk's transpose for the backward GEMM
        l_rows = max(_pad_rows(nb) for _, nb in blocks)
        l32 = torch.empty(max(l_rows * rows, -(-d * rows // 4)), dtype=torch.int32, device=dev)
        lab_pad = _pad_rows(n_lab)
        g32 = torch.empty(lab_pad, d, dtype=torch.int32, device=dev) if n_lab else None
        pq_buf = torch.empty(lab_pad * rows, dtype=torch.int8, device=dev)
        q_t = l32.view(torch.int8)
        views = {}  # chunk rows -> (int32 logits, P_q): two sizes at most
        for c in range(self.n_chunks):
            c0 = c * rows
            vc = min(rows, vocab - c0)
            slot = st._acquire(c, compute)
            q = slot[:vc * d].view(torch.int8).view(vc, d)
            sv = slot[vc * d:vc * (d + 4)].view(torch.float32)
            chunk = views.get(vc)
            if chunk is None:
                chunk = views[vc] = ([(h_q[b0:b0 + _pad_rows(nb)], l32[:_pad_rows(nb) * vc].view(-1, vc))
                                      for b0, nb in blocks], pq_buf[:lab_pad * vc].view(lab_pad, vc))
            gemms, p_q = chunk
            for (b0, nb), (h_b, logits) in zip(blocks, gemms):
                torch._int_mm(h_b, q.t(), out=logits)
                head_i8_chunk_kernel[(nb,)](
                    logits, vc, b0, p_q, vc, s_h, sv, targets, m, s, e, best, x_t, idx, alpha, sp,
                    q, d, wt, wt.stride(0), int(c == 0), c0, vc, n_lab, d,
                    IGNORE=IGNORE_INDEX, BLOCK=self.block, BLOCK_D=BLOCK_D, num_warps=self.num_warps,
                )
            if n_lab:
                q_cols = q_t[:d * vc].view(d, vc)
                int8_transpose_kernel[(triton.cdiv(vc, T_TILE), triton.cdiv(d, T_TILE))](
                    q, q_cols, vc, d, T=T_TILE, num_warps=T_WARPS)
                st._release(c, compute)  # the slot is read no more: its next chunk may come
                torch._int_mm(p_q, q_cols.t(), out=g32)
                head_i8_acc_kernel[(n_lab, triton.cdiv(d, ACC_BLOCK))](
                    acc, acc.stride(0), g32, g32.stride(0), alpha, sp, int(c == 0), d,
                    BLOCK=ACC_BLOCK, num_warps=4)
            else:
                st._release(c, compute)
        head_finish_kernel[(n,)](
            targets, m, s, e, x_t, idx, loss, entropy, correct,
            acc, acc.stride(0), wt, wt.stride(0), grad_scale, n_lab, d,
            IGNORE=IGNORE_INDEX, BLOCK_D=BLOCK_D, num_warps=4,
        )
        if n_lab == 0:
            acc.zero_()
        elif self.inv_col is not None:
            acc[:n_lab].mul_(self.inv_col)  # dL/dh = sum_v g_v (W_v * c) / c

    def _finish(self, st, compute):
        # every copy into the ring has been waited for by the compute stream (st._acquire): freeing
        # it on the compute stream needs no record_stream
        for c in range(st.n_chunks):
            if st._owner[c % st.slots] == c:
                compute.wait_event(st._ready[c % st.slots])
        st._ring, st._owner, st._slot_views, st._ring_pack = None, [-1] * st.slots, None, None
        st.heads += 1
        st.int8_heads += 1
        if st._spans:
            st._harvest()


def _join_at_exit(ref):
    obj = ref()
    if obj is not None:
        obj._stop.set()
        obj.join()


def warm_kernels(obj=None):
    """Compiles the three kernels for the classes the head launches (in the build thread)."""
    from .warmup import HIDDEN, compile_kernel, training_half_dtype

    half, f32, i8, i32, i64 = training_half_dtype(), torch.float32, torch.int8, torch.int32, torch.int64
    rows = obj.rows if obj is not None else 4096
    hidden = obj.hidden if obj is not None else HIDDEN
    warps = obj.num_warps if obj is not None else WARPS
    # stride_l = rows of the chunk (4096 and the last 384: divisible by 16), the hidden size 1536;
    # row0, first, c0, vc, n_lab not specialized
    compile_kernel(head_i8_chunk_kernel, i32, rows, 0, i8, rows, f32, f32, i64, f32, f32, f32, f32, f32, i32,
                   f32, f32, i8, hidden, half, hidden, 1, 0, rows, 387, hidden,
                   IGNORE=IGNORE_INDEX, BLOCK=triton.next_power_of_2(rows), BLOCK_D=BLOCK_D, num_warps=warps)
    compile_kernel(head_i8_acc_kernel, f32, hidden, i32, hidden, f32, f32, 1, hidden, BLOCK=ACC_BLOCK,
                   num_warps=4)
    compile_kernel(int8_transpose_kernel, i8, i8, rows, hidden, T=T_TILE, num_warps=T_WARPS)
