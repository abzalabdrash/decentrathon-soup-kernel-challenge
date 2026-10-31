"""The MLP's (gate|up) crosses PCIe as int8 (task P1, SOUP_SOLUTION_OFFLOAD_INT8, on by default in the
`soup train` process only, like int8_bwd.py; =1 also under pytest, =0 off everywhere). Organizers'
answer 43: a tensor saved only for the backward may be kept in int8 with a scale (lossy) when the
forward, the loss and the metrics do not change and all 10 checks pass.

offload.py sends every layer's (gate|up) (2I, M) half, 36 KB per token, to pinned host memory in the
forward and brings it back in the backward: 0.83 GB each way per step at ~830 tokens, 2 GB at the
jury's 2,048-token steps, on a PCIe link the NF4 weights, the head and the layer inputs share. Here
the early offload of the fused MLP (offload.offload_early) first quantizes it on the compute stream
(`gu_quant_kernel`): one int8 per element and one fp32 scale per token and group of GROUP features
(absmax / 127, round to nearest; a group never straddles gate and up), both in one uint8 buffer of
~51.6 % of the half tensor's bytes. That buffer is what goes to the host and comes back; the unpack
rebuilds the half (2I, M) tensor the MLP's backward expects (`gu_dequant_kernel`), so fused_proj.py
and int8_bwd.py read it as before. The half (gate|up) itself is no longer held by the offload while its
copy runs (only the smaller buffer is), so it is freed when the MLP's forward returns.

Numerics: the forward, the loss and the eval are untouched; the SwiGLU derivative in the backward
reads (gate|up) with an error of at most half a quantization step of its group (relative error of
the LoRA gradients against the fp16 offload: bench_offload_int8.py, test_offload_int8.py).

The two kernels are compiled by a thread started at the first early offload (the first START_CALLS
offloads stay half meanwhile, four steps of 28 layers); the next one waits for the thread if it is not
done, so which offloads go int8 never depends on timing (int8_bwd.py does the same).
"""

import atexit
import os
import threading

import torch
import triton
import triton.language as tl

from .int8_bwd import _soup_train_process

_STATE = {"on": os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8", "1" if _soup_train_process() else "0")
          .strip().lower() not in ("0", "false", "no", ""), "thread": None, "calls": 0}
START_CALLS = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_START_CALLS", "112"))
# features per scale (64, 128 or 256 divide Qwen2.5-1.5B's 8960); tokens per program
GROUP = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_GROUP", "128"))
TILE_M = 64 if GROUP <= 128 else 32
_READY = threading.Event()
HALF_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def gu_quant_kernel(x_ptr, q_ptr, s_ptr, tokens, GROUP: tl.constexpr, TM: tl.constexpr):
    # x (rows, tokens) half -> q (rows, tokens) int8 and s (rows // GROUP, tokens) fp32: program (g, t)
    # takes rows [g*GROUP, +GROUP) x tokens [t*TM, +TM); each token's absmax over the group's rows
    # gives its scale amax / 127, the values round to nearest (a zero column: scale 0, zeros)
    grp = tl.program_id(0)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    rows = grp * GROUP + tl.arange(0, GROUP)
    at = rows[:, None].to(tl.int64) * tokens + tok[None, :]
    x = tl.load(x_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    inv = tl.where(amax > 0.0, 127.0 / amax, 0.0)
    q = tl.minimum(tl.maximum(tl.floor(x * inv[None, :] + 0.5), -127.0), 127.0).to(tl.int8)
    tl.store(q_ptr + at, q, mask=live[None, :])
    tl.store(s_ptr + grp * tokens + tok, amax / 127.0, mask=live)


@triton.jit
def gu_dequant_kernel(q_ptr, s_ptr, out_ptr, tokens, GROUP: tl.constexpr, TM: tl.constexpr):
    # the inverse of gu_quant_kernel: out[r, t] = q[r, t] * s[r // GROUP, t] in out's half dtype
    grp = tl.program_id(0)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    rows = grp * GROUP + tl.arange(0, GROUP)
    at = rows[:, None].to(tl.int64) * tokens + tok[None, :]
    q = tl.load(q_ptr + at, mask=live[None, :], other=0).to(tl.float32)
    s = tl.load(s_ptr + grp * tokens + tok, mask=live, other=0.0)
    tl.store(out_ptr + at, (q * s[None, :]).to(out_ptr.dtype.element_ty), mask=live[None, :])


# ---------------------------------------------------------------------------------------------


def _scales_offset(rows, tokens):
    return -(-rows * tokens // 16) * 16


def wire_bytes(rows, tokens):
    """Bytes of the int8 form of a (rows, tokens) half tensor: the int8 values, then the scales."""
    return _scales_offset(rows, tokens) + 4 * (rows // GROUP) * tokens


def views(wire, rows, tokens):
    """(q (rows, tokens) int8, s (rows // GROUP, tokens) fp32) inside a uint8 buffer of wire_bytes."""
    off = _scales_offset(rows, tokens)
    q = wire[:rows * tokens].view(torch.int8).view(rows, tokens)
    s = wire[off:off + 4 * (rows // GROUP) * tokens].view(torch.float32).view(rows // GROUP, tokens)
    return q, s


def _grid(rows, tokens):
    return (rows // GROUP, triton.cdiv(tokens, TILE_M))


class Codec:
    """What the offload keeps to rebuild one tensor: its shape and dtype (the int8 buffer itself is
    the offload's GPU or host copy)."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def decode(self, wire):
        """The half (rows, tokens) tensor from its int8 buffer on the GPU, on the current stream."""
        rows, tokens = self.shape
        q, s = views(wire, rows, tokens)
        out = torch.empty(self.shape, dtype=self.dtype, device=wire.device)
        gu_dequant_kernel[_grid(rows, tokens)](q, s, out, tokens, GROUP=GROUP, TM=TILE_M, num_warps=4)
        return out


def encode(tensor):
    """(uint8 buffer of wire_bytes on the tensor's device, Codec), the quantization queued on the
    current stream."""
    rows, tokens = tensor.shape
    wire = torch.empty(wire_bytes(rows, tokens), dtype=torch.uint8, device=tensor.device)
    q, s = views(wire, rows, tokens)
    gu_quant_kernel[_grid(rows, tokens)](tensor, q, s, tokens, GROUP=GROUP, TM=TILE_M, num_warps=4)
    return wire, Codec(tensor.shape, tensor.dtype)


def applies(tensor):
    """Whether this early offload goes int8: switched on, a contiguous 2-D half CUDA tensor whose rows
    split into groups, and START_CALLS such offloads ran before (the kernels compile meanwhile)."""
    if not _STATE["on"]:
        return False
    if (tensor.dtype not in HALF_DTYPES or tensor.dim() != 2 or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.shape[0] % GROUP or tensor.shape[1] < 1):
        return False
    _STATE["calls"] += 1
    if not _READY.is_set():
        _start_compile()
    if _STATE["calls"] <= START_CALLS:
        return False  # half for the first START_CALLS, compiled or not: the switch point is fixed
    if not _READY.is_set():
        _STATE["thread"].join()
    return _READY.is_set()


def _start_compile():
    if _STATE["thread"] is not None:
        return

    def work():
        try:
            if _warm() is None:
                _READY.set()
        except Exception as exc:  # noqa: BLE001 - the offload stays half
            print(f"[solution] int8 offload kernels not compiled ({type(exc).__name__}: {exc}); "
                  "the half offload stays", flush=True)

    thread = threading.Thread(target=work, name="soup-offload-int8-compile", daemon=True)
    _STATE["thread"] = thread
    thread.start()
    atexit.register(thread.join, 120.0)  # never leave the interpreter while it compiles


def compile_now():
    """Compiles the kernels in the calling thread, so that the next eligible offload goes int8
    (tests and benchmarks)."""
    if not _READY.is_set() and _warm() is None:
        _READY.set()
    _STATE["calls"] = max(_STATE["calls"], START_CALLS)


def _warm():
    """The two kernels in the classes training launches (tokens a multiple of 16: unpad.py's tail)."""
    from .warmup import SKIPPED, TOKENS, compile_kernel, training_half_dtype

    if not _STATE["on"]:
        return SKIPPED
    tokens = -(-TOKENS // 16) * 16
    half = training_half_dtype()
    compile_kernel(gu_quant_kernel, half, torch.int8, torch.float32, tokens, GROUP=GROUP, TM=TILE_M,
                   num_warps=4)
    compile_kernel(gu_dequant_kernel, torch.int8, torch.float32, half, tokens, GROUP=GROUP, TM=TILE_M,
                   num_warps=4)
    return None
