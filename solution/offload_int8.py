"""The MLP's (gate|up) crosses PCIe as int8 (task P1, SOUP_SOLUTION_OFFLOAD_INT8, on by default in every
process like int8_bwd.py, =0 off; solution/conftest.py switches it off for the bitwise tests of the half
path, test_offload_int8.py switches it on). Organizers' answer 43: a tensor saved only for the backward may be kept in int8 with a scale (lossy) when the
forward, the loss and the metrics do not change and all 10 checks pass.

offload.py sends every layer's (gate|up) (2I, M) half, 36 KB per token, to pinned host memory in the
forward and brings it back in the backward: 0.83 GB each way per step at ~830 tokens, 2 GB at the
2,048-token steps of long rows, on a PCIe link the NF4 weights, the head and the layer inputs share.

The int8 form: for every element (i, t) of the gate half, one int16 whose low byte is gate[i, t] and
high byte up[i, t] in int8, and one fp32 scale per token and group of GROUP features of each half
(absmax / 127, round to nearest). That is 2 I M bytes of values (the size of the gate half) plus the
scales, 51.6 % of the half tensor. `gu_quant_kernel` with SWIGLU computes the MLP's SwiGLU forward
exactly as the example kernel does (gate * sigmoid(gate) * up in fp32, one rounding: the same hidden
activation bit for bit) and in the same pass writes the int8 form in place, over the gate half of
(gate|up) (each element's two bytes, read before by the same program only): no extra read of (gate|up)
and no new buffer of its size, only the scales. fused_proj.py's MLP forward launches it instead of the
example SwiGLU when offload.offload_early says the tensor goes int8, then the D2H of those 2 I M bytes
and the scales starts at once (half the bytes of the half offload's copy). The unpack rebuilds the half
(2I, M) tensor the backward expects (`gu_dequant_kernel`), or, inside raw(True), hands the int8 form
to int8_bwd.py's SwiGLU backward, which reads it directly.

Numerics: the forward, the loss and the eval are untouched; the SwiGLU derivative in the backward reads
(gate|up) with an error of at most half a quantization step of its group (relative error of the LoRA
gradients against the half offload: bench_offload_int8.py, test_offload_int8.py).

The two kernels are compiled by a thread started at the first MLP backward of the second step (raw());
the first START_CALLS offloads stay half (seven steps of 28 layers), the next one waits for the thread if
it is not done, so which offloads go int8 never depends on timing (int8_bwd.py does the same).
"""

import atexit
import contextlib
import os
import threading

import torch
import triton
import triton.language as tl

_STATE = {"on": os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8", "1").strip().lower()
          not in ("0", "false", "no", ""), "thread": None, "calls": 0, "backwards": 0}
# seven steps of 28 layers: soup-p1-abba-jury3 switched at step 4 and waited 0.4 s there for the thread
START_CALLS = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_START_CALLS", "196"))
# MLP backwards before the compile thread starts: not during the first step (soup-p1-abba-pub2: a
# thread started there put step 0 +0.13 s later); six steps are left for it before START_CALLS
COMPILE_AFTER = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_COMPILE_AFTER", "28"))
# features per scale (64, 128 or 256 divide Qwen2.5-1.5B's 8960); tokens per program
GROUP = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_GROUP", "128"))
# T4 (soup-p1-bench2, quantization alone): 128 tokens x 4 or 8 warps 0.28 ms at 832 tokens, 0.55 at 2,048;
# 64 x 4 0.32 / 0.75; 32 x 4 0.49 / 1.22
TILE_M = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_TILE_M", "128" if GROUP <= 128 else "64"))
WARPS = int(os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_WARPS", "8"))
# SOUP_SOLUTION_OFFLOAD_INT8_FUSED=1: the int8 MLP backward (int8_bwd.py) reads the int8 form itself
# inside its SwiGLU backward (raw), no rebuild pass; "now" while fused_proj's backward unpacks for it
RAW = {"on": os.environ.get("SOUP_SOLUTION_OFFLOAD_INT8_FUSED", "1").strip().lower()
       not in ("0", "false", "no", ""), "now": False}
_READY = threading.Event()
HALF_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def gu_quant_kernel(gu_ptr, hidden_ptr, q_ptr, s_ptr, inter, tokens, GROUP: tl.constexpr,
                    TM: tl.constexpr, SWIGLU: tl.constexpr):
    # gu (2I, tokens) half -> q (I, tokens) int16 (low byte the gate's int8, high byte the up's) and
    # s (2I // GROUP, tokens) fp32: program (g, t) takes features [g*GROUP, +GROUP) of both halves x
    # tokens [t*TM, +TM); each token's absmax over a group gives its scale amax / 127, values round to
    # nearest (a zero column: scale 0, zeros). SWIGLU: also hidden = gate * sigmoid(gate) * up as
    # solution/swiglu.py's forward computes it. q may be the gate half itself (in place: every element
    # is stored after all of this program's loads, and no other program touches it)
    grp = tl.program_id(0)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    rows = grp * GROUP + tl.arange(0, GROUP)
    at = rows[:, None].to(tl.int64) * tokens + tok[None, :]
    gate = tl.load(gu_ptr + at, mask=live[None, :], other=0.0).to(tl.float32)
    up = tl.load(gu_ptr + inter.to(tl.int64) * tokens + at, mask=live[None, :], other=0.0).to(tl.float32)
    if SWIGLU:
        out = gate * tl.sigmoid(gate) * up
        tl.store(hidden_ptr + at, out.to(hidden_ptr.dtype.element_ty), mask=live[None, :])
    amax_g = tl.max(tl.abs(gate), axis=0)
    amax_u = tl.max(tl.abs(up), axis=0)
    inv_g = tl.where(amax_g > 0.0, 127.0 / amax_g, 0.0)
    inv_u = tl.where(amax_u > 0.0, 127.0 / amax_u, 0.0)
    q_g = tl.minimum(tl.maximum(tl.floor(gate * inv_g[None, :] + 0.5), -127.0), 127.0).to(tl.int32)
    q_u = tl.minimum(tl.maximum(tl.floor(up * inv_u[None, :] + 0.5), -127.0), 127.0).to(tl.int32)
    tl.store(q_ptr + at, ((q_u << 8) | (q_g & 255)).to(tl.int16), mask=live[None, :])
    tl.store(s_ptr + grp * tokens + tok, amax_g / 127.0, mask=live)
    tl.store(s_ptr + (inter // GROUP + grp) * tokens + tok, amax_u / 127.0, mask=live)


@triton.jit
def gu_dequant_kernel(q_ptr, s_ptr, out_ptr, inter, tokens, GROUP: tl.constexpr, TM: tl.constexpr):
    # the inverse of gu_quant_kernel: out (2I, tokens) in its half dtype, gate rows then up rows
    grp = tl.program_id(0)
    tok = tl.program_id(1) * TM + tl.arange(0, TM)
    live = tok < tokens
    rows = grp * GROUP + tl.arange(0, GROUP)
    at = rows[:, None].to(tl.int64) * tokens + tok[None, :]
    packed = tl.load(q_ptr + at, mask=live[None, :], other=0).to(tl.int32)
    s_g = tl.load(s_ptr + grp * tokens + tok, mask=live, other=0.0)
    s_u = tl.load(s_ptr + (inter // GROUP + grp) * tokens + tok, mask=live, other=0.0)
    gate = ((packed << 24) >> 24).to(tl.float32) * s_g[None, :]
    up = (packed >> 8).to(tl.float32) * s_u[None, :]
    tl.store(out_ptr + at, gate.to(out_ptr.dtype.element_ty), mask=live[None, :])
    tl.store(out_ptr + inter.to(tl.int64) * tokens + at, up.to(out_ptr.dtype.element_ty),
             mask=live[None, :])


# ---------------------------------------------------------------------------------------------


def _scales_offset(rows, tokens):
    return -(-rows * tokens // 16) * 16


def wire_bytes(rows, tokens):
    """Bytes of the int8 form of a (rows, tokens) half tensor: the int8 values, then the scales."""
    return _scales_offset(rows, tokens) + 4 * (rows // GROUP) * tokens


def views(wire, rows, tokens):
    """(q (rows / 2, tokens) int16, s (rows // GROUP, tokens) fp32) inside a uint8 buffer of
    wire_bytes, or the pair itself (made in place by swiglu_quant)."""
    if isinstance(wire, tuple):
        return wire
    off = _scales_offset(rows, tokens)
    q = wire[:rows * tokens].view(torch.int16).view(rows // 2, tokens)
    s = wire[off:off + 4 * (rows // GROUP) * tokens].view(torch.float32).view(rows // GROUP, tokens)
    return q, s


def _grid(rows, tokens):
    return (rows // 2 // GROUP, triton.cdiv(tokens, TILE_M))


class Codec:
    """What the offload keeps to rebuild one tensor: its shape and dtype (the int8 form itself is the
    offload's GPU or host copy: one uint8 buffer of wire_bytes, or the (q, s) pair of swiglu_quant)."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype

    def unpacked(self, wire):
        """What the unpack hands over: the uint8 buffer itself inside raw(True), else decode(wire) (a
        (q, s) pair still on the GPU is rebuilt too: autograd takes a tensor)."""
        return wire if RAW["now"] and not isinstance(wire, tuple) else self.decode(wire)

    def decode(self, wire):
        """The half (rows, tokens) tensor from its int8 form on the GPU, on the current stream."""
        rows, tokens = self.shape
        q, s = views(wire, rows, tokens)
        out = torch.empty(self.shape, dtype=self.dtype, device=q.device)
        gu_dequant_kernel[_grid(rows, tokens)](q, s, out, rows // 2, tokens, GROUP=GROUP, TM=TILE_M,
                                               num_warps=WARPS)
        return out


@contextlib.contextmanager
def raw(on):
    """While active with `on` (and SOUP_SOLUTION_OFFLOAD_INT8_FUSED), the unpack of an int8 handle
    returns its uint8 buffer (wire_bytes of its shape) instead of the rebuilt half tensor."""
    _STATE["backwards"] += 1
    if (_STATE["on"] and _STATE["thread"] is None and not _READY.is_set()
            and _STATE["backwards"] > COMPILE_AFTER):
        _start_compile()
    previous = RAW["now"]
    RAW["now"] = bool(on) and RAW["on"]
    try:
        yield
    finally:
        RAW["now"] = previous


def encode(tensor):
    """(uint8 buffer of wire_bytes with the int8 form of `tensor`, Codec), queued on the current
    stream; `tensor` is not changed (tests, benchmarks)."""
    rows, tokens = tensor.shape
    wire = torch.empty(wire_bytes(rows, tokens), dtype=torch.uint8, device=tensor.device)
    q, s = views(wire, rows, tokens)
    gu_quant_kernel[_grid(rows, tokens)](tensor, tensor, q, s, rows // 2, tokens, GROUP=GROUP,
                                         TM=TILE_M, SWIGLU=False, num_warps=WARPS)
    return wire, Codec(tensor.shape, tensor.dtype)


def swiglu_quant(gate_up, hidden):
    """hidden = SwiGLU(gate_up) as the example kernel computes it, and gate_up's int8 form written in
    place over its gate half: returns (q, s), q a view of gate_up's storage (gate_up's values are gone),
    s the new scales. On the current stream."""
    rows, tokens = gate_up.shape
    q = gate_up[:rows // 2].view(torch.int16)
    s = torch.empty((rows // GROUP, tokens), dtype=torch.float32, device=gate_up.device)
    gu_quant_kernel[_grid(rows, tokens)](gate_up, hidden, q, s, rows // 2, tokens, GROUP=GROUP,
                                         TM=TILE_M, SWIGLU=True, num_warps=WARPS)
    return q, s


def applies(tensor):
    """Whether this early offload goes int8: switched on, a contiguous 2-D half CUDA tensor whose two
    halves split into groups, and START_CALLS such offloads ran before (the kernels compile meanwhile)."""
    if not _STATE["on"]:
        return False
    if (tensor.dtype not in HALF_DTYPES or tensor.dim() != 2 or not tensor.is_cuda
            or not tensor.is_contiguous() or tensor.shape[0] % (2 * GROUP) or tensor.shape[1] < 1):
        return False
    _STATE["calls"] += 1
    if _STATE["calls"] <= START_CALLS:
        return False  # half for the first START_CALLS, compiled or not: the switch point is fixed
    if not _READY.is_set():
        _start_compile()  # raw() starts it at the first MLP backward; here only without one
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
    """The training's two kernels (tokens a multiple of 16: unpad.py's tail)."""
    from .warmup import SKIPPED

    if not _STATE["on"]:
        return SKIPPED
    _warm_codec()
    if raw_in_use():  # the int8 MLP backward reading the int8 form
        from . import int8_bwd

        int8_bwd.warm_int8_in(GROUP)
    return None


def _warm_codec():
    """The codec's two kernels (warmup.py's offload_int8_kernels registration compiles them alone)."""
    from .warmup import INTERMEDIATE, TOKENS, compile_kernel, training_half_dtype

    tokens = -(-TOKENS // 16) * 16
    half = training_half_dtype()
    compile_kernel(gu_quant_kernel, half, half, torch.int16, torch.float32, INTERMEDIATE, tokens,
                   GROUP=GROUP, TM=TILE_M, SWIGLU=True, num_warps=WARPS)
    compile_kernel(gu_dequant_kernel, torch.int16, torch.float32, half, INTERMEDIATE, tokens,
                   GROUP=GROUP, TM=TILE_M, num_warps=WARPS)


def raw_in_use():
    """Whether int8_bwd.py's SwiGLU backward reads the int8 form directly (its INT8_IN variants)."""
    from . import int8_bwd

    return RAW["on"] and int8_bwd._STATE["on"]
