"""What the step keeps on the GPU, by the step's size (task: activations policy under a fixed peak).

SOUP_SOLUTION_RECOMPUTE_ATTN=cap (fused_proj.recompute_layers): a step of N tokens keeps the
attention of its last `kept_layers(N)` layers for the backward instead of recomputing it, as many as
fit between the default policy's predicted peak of the step and a fixed cap (CAP_MIB of allocated
memory, a number in the code): long steps (which set the run's peak) recompute every layer as
before, shorter ones spend the headroom under the cap on not recomputing. The decision depends on
the step's token count only (the same in tests and in the measured run). T4, default policy
(attention recomputed in every layer), peak allocated per step against its tokens (soup-mp-pub1 /
soup-mp-jury1, SOUP_SOLUTION_MEM_LOG): 107 + 0.086 MiB per token, 230 MiB at the public rows'
1,440-token step (reserved 274), 251 at 1,680 tokens; every layer that keeps its attention adds
~10 KB per token (0.265 MiB per token over the 28 layers).

SOUP_SOLUTION_MEM_LOG=1 (diagnostics, off by default): every training step's peak of allocated and
reserved CUDA memory against its token count, printed at exit (one line per step), so that the
peak of a step can be modelled from its size for each policy. The step boundary is the first
layer's attention forward with autograd on (fused_proj.py); the allocator's peak statistics are
reset there, so with the log on torch's run peak is that of the last step only (the harness's
nvidia-smi peak is unaffected).
"""

import atexit
import os

import torch

LOG = os.environ.get("SOUP_SOLUTION_MEM_LOG", "0").strip().lower() not in ("0", "false", "no", "")
MIB = 1 << 20
# the default policy's step peak (allocated MiB = BASE_MIB + PER_TOKEN_MIB * tokens) and the cap
BASE_MIB, PER_TOKEN_MIB = 107.0, 0.086
CAP_MIB = float(os.environ.get("SOUP_SOLUTION_STEP_CAP_MIB", "250") or "250")
_STEPS = []  # [tokens, peak allocated MiB, peak reserved MiB] per step
_STATE = {"tokens": None, "registered": False}


def kept_layers(tokens, layer_bytes_per_token, layers):
    """How many layers of a step of `tokens` tokens may keep their attention for the backward (each
    `layer_bytes_per_token` more per token than recomputing it) without the step's predicted peak
    going over CAP_MIB."""
    if tokens <= 0 or layer_bytes_per_token <= 0:
        return 0
    budget = (CAP_MIB - BASE_MIB - PER_TOKEN_MIB * tokens) * MIB
    if budget <= 0:
        return 0
    return min(layers, int(budget // (tokens * layer_bytes_per_token)))


def step_begin(tokens):
    """The first layer's forward of a training step of `tokens` tokens (fused_proj.py)."""
    if not LOG:
        return
    if _STATE["tokens"] is not None:
        _STEPS.append([_STATE["tokens"], torch.cuda.max_memory_allocated() / MIB,
                       torch.cuda.max_memory_reserved() / MIB])
    elif not _STATE["registered"]:
        atexit.register(report)
        _STATE["registered"] = True
    torch.cuda.reset_peak_memory_stats()
    _STATE["tokens"] = int(tokens)


def report():
    """The per-step lines (the last step's peak included)."""
    if _STATE["tokens"] is not None:
        _STEPS.append([_STATE["tokens"], torch.cuda.max_memory_allocated() / MIB,
                       torch.cuda.max_memory_reserved() / MIB])
        _STATE["tokens"] = None
    for i, (tokens, alloc, reserved) in enumerate(_STEPS):
        print(f"[mem_policy] step {i}: tokens {tokens}, peak allocated {alloc:.1f} MiB, "
              f"reserved {reserved:.1f} MiB", flush=True)
