"""What the step keeps on the GPU, by the step's size (task: activations policy under a fixed peak).

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
_STEPS = []  # [tokens, peak allocated MiB, peak reserved MiB] per step
_STATE = {"tokens": None, "registered": False}


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
