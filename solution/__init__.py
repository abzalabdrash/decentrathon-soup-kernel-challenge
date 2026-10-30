"""Our solution: Triton kernels and patches that make the fixed `soup train` run on a Kaggle T4
faster and lighter without changing what it trains.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Every patch is a class-level method replacement looked
up at call time, guarded so that anything that is not the real CUDA Qwen2 training step (Soup's
regression tests with fake modules, eval, generation, CPU) takes the original code path.

Modules:
- fused_ce.py: lm_head + cross-entropy + TRL metrics without the (B, T, V) logits (own kernel);
- trainer_patch.py: TRL's compute_loss logging the same metrics from the fused head, and Soup's
  rewind recorder fed through its own hook for steps without logits;
- swiglu.py: the kit's example (kept on; it does not count as ours).
"""

import os


def apply():
    from .fused_ce import patch_qwen2_causal_lm
    from .swiglu import patch_qwen2_mlp
    from .trainer_patch import patch_sft_trainer

    chunk = int(os.environ.get("SOUP_SOLUTION_CE_CHUNK", "1024"))
    patch_qwen2_causal_lm(chunk=chunk)
    patch_sft_trainer()
    patch_qwen2_mlp()  # the example: works out of the box, does not count as our own kernel
