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
- unpad.py: the layers run on the packed real tokens of the batch, attention per row;
- mlp16.py: the MLP's gate/up/SwiGLU in half storage (uses the kit's example SwiGLU kernel, which
  does not count as ours);
- embed16.py: the tied embedding / lm_head weight back in fp16 after Soup's setup.

Switches (environment, for experiments): SOUP_SOLUTION_CE_CHUNK (rows of logits per chunk, default
512), SOUP_SOLUTION_UNPAD=0, SOUP_SOLUTION_MLP16=0, SOUP_SOLUTION_EMBED16=0.
"""

import os


def _on(name):
    return os.environ.get(name, "1") not in ("0", "false", "no")


def apply():
    from .phases import mark, patch_soup_phases

    mark("apply begin")
    from .embed16 import patch_soup_setup
    from .fused_ce import patch_qwen2_causal_lm
    from .mlp16 import patch_qwen2_mlp_half
    from .trainer_patch import patch_sft_trainer
    from .unpad import patch_qwen2_attention

    chunk = int(os.environ.get("SOUP_SOLUTION_CE_CHUNK", "512"))
    patch_qwen2_causal_lm(chunk=chunk, unpad_enabled=_on("SOUP_SOLUTION_UNPAD"))
    patch_qwen2_attention()
    patch_sft_trainer()
    if _on("SOUP_SOLUTION_MLP16"):
        patch_qwen2_mlp_half()
    else:
        from .swiglu import patch_qwen2_mlp

        patch_qwen2_mlp()
    if _on("SOUP_SOLUTION_EMBED16"):
        patch_soup_setup()
    patch_soup_phases()
    mark("apply end")
