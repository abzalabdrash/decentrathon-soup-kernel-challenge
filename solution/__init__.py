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
- embed16.py: the tied embedding / lm_head weight back in fp16 after Soup's setup;
- rmsnorm.py: RMSNorm and residual-add + RMSNorm in one Triton kernel each way (own kernels);
- nf4.py: NF4 dequantization in one Triton kernel, used by bitsandbytes' MatMul4Bit (own kernel);
- rope.py: rotary embedding of q and k in one Triton kernel each way (own kernel);
- fused_proj.py: q+k+v in one GEMM with LoRA by addmm, attention in half, gate+up in one GEMM;
- offload.py: the MLP's saved gate/up to pinned host memory (or recomputed) during the backward;
- weight_stream.py: the layers' 4-bit weights in pinned host memory, a ring of K layers on the GPU;
- warmup.py: the kernels compiled by a background thread while Soup loads the model;
- phases.py: wall-clock marks of the run's phases in the log.

Switches (environment, for experiments): SOUP_SOLUTION_CE_CHUNK (rows of logits per chunk, default
512), SOUP_SOLUTION_UNPAD=0, SOUP_SOLUTION_MLP16=0, SOUP_SOLUTION_EMBED16=0, SOUP_SOLUTION_RMSNORM=0,
SOUP_SOLUTION_NF4=0, SOUP_SOLUTION_ROPE=0, SOUP_SOLUTION_FUSED_PROJ=0, SOUP_SOLUTION_MLP_MEMORY=
offload|recompute|none, SOUP_SOLUTION_WEIGHT_STREAM=0 (SOUP_SOLUTION_WEIGHT_SLOTS=2), SOUP_SOLUTION_WARMUP=0.
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
    if _on("SOUP_SOLUTION_RMSNORM"):
        from .rmsnorm import patch_qwen2_decoder_layer, patch_qwen2_rmsnorm

        patch_qwen2_rmsnorm()
        patch_qwen2_decoder_layer()
    if _on("SOUP_SOLUTION_NF4"):
        from .nf4 import patch_bnb_matmul4bit

        patch_bnb_matmul4bit()
    if _on("SOUP_SOLUTION_ROPE"):
        from .rope import patch_qwen2_rope

        patch_qwen2_rope()
    if _on("SOUP_SOLUTION_FUSED_PROJ"):
        from .fused_proj import patch_qwen2_attention_fused, patch_qwen2_mlp_fused

        patch_qwen2_attention_fused()  # after unpad's attention patch and rope: it falls back to them
        patch_qwen2_mlp_fused()  # after mlp16's patch: it falls back to it
    mode = os.environ.get("SOUP_SOLUTION_MLP_MEMORY", "offload").strip().lower()
    if mode == "recompute" and _on("SOUP_SOLUTION_FUSED_PROJ"):
        mode = "offload"  # recompute replaces the MLP forward; with the fused MLP use the offload
    from .offload import patch_mlp_memory

    patch_mlp_memory(mode)
    if _on("SOUP_SOLUTION_WEIGHT_STREAM"):
        from .weight_stream import patch_weight_stream

        patch_weight_stream(slots=int(os.environ.get("SOUP_SOLUTION_WEIGHT_SLOTS", "2")))
    patch_soup_phases()
    if _on("SOUP_SOLUTION_WARMUP"):
        from .warmup import patch_soup_warmup, start_background_warmup

        patch_soup_warmup(start_in_setup=False)
        start_background_warmup()
    mark("apply end")
