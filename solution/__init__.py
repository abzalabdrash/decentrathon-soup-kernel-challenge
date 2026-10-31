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
- offload.py: the MLP's saved gate|up to pinned host memory on a side stream with events (prepinned
  slabs), or recomputed in the backward (SOUP_SOLUTION_MLP_MEMORY=recompute: +50 ms per step for
  24 MiB on the T4, so off by default);
- alloc_watch.py: the caching allocator's retries and segment frees, logged once per run;
- head_stream.py: the tied embedding / lm_head weight in pinned host memory for the whole run,
  the head computed in one streamed pass over the vocabulary (own kernels); switched on from
  fused_ce.patch_qwen2_causal_lm (SOUP_SOLUTION_HEAD_STREAM=0 turns it off);
- weight_stream.py: the layers' 4-bit weights in pinned host memory, a ring of K layers on the GPU;
- warmup.py: the kernels compiled by a background thread while Soup loads the model (triton's own
  hash started with apply());
- startup.py: apply()'s imports without work the run does not use (trl's second metadata scan,
  scikit-learn, wandb, bitsandbytes' Inductor), without stubs in sys.modules; a first-step phase
  mark; a profile of the trainer build (diagnostics). The garbage collector and datasets'
  fingerprints are not touched (organizers' answers 13 and 31);
- phases.py: wall-clock marks of the run's phases in the log;
- diag.py: diagnostics of a run (SOUP_SOLUTION_DIAG=1 only): memory at the marks, a PCIe probe,
  an allocation snapshot of one step, a chrome trace of two steps.

Switches (environment, for experiments): SOUP_SOLUTION_CE_CHUNK (rows of logits per chunk, default
512), SOUP_SOLUTION_UNPAD=0, SOUP_SOLUTION_MLP16=0, SOUP_SOLUTION_EMBED16=0, SOUP_SOLUTION_RMSNORM=0,
SOUP_SOLUTION_NF4=0, SOUP_SOLUTION_ROPE=0, SOUP_SOLUTION_FUSED_PROJ=0, SOUP_SOLUTION_MLP_MEMORY=
offload|recompute|none, SOUP_SOLUTION_WEIGHT_STREAM=0 (SOUP_SOLUTION_WEIGHT_SLOTS=2), SOUP_SOLUTION_WARMUP=0,
SOUP_SOLUTION_PAD_TOKENS=16 (0 turns the alignment tail off), SOUP_SOLUTION_VARLEN_ATTN=0,
SOUP_SOLUTION_REBUILD_X16=0, SOUP_SOLUTION_CARRY_RESIDUAL=0, SOUP_SOLUTION_HEAD_STREAM=0
(SOUP_SOLUTION_HEAD_ROWS, SOUP_SOLUTION_HEAD_SLOTS, SOUP_SOLUTION_HEAD_PREFETCH=norm|embedding|head),
SOUP_SOLUTION_IMPORT_DIET=0, SOUP_SOLUTION_HEAD_PACK=1 (off by default),
SOUP_SOLUTION_EXIT_RELEASE=1 (on by default, exit_release.py; =0 turns off, =all also empties the cache),
SOUP_SOLUTION_INT8_BWD=0 (int8 MLP backward, int8_bwd.py; on by default, answer 29),
SOUP_SOLUTION_EARLY_RELEASE=0 (the int8 MLP backward releases each 4-bit slot right after its last read;
on by default), SOUP_SOLUTION_INT8_FUSED_QUANT=0 (its gradient quantization and step-7 scaling as own
kernels, bit-for-bit torch's ops; on by default),
diagnostics SOUP_SOLUTION_PROFILE_STARTUP=1.
"""

import os


def _on(name):
    return os.environ.get(name, "1") not in ("0", "false", "no")


def apply():
    from .phases import mark, patch_soup_phases

    mark("apply begin")
    # Defaults of switches that other modules read at import time (int8_bwd.py reads its own when
    # fused_proj imports it), so they come before any patch import; an environment value wins.
    os.environ.setdefault("SOUP_SOLUTION_EXIT_RELEASE", "1")  # E1: pinned host memory handed back on a thread at the last step; not "all" (check 9)
    # answer 29 (chat, 31 Oct): int8 tensor cores for the frozen linear layers in the forward and the backward by
    # input, and for lm_head, are allowed when all 10 checks pass; the MLP backward int8 (int8_bwd.py) measured
    # 10/10 on the public rows (43.1 s / 395 MiB) and on the jury rows (s 1.91 against 1.80-1.85 without it).
    # int8_bwd.py reads SOUP_SOLUTION_INT8_BWD itself (default on, the same in every process); the bitwise
    # tests of the fp16 path pin it off in solution/conftest.py.
    if _on("SOUP_SOLUTION_WARMUP"):
        from .warmup import start_early

        start_early()  # triton's own hash now, next to the imports, not next to the model load
    if _on("SOUP_SOLUTION_IMPORT_DIET"):
        from .startup import prepare_imports

        prepare_imports()  # trl without a second metadata scan, without sklearn and wandb
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
    from .offload import patch_mlp_memory

    patch_mlp_memory()  # SOUP_SOLUTION_MLP_MEMORY=offload (default, event-based, prepinned)|recompute|none
    from .alloc_watch import patch_soup_alloc_watch

    patch_soup_alloc_watch()  # one log line per run: allocator retries, segment frees, reserved vs peak
    if _on("SOUP_SOLUTION_WEIGHT_STREAM"):
        from .weight_stream import patch_weight_stream

        patch_weight_stream(slots=int(os.environ.get("SOUP_SOLUTION_WEIGHT_SLOTS", "2")))
    patch_soup_phases()
    if os.environ.get("SOUP_SOLUTION_EXIT_RELEASE", "0") not in ("0", "false", "no", ""):
        from .exit_release import patch_soup_exit_release

        patch_soup_exit_release()  # off by default: pinned host memory released at the last step's end
    from .startup import patch_soup_first_step_mark, patch_soup_startup_profile

    patch_soup_first_step_mark()
    if os.environ.get("SOUP_SOLUTION_PROFILE_STARTUP", "0") not in ("0", "false", "no", ""):
        patch_soup_startup_profile()  # off by default: cProfile of the trainer build, see startup.py
    if os.environ.get("SOUP_SOLUTION_DIAG", "0") not in ("0", "false", "no", ""):
        from .diag import patch_soup_diag

        patch_soup_diag()  # off by default: memory marks, PCIe probe, one step's allocation history, a trace
    if os.environ.get("SOUP_SOLUTION_CPU_PROFILE", "0") not in ("0", "false", "no", ""):
        from .cpu_profile import patch_soup_cpu_profile

        patch_soup_cpu_profile()  # off by default: the CPU side of training steps 5-12, see cpu_profile.py
    if _on("SOUP_SOLUTION_WARMUP"):
        from .warmup import patch_soup_warmup, start_background_warmup, watch_compiles

        watch_compiles()  # logs any kernel compiled or loaded outside the warm-up
        patch_soup_warmup(start_in_setup=False)
        start_background_warmup()  # compiles every registered kernel in a background thread
    if _on("SOUP_SOLUTION_IMPORT_DIET"):
        pass  # gc.freeze() of the imported objects removed: the organizers forbade collector tweaks in apply() (answer 31)
    mark("apply end")
