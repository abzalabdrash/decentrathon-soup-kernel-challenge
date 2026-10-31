# B1: the step boundary and the host waits inside the step (branch claude/b1-boundary-e6120e)

## Where the time went (T4 trace of main a6c3fe3, soup-b1-diag0, steps 8-9 under the profiler)

- Harness: `Probe.on_step_begin` and `on_step_end` call `torch.cuda.synchronize()` (harness/_train.py),
  so the queue is empty at both ends of every step; anything between `on_step_end` and the next
  `on_step_begin` (log, Soup callbacks, DataLoader) runs with the GPU idle no matter what.
- Inside the step: the forward is GPU-bound (the head's `int(labelled.sum().item())` waited 69.6 ms
  for it), the head is PCIe-bound, the TRL metric `.item()` waited 33 ms for the head, the backward is
  GPU-bound (transformers' nan/inf check of the loss waited 62.9 ms for it). So the in-step drains
  cost only ~2-4 ms of GPU idle (the backward start), and the CPU has ~60 ms of slack at the end of
  the backward.
- Boundary (backward end 530.8 ms -> first layer kernel 562.0 ms): ~31 ms GPU idle:
  FLOP counter walk 4.9 ms, clip launches ~5 ms (Python over all ~450 parameters), AdamW 2.8 ms
  (GPU 0.8), after Adam (scaler update, lr, zero_grad walk, callbacks) 3.2 ms, log + callbacks
  4.6 ms, DataLoader fetch/collate 2.7 ms, step start (model.train walk, compute_loss chain) 3.4 ms,
  forward start (pack_batch nonzero/tolist, embedding host gather, cu_seqlens copy) 2.7 ms.
- night-boundary's `train()` fast path never engaged: Qwen2ForCausalLM/Qwen2Model use
  `PreTrainedModel.train` in transformers 5.16, and its own `PeftModel.train` wrapper failed its
  "plain" check. Fixed here (both accepted; `use_kernels` checked per call).

## What changed (each behind a switch, defaults below)

(filled in after the T4 measurements)
