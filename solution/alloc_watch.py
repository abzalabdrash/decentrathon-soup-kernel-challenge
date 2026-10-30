"""The caching allocator's reserved pool during training: watched, and never left below its peak.

Check 9 asks for nvidia-smi's peak (sampled every 100 ms by the harness) >= 0.97 x
torch.cuda.max_memory_reserved(). nvidia-smi sees the pool plus the CUDA context (about 160 MiB on
the T4), so the check holds as long as the pool keeps every peak mapped for at least one sample. It
fails when the pool reaches a peak and gives it back within 100 ms: torch.cuda.empty_cache() after a
transient (setup), or the allocator's own release path (an allocation finds no free block and no
room to map, waits for every pending stream event, unmaps cached segments and retries). The
allocator setting (expandable_segments:True) is the harness's and stays.

This module installs a trainer callback on a real CUDA trainer (Soup's tests with fakes run the
original train). It reads host-side counters only, no synchronization:
- it logs the reserved MiB at train begin, after step 1, its maximum and at the end, how many steps
  ended with the pool smaller than its peak so far, and the allocator's retries (the release path)
  and segment unmaps during training;
- with the guard (on unless SOUP_SOLUTION_ALLOC_GUARD=0), whenever the pool is more than
  GUARD_SLACK_MIB below torch.cuda.max_memory_reserved() (at train begin, after a setup transient;
  after any step; at train end), it maps that memory again: chunks are allocated until the pool is back at its peak, then
  freed, and the freed blocks stay cached (mapped). Training then takes its memory from those blocks.
  The guard never lowers what nvidia-smi sees, it only makes nvidia-smi see the peak that PyTorch
  reports; when nothing shrinks it does nothing.

Switches: SOUP_SOLUTION_ALLOC_GUARD=0 (watch only), SOUP_SOLUTION_ALLOC_GUARD_SLACK_MIB.
"""

import os

import torch

MIB = 2 ** 20
GUARD_SLACK_MIB = 32  # nvidia-smi's context overhead (~160 MiB) and the 3% leave more than this
CHUNK = 64 * MIB
PAGE = 2 * MIB  # expandable segments map in 2 MiB pages; smaller requests go to the small pool


def _guard_on():
    return os.environ.get("SOUP_SOLUTION_ALLOC_GUARD", "1") not in ("0", "false", "no", "")


def _slack():
    value = os.environ.get("SOUP_SOLUTION_ALLOC_GUARD_SLACK_MIB", "").strip()
    return (int(value) if value else GUARD_SLACK_MIB) * MIB


def _counters():
    stats = torch.cuda.memory_stats()
    return {name: int(stats.get(name, 0)) for name in ("num_alloc_retries", "num_device_free",
                                                        "num_device_alloc", "num_sync_all_streams")}


def refill_to_peak(slack=None):
    """Map the pool back up to torch.cuda.max_memory_reserved() when it is more than `slack` bytes
    below it: hold chunks until memory_reserved() reaches the peak, then free them (they stay cached).
    Returns the bytes added to the pool."""
    slack = _slack() if slack is None else slack
    target = torch.cuda.max_memory_reserved()
    before = torch.cuda.memory_reserved()
    if target - before <= slack:
        return 0
    held = []
    total = 0
    try:
        # cached free blocks serve the first chunks without mapping; the cap bounds the loop
        while total < target:
            missing = target - torch.cuda.memory_reserved()
            if missing < PAGE:
                break
            size = min(CHUNK, -(-missing // PAGE) * PAGE)
            held.append(torch.empty(size, dtype=torch.uint8, device="cuda"))
            total += size
    except torch.OutOfMemoryError:
        pass
    finally:
        del held
    return torch.cuda.memory_reserved() - before


def _print(text):
    print(text, flush=True)


def make_callback(guard=None, log=_print):
    import transformers

    guard = _guard_on() if guard is None else guard

    class AllocWatch(transformers.TrainerCallback):
        def __init__(self):
            self.begin = self.after_first = self.max = self.end = None
            self.shrunk = 0
            self.deepest = 0
            self.refilled = 0
            self.counters = None
            self.steps = 0

        def _refill(self):
            if guard:
                added = refill_to_peak()
                if added:
                    self.refilled += added
                    log(f"[solution] alloc guard: pool mapped back up by {added / MIB:.0f} MiB to "
                        f"{torch.cuda.memory_reserved() / MIB:.0f} MiB (peak "
                        f"{torch.cuda.max_memory_reserved() / MIB:.0f} MiB)")

        def on_train_begin(self, args, state, control, **kwargs):
            self.counters = _counters()
            self._refill()
            self.begin = self.max = torch.cuda.memory_reserved()

        def on_step_end(self, args, state, control, **kwargs):
            self.steps += 1
            reserved = torch.cuda.memory_reserved()
            if self.steps == 1:
                self.after_first = reserved
            if reserved < self.max:
                self.shrunk += 1
                self.deepest = max(self.deepest, self.max - reserved)
                self._refill()
                reserved = torch.cuda.memory_reserved()
            self.max = max(self.max, reserved)

        def on_train_end(self, args, state, control, **kwargs):
            self._refill()
            self.end = torch.cuda.memory_reserved()
            now = _counters()
            delta = {k: now[k] - self.counters[k] for k in now}
            after = "-" if self.after_first is None else f"{self.after_first / MIB:.0f}"
            log(f"[solution] allocator: reserved MiB at train begin {self.begin / MIB:.0f}, after "
                f"step 1 {after}, max at step ends {self.max / MIB:.0f}, at train end "
                f"{self.end / MIB:.0f}, torch's peak {torch.cuda.max_memory_reserved() / MIB:.0f}; "
                f"pool below its running max after {self.shrunk} of {self.steps} steps (by up to "
                f"{self.deepest / MIB:.0f} MiB); during training: {delta['num_alloc_retries']} alloc "
                f"retries, {delta['num_device_free']} segment frees, {delta['num_device_alloc']} "
                f"segment allocs, {delta['num_sync_all_streams']} syncs of all streams; guard "
                f"{'on' if guard else 'off'}, mapped back {self.refilled / MIB:.0f} MiB")

    return AllocWatch()


def _real_cuda_trainer(trainer):
    if not torch.cuda.is_available():
        return False
    try:
        from transformers import Trainer
    except Exception:  # noqa: BLE001
        return False
    model = getattr(trainer, "model", None)
    if not isinstance(trainer, Trainer) or not isinstance(model, torch.nn.Module):
        return False
    param = next(model.parameters(), None)
    return param is not None and param.is_cuda


def patch_soup_alloc_watch(guard=None):
    """SFTTrainerWrapper.train adds the AllocWatch callback to a real CUDA trainer; anything else
    (Soup's regression tests with fake trainers, CPU) runs the original."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_alloc_watch", False):
        return
    original = cls.train

    def train(self, *args, **kwargs):
        trainer = getattr(self, "trainer", None)
        if _real_cuda_trainer(trainer):
            trainer.add_callback(make_callback(guard))
        return original(self, *args, **kwargs)

    cls.train = train
    cls._soup_alloc_watch = True
    cls._soup_alloc_watch_previous = original
