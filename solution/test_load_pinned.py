"""load_pinned.py: the checkpoint -> GPU copies of the model load through pinned buffers that a
thread fills ahead of the loader.

CPU tests (run anywhere): the copier's protocol with its device operations replaced by CPU ones
(copies made ahead are done later by a fake copy engine, so a buffer refilled before its copy is out
would show), against transformers' own spawn_materialize / _materialize_copy on a safetensors file:
every tensor bit for bit for every window and number of buffers; tensors asked out of order, one
failed staging, a broken staging thread, a non-default stream, closing mid-way, jobs from another
thread, the plan closed once the second pass began.

CUDA tests: a tiny 4-bit Qwen2 (bf16 on disk, loaded in fp16) loaded in Soup's setup order with
load_stream.py's moves, with the copies (windows 0, 1, 2) against the plain transformers load and the
same setup without the copies: every packed weight, absmax and quant-state tensor and every other
parameter and buffer bit for bit, for every layer; the same training step; the load's GPU peak not
above the original with window 0 and above it by at most the window's tensors otherwise; loads
outside Soup's setup untouched. The real Qwen2.5-1.5B-Instruct checkpoint tensor by tensor against
_materialize_copy, and as a whole load against the plain load (skipped when it cannot be fetched).
"""

import gc
import glob
import os
import queue
import threading
import time

import pytest
import torch

VOCAB, HIDDEN, LAYERS = 1000, 128, 4
REAL = "Qwen/Qwen2.5-1.5B-Instruct"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _same(a, b):
    """Same dtype, shape and bytes (NaN payloads included)."""
    if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.dtype != b.dtype or a.shape != b.shape:
        return False
    a, b = a.detach().reshape(-1).contiguous(), b.detach().reshape(-1).contiguous()
    if a.device != b.device:
        a, b = a.cpu(), b.cpu()
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# ---- CPU: the protocol -----------------------------------------------------------------------------


class _Event:
    def __init__(self):
        self._done = threading.Event()

    def set(self):
        self._done.set()

    def synchronize(self):
        self._done.wait()


class _Engine:
    """A copy engine on the CPU: copies run in issue order in a thread, `delay` seconds each."""

    def __init__(self, delay):
        self.delay, self.queue = delay, queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, out, source, cast=None):
        """Copies `source` into `out` (into `out` viewed as `cast`, then cast in place, as the GPU
        does with the raw bytes)."""
        event = _Event()
        self.queue.put((out, source, cast, event))
        return event

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            out, source, cast, event = item
            if out is not None:
                if self.delay:
                    time.sleep(self.delay)
                raw = out if cast is None else out.view(cast)
                raw.copy_(source)
                if cast is not None:
                    out.copy_(raw)
            event.set()

    def drain(self):
        self.submit(None, None).synchronize()

    def stop(self):
        self.queue.put(None)
        self.thread.join()


class _Buffer:
    def __init__(self, nbytes):
        self.nbytes = nbytes
        self.buffer = torch.full((nbytes,), 0xAB, dtype=torch.uint8)  # garbage until staged

    def unpin(self):
        self.buffer = None


def _cpu_copier(window, slots, delay=0.0003, default_stream=True, stage=None, pin=None, cast="cpu"):
    from solution.load_pinned import PinnedCopier

    class CpuCopier(PinnedCopier):
        def _make_side_stream(self):
            return "side"

        def _thread_setup(self):
            pass

        def _pin(self, nbytes):
            if pin is not None:
                pin(nbytes)
            buffer = _Buffer(nbytes)
            self.made.append(buffer)
            return buffer

        def _on_default_stream(self):
            return default_stream

        def _submit(self, entry):
            out = torch.empty(entry.shape, dtype=entry.dtype)
            cast = entry.stage_dtype if entry.stage_dtype != entry.dtype else None
            return out, self.engine.submit(out, entry.staged, cast)

        def _copy(self, entry):  # on the consumer's stream: done before the consumer goes on
            out, event = self._submit(entry)
            event.synchronize()
            return out, event

        def _copy_ahead(self, entry):  # in flight until the engine gets to it
            return self._submit(entry)

        def _hand_over(self, event):
            event.synchronize()

        def _synchronize(self):
            self.engine.drain()

    if stage is not None:
        CpuCopier._stage = staticmethod(stage)
    copier = CpuCopier("cpu", window=window, slots=slots, cast=cast, log=lambda *_: None)
    copier.made, copier.engine = [], _Engine(delay)
    return copier


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A safetensors file with the kinds of tensors of a decoder checkpoint, bf16 values that fp16
    rounds, overflows and flushes, and tensors the copier must leave alone (ints, the embedding)."""
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(0)
    tensors = {}
    for layer in range(3):
        p = f"model.layers.{layer}."
        tensors[p + "input_layernorm.weight"] = torch.randn(64, generator=gen).to(torch.bfloat16)
        tensors[p + "mlp.down_proj.weight"] = (torch.randn(64, 160, generator=gen) * 3).to(torch.bfloat16)
        tensors[p + "mlp.gate_proj.weight"] = torch.randn(160, 64, generator=gen).to(torch.bfloat16)
        tensors[p + "mlp.up_proj.weight"] = torch.randn(160, 64, generator=gen).to(torch.bfloat16)
        tensors[p + "self_attn.k_proj.weight"] = torch.randn(16, 64, generator=gen)  # fp32, kept
        tensors[p + "self_attn.q_proj.bias"] = torch.randn(64, generator=gen).to(torch.bfloat16)
        tensors[p + "self_attn.q_proj.weight"] = torch.randn(64, 64, generator=gen).to(torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randn(300, 64, generator=gen).to(torch.bfloat16)
    tensors["model.extremes"] = torch.tensor(
        [1e5, -7e4, 65504.0, 65520.0, 1e-8, -3e-8, 6e-5, 2.0 ** -24, 2.0 ** -25, 3 * 2.0 ** -26, 0.0, -0.0,
         float("inf"), float("-inf"), float("nan"), 1.0 + 2.0 ** -7]).to(torch.bfloat16)
    tensors["model.norm.weight"] = torch.randn(64, generator=gen).to(torch.float16)
    tensors["model.steps"] = torch.arange(5)  # int64: the original job
    path = tmp_path_factory.mktemp("load_pinned") / "model.safetensors"
    save_file(tensors, str(path))
    return path


def _spec(key, slice_):
    """(device, dtype) as the loop asks for them: fp16 on the copier's device, the embedding read
    elsewhere (load_stream.py reads it to the host: not the copier's), fp32 and ints kept."""
    if key == "model.embed_tokens.weight":
        return None, torch.float16
    return "cpu", {"BF16": torch.float16, "F16": torch.float16, "F32": torch.float32}.get(slice_.get_dtype())


def _planned(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        return sum(1 for k in f.keys() if _spec(k, f.get_slice(k)) in (("cpu", torch.float16), ("cpu", torch.float32)))


def _run(copier, path, order=None, stop_after=None, pause=0.0, in_thread=False):
    """transformers' two passes over the checkpoint (spawn_materialize, then the jobs in `order`),
    with the copier's jobs; returns (the jobs' tensors, _materialize_copy's)."""
    from safetensors import safe_open
    from transformers import core_model_loading as cml

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = sorted(f.keys(), key=cml.dot_natural_key)
        slices = {k: f.get_slice(k) for k in keys}
        jobs = {}
        for k in keys:
            args = (None, slices[k], *_spec(k, slices[k]))
            job = cml.spawn_materialize(*args)
            jobs[k] = copier.add(job, cml.spawn_materialize, args, {})
        results = {}

        def second_pass():
            for i, k in enumerate(order or keys):
                if stop_after is not None and i == stop_after:
                    copier.close()
                results[k] = jobs[k]().clone()  # used at once, as the quantization uses it
                if pause:
                    time.sleep(pause)

        if in_thread:
            worker = threading.Thread(target=second_pass)
            worker.start()
            worker.join()
        else:
            second_pass()
        reference = {k: cml._materialize_copy(slices[k], *_spec(k, slices[k])) for k in keys}
    return results, reference


def _check(results, reference):
    assert results.keys() == reference.keys()
    for key in reference:
        assert _same(results[key], reference[key]), key


@pytest.mark.parametrize("cast", ["cpu", "gpu"])
@pytest.mark.parametrize("window,slots", [(0, 1), (0, 2), (0, 4), (1, 1), (1, 2), (1, 4), (2, 2), (2, 3), (2, 6)])
def test_copies_bit_equal_to_materialize_copy(checkpoint, window, slots, cast):
    """(On the CPU, cast='gpu' runs the raw bytes and the in-place cast through the fake engine:
    the protocol, not the GPU's arithmetic; that is checked on CUDA below.)"""
    copier = _cpu_copier(window, slots, cast=cast)
    try:
        results, reference = _run(copier, checkpoint)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    planned = _planned(checkpoint)
    assert (copier.planned, copier.copied, copier.fallbacks, copier.reason) == (planned, planned, 0, None)
    assert copier.copied_ahead <= (planned - 1 if window else 0)
    assert not copier._filler.is_alive() and 0 < len(copier.made) <= slots
    assert all(b.buffer is None for b in copier.made)  # unpinned at close
    assert copier.slot_bytes == 160 * 64 * 2


@pytest.mark.parametrize("cast", ["cpu", "gpu"])
def test_copies_made_ahead_while_the_loop_works(checkpoint, cast):
    """With work between the jobs the thread stays ahead and the copies go ahead of the asks; the
    fake engine copies late, so a buffer refilled before its copy was out would corrupt a tensor."""
    copier = _cpu_copier(window=2, slots=3, delay=0.002, cast=cast)
    try:
        results, reference = _run(copier, checkpoint, pause=0.004)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert copier.copied == _planned(checkpoint) and copier.copied_ahead > 0 and copier.reason is None


def test_out_of_order_takes_the_original_path(checkpoint):
    from safetensors import safe_open
    from transformers import core_model_loading as cml

    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        keys = sorted(f.keys(), key=cml.dot_natural_key)
    order = list(keys)
    i, j = order.index("model.layers.1.mlp.gate_proj.weight"), order.index("model.layers.1.mlp.up_proj.weight")
    order[i], order[j] = order[j], order[i]
    copier = _cpu_copier(window=1, slots=2)
    try:
        results, reference = _run(copier, checkpoint, order=order)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert "asked" in copier.reason and 0 < copier.copied < _planned(checkpoint)
    assert not copier._filler.is_alive()


def test_failed_staging_of_one_tensor(checkpoint):
    from solution.load_pinned import PinnedCopier

    def stage(entry, buffer):
        if entry.index == 4:
            raise ValueError("unexpected slice")
        return PinnedCopier._stage(entry, buffer)

    copier = _cpu_copier(window=1, slots=2, stage=stage)
    try:
        results, reference = _run(copier, checkpoint)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    planned = _planned(checkpoint)
    assert (copier.copied, copier.fallbacks, copier.reason) == (planned - 1, 1, None)


def test_broken_staging_thread(checkpoint):
    def pin(nbytes):
        raise RuntimeError("no pinned memory")

    copier = _cpu_copier(window=1, slots=2, pin=pin)
    try:
        results, reference = _run(copier, checkpoint)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert copier.copied == 0 and copier.fallbacks == _planned(checkpoint) and "staging thread" in copier.reason


def test_non_default_stream_takes_the_original_path(checkpoint):
    copier = _cpu_copier(window=1, slots=2, default_stream=False)
    try:
        results, reference = _run(copier, checkpoint)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert copier.copied == 0 and "default stream" in copier.reason


def test_closed_mid_way(checkpoint):
    copier = _cpu_copier(window=2, slots=3)
    try:
        results, reference = _run(copier, checkpoint, stop_after=9)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert 0 < copier.copied < _planned(checkpoint) and not copier._filler.is_alive()
    assert all(b.buffer is None for b in copier.made)


def test_jobs_from_another_thread_take_the_original_path(checkpoint):
    copier = _cpu_copier(window=1, slots=2)
    try:
        results, reference = _run(copier, checkpoint, in_thread=True)
    finally:
        copier.close()
        copier.engine.stop()
    _check(results, reference)
    assert copier.copied == 0 and copier._filler is None


def test_plan_closes_when_the_second_pass_begins(checkpoint):
    from concurrent.futures import ThreadPoolExecutor

    from safetensors import safe_open
    from transformers import core_model_loading as cml

    copier = _cpu_copier(window=0, slots=2)
    try:
        with safe_open(str(checkpoint), framework="pt", device="cpu") as f, ThreadPoolExecutor(1) as pool:
            first = f.get_slice("model.layers.0.mlp.up_proj.weight")
            threaded = (pool, first, "cpu", torch.float16)  # an async load: a Future, left alone
            future = cml.spawn_materialize(*threaded)
            assert copier.add(future, cml.spawn_materialize, threaded, {}) is future
            future.result()
            args = (None, first, "cpu", torch.float16)
            job = copier.add(cml.spawn_materialize(*args), cml.spawn_materialize, args, {})
            assert len(copier._entries) == 1
            assert _same(job(), cml._materialize_copy(first, "cpu", torch.float16))
            late = cml.spawn_materialize(*args)
            assert copier.add(late, cml.spawn_materialize, args, {}) is late  # the plan is closed
    finally:
        copier.close()
        copier.engine.stop()
    assert copier.copied == copier.planned == 1


def test_switch_and_hook(monkeypatch):
    from solution import load_pinned, load_stream

    monkeypatch.delenv("SOUP_SOLUTION_LOAD_PINNED", raising=False)
    assert not load_pinned.enabled() and load_pinned.new_copier("cpu") is None
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "0")
    assert not load_pinned.enabled()
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "1")
    assert load_pinned.enabled()
    calls = []

    def spawn(thread_pool, tensor, device=None, dtype=None, sharding_op=None, tensor_idx=None):
        calls.append(tensor)
        return lambda: tensor

    hooked = load_stream._spawner(spawn)
    monkeypatch.setitem(load_stream._ACTIVE, "load", None)
    job = hooked(None, "t", "cpu", None)
    assert job() == "t" and calls == ["t"]  # no load in progress: the loop's own job


# ---- CUDA: the loads ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=384, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(4.0)
        model.model.layers[1].mlp.down_proj.weight[0, :8] = torch.tensor(
            [1e5, -7e4, 65520.0, 1e-8, 2.0 ** -25, 3 * 2.0 ** -26, -0.0, 1.0 + 2.0 ** -7])  # fp16 edge cases
    path = tmp_path_factory.mktemp("tiny_qwen2_load_pinned")
    model.to(torch.bfloat16).save_pretrained(path)  # bf16 on disk, loaded in fp16, as the task's checkpoint
    return path


@pytest.fixture
def patched(monkeypatch):
    """The patches of apply() that take part in the setup, the copies switched on, a fake Soup setup."""
    import peft
    from soup_cli.trainer import sft as soup_sft
    from transformers import core_model_loading, modeling_utils

    from solution import fused_ce, head_stream, load_pinned, load_stream, weight_stream
    from solution.unpad import patch_qwen2_attention

    cls = soup_sft.SFTTrainerWrapper
    monkeypatch.setenv("SOUP_SOLUTION_HEAD_ROWS", "96")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "1")
    monkeypatch.setattr(cls, "_setup_transformers", lambda self: self._fake_setup())
    monkeypatch.setattr(cls, "_soup_head_stream", False, raising=False)
    monkeypatch.setattr(cls, "_soup_weight_stream", False, raising=False)
    prepare = peft.prepare_model_for_kbit_training
    while hasattr(prepare, "_soup_original"):
        prepare = prepare._soup_original
    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", prepare)
    monkeypatch.setitem(load_stream._STATE, "layers", False)
    monkeypatch.setitem(load_stream._STATE, "embedding", False)
    for module, name in ((modeling_utils, "caching_allocator_warmup"),
                         (modeling_utils, "convert_and_load_state_dict_in_model"),
                         (core_model_loading, "get_device"), (core_model_loading, "set_param_for_module"),
                         (core_model_loading, "spawn_materialize")):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setitem(load_pinned.LAST, "copier", None)
    fused_ce.patch_qwen2_causal_lm()
    patch_qwen2_attention()
    head_stream.patch_head_stream()
    weight_stream.patch_weight_stream(slots=2)
    assert getattr(core_model_loading.spawn_materialize, "_soup_load_stream", False)
    return cls


def _load(path):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    return AutoModelForCausalLM.from_pretrained(path, quantization_config=quant, dtype=torch.float16,
                                                device_map={"": 0})


def _snapshot(model):
    """Every tensor that defines the loaded model, on the CPU: parameters and buffers (4-bit weights
    as their packed data, the embedding from head_stream.py's host copy when it holds it) and each
    4-bit weight's quant state (absmax, code, offset, the nested state) with its fields."""
    import bitsandbytes as bnb

    from solution.head_stream import streamer_for

    out = {}
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        streamer = streamer_for(tensor)
        value = streamer.host if streamer is not None else tensor.data
        out[name] = value.detach().to("cpu", copy=True)
    for name, module in model.named_modules():
        if not isinstance(module, bnb.nn.Linear4bit):
            continue
        qs = module.weight.quant_state
        out[f"{name}.quant_state"] = (tuple(qs.shape), qs.dtype, qs.blocksize, qs.quant_type, qs.nested)
        out[f"{name}.absmax"] = qs.absmax.to("cpu", copy=True)
        out[f"{name}.code"] = qs.code.to("cpu", copy=True)
        if qs.nested:
            out[f"{name}.offset"] = qs.offset.to("cpu", copy=True)
            out[f"{name}.state2"] = (qs.state2.blocksize, qs.state2.dtype, qs.state2.quant_type, qs.state2.nested)
            out[f"{name}.state2.absmax"] = qs.state2.absmax.to("cpu", copy=True)
            out[f"{name}.state2.code"] = qs.state2.code.to("cpu", copy=True)
    return out


def _compare(ours, theirs):
    assert ours.keys() == theirs.keys()
    differ = [k for k in theirs if (not _same(ours[k], theirs[k]) if torch.is_tensor(theirs[k]) else ours[k] != theirs[k])]
    assert not differ, differ[:10]


def _release(model):
    """Gives back the pinned embedding load_stream.py's moves made for a model loaded in a fake setup
    (the layers' host buffers go with the model)."""
    from solution.head_stream import streamer_for

    streamer = streamer_for(model.get_input_embeddings().weight)
    if streamer is not None:
        streamer.restore()


def _load_in_setup(cls, path, inspect):
    """Soup's setup with only the load (load_stream.py handles it): the load's GPU peak and
    `inspect(model)` of the loaded model, inside the setup."""
    from solution import load_stream

    seen = {}

    def fake_setup(self):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        model = _load(path)
        torch.cuda.synchronize()
        seen["seconds"] = time.perf_counter() - started
        seen["peak"] = torch.cuda.max_memory_allocated() - before
        seen["result"] = inspect(model)
        load_stream.LAST["load"] = None  # it holds the model
        _release(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None  # nothing for the post-setup hooks of the streamers

    wrapper = cls.__new__(cls)
    wrapper._fake_setup = fake_setup.__get__(wrapper)
    cls._setup_transformers(wrapper)
    return seen


@cuda
def test_gpu_cast_equals_cpu_cast_for_every_bf16_value():
    """SOUP_SOLUTION_LOAD_PINNED_CAST=gpu: the raw bf16 bytes copied into the fp16 tensor's memory
    and cast there in place on the GPU (the copier's own _copy_into) give the bytes of the CPU cast
    the original `.to()` runs, for all 65,536 bf16 values, at several alignments on both sides (the
    vector loops and their tails). NaN stays NaN; its payload is not compared (NaN weights do not
    train)."""
    from solution.load_pinned import PinnedCopier, _Entry

    values = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    for offset in (0, 1, 3):
        src = values[offset:]
        cpu = src.to(torch.float16)
        nan = torch.isnan(src.float())
        for shift in (0, 1):
            entry = _Entry(0, None, tuple(src.shape), torch.bfloat16, torch.float16, torch.bfloat16, None)
            entry.staged = src.clone().pin_memory()
            out = torch.empty(src.numel() + shift, dtype=torch.float16, device="cuda")[shift:]
            PinnedCopier._copy_into(out, entry)
            gpu = out.cpu()
            assert torch.equal(cpu.view(torch.int16)[~nan], gpu.view(torch.int16)[~nan]), (offset, shift)
            assert torch.isnan(gpu[nan].float()).all() and torch.isnan(cpu[nan].float()).all()
    assert int(torch.isnan(values.float()).sum()) == 254 and int((cpu.float().isinf()).sum()) > 2


@cuda
@pytest.mark.parametrize("window,cast", [(0, "cpu"), (1, "cpu"), (2, "cpu"), (0, "gpu"), (1, "gpu")])
def test_tiny_load_bit_equal_and_peak(patched, tiny_dir, monkeypatch, window, cast):
    from solution import load_pinned

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_WINDOW", str(window))
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_SLOTS", "2" if window == 2 else "4")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", cast)
    ours = _load_in_setup(patched, tiny_dir, _snapshot)
    copier = load_pinned.LAST["copier"]
    planned = LAYERS * 12 + 1  # 7 weights, 3 biases and 2 norms per layer, the final norm
    assert copier is not None and (copier.planned, copier.copied, copier.fallbacks, copier.reason) == \
        (planned, planned, 0, None) and copier.cast == cast
    if window:
        assert copier.copied_ahead > 0

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "0")  # the same setup without the copies
    load_pinned.LAST["copier"] = None
    without = _load_in_setup(patched, tiny_dir, _snapshot)
    assert load_pinned.LAST["copier"] is None
    _compare(ours["result"], without["result"])

    plain = _load(tiny_dir)  # outside Soup's setup: transformers' own load, everything on the GPU
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    _compare(ours["result"], theirs)

    largest = 384 * HIDDEN * 2  # an MLP weight in fp16
    if window == 0:
        assert ours["peak"] <= without["peak"]
    else:
        assert ours["peak"] <= without["peak"] + window * (-(-largest // 512) * 512)


@cuda
@pytest.mark.parametrize("cast", ["cpu", "gpu"])
def test_tiny_training_step_is_the_same(patched, tiny_dir, monkeypatch, cast):
    """The whole setup (load with the copies, kbit preparation, peft, the streamers adopting the
    loaded weights) gives the same training step bit for bit as without the copies."""
    import peft
    from peft import LoraConfig, get_peft_model

    from solution import load_pinned
    from solution.head_stream import streamer_for
    from solution.weight_stream import streamer_of

    def setup():
        def fake_setup(self):
            model = peft.prepare_model_for_kbit_training(_load(tiny_dir), use_gradient_checkpointing=False)
            torch.manual_seed(1)
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                                     target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            gen = torch.Generator().manual_seed(2)
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.data.copy_(torch.randn(param.shape, generator=gen) * 0.02)
            self.model = model

        wrapper = patched.__new__(patched)
        wrapper._fake_setup = fake_setup.__get__(wrapper)
        patched._setup_transformers(wrapper)
        return wrapper.model

    def step(model):
        gen = torch.Generator().manual_seed(3)
        lengths = (40, 31, 17)
        ids = torch.randint(0, VOCAB, (3, max(lengths)), generator=gen).cuda()
        mask = (torch.arange(ids.shape[1])[None, :] < torch.tensor(lengths)[:, None]).long().cuda()
        labels = ids.masked_fill(mask == 0, -100)
        labels[:, :5] = -100
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        (loss * 1024.0).backward()
        return [loss.detach()] + [p.grad.clone() for _, p in model.named_parameters() if p.requires_grad]

    def cleanup(model):
        streamer = streamer_of(model)
        if streamer is not None:
            streamer.restore()
        head = streamer_for(model.get_base_model().get_input_embeddings().weight)
        if head is not None:
            head.restore()

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", cast)
    model = setup()
    copier = load_pinned.LAST["copier"]
    assert copier is not None and copier.copied == copier.planned > 0 and copier.reason is None
    assert copier.cast == cast
    assert streamer_of(model) is not None  # the weight streamer adopted the layers loaded through the copies
    ours = step(model)
    cleanup(model)
    del model

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "0")
    reference = setup()
    theirs = step(reference)
    cleanup(reference)
    assert len(ours) == len(theirs) and all(torch.equal(a, b) for a, b in zip(ours, theirs))


@cuda
def test_loads_outside_soup_setup_are_untouched(patched, tiny_dir):
    from solution import load_pinned

    load_pinned.LAST["copier"] = None
    model = _load(tiny_dir)
    assert load_pinned.LAST["copier"] is None
    del model


@cuda
def test_side_stream_current_takes_the_original_path(tiny_dir):
    """A job asked while the current stream is not the default one (bitsandbytes' kernels run on the
    legacy default stream) is the original one, and so is every later job."""
    from safetensors import safe_open
    from transformers import core_model_loading as cml

    from solution.load_pinned import PinnedCopier

    device = torch.device("cuda", torch.cuda.current_device())
    copier = PinnedCopier(device, window=1, slots=2, log=lambda *_: None)
    with safe_open(str(tiny_dir / "model.safetensors"), framework="pt", device="cpu") as f:
        keys = sorted(f.keys(), key=cml.dot_natural_key)[:6]
        slices = [f.get_slice(k) for k in keys]
        jobs = []
        for s in slices:
            args = (None, s, device.index, torch.float16)
            jobs.append(copier.add(cml.spawn_materialize(*args), cml.spawn_materialize, args, {}))
        try:
            side = torch.cuda.Stream(device)
            with torch.cuda.stream(side):
                got = [job() for job in jobs]
            side.synchronize()
        finally:
            copier.close()
        for s, tensor in zip(slices, got):
            assert _same(tensor, cml._materialize_copy(s, device.index, torch.float16))
    assert copier.copied == 0 and "default stream" in copier.reason


@pytest.fixture(scope="module")
def real_dir():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REAL)  # what the harness's warm-up downloads
    except Exception as exc:  # noqa: BLE001 - offline
        pytest.skip(f"{REAL} not available: {exc!r}")


@cuda
@pytest.mark.parametrize("cast", ["cpu", "gpu"])
def test_real_checkpoint_tensor_by_tensor(real_dir, cast):
    """Every tensor of the real checkpoint through the copies (window 1, 4 buffers) against
    transformers' _materialize_copy in fp16 as the loop asks for them: to the GPU, the embedding to
    the host (load_stream.py's device for it: the loop's own job)."""
    from safetensors import safe_open
    from transformers import core_model_loading as cml

    from solution.load_pinned import PinnedCopier

    device = torch.device("cuda", torch.cuda.current_device())
    handles = [safe_open(f, framework="pt", device="cpu") for f in sorted(glob.glob(os.path.join(real_dir, "*.safetensors")))]
    slices = {k: h.get_slice(k) for h in handles for k in h.keys()}
    keys = sorted(slices, key=cml.dot_natural_key)
    where = {k: "cpu" if k == "model.embed_tokens.weight" else device.index for k in keys}
    copier = PinnedCopier(device, window=1, slots=4, cast=cast, log=print)
    jobs = []
    for k in keys:
        args = (None, slices[k], where[k], torch.float16)
        jobs.append(copier.add(cml.spawn_materialize(*args), cml.spawn_materialize, args, {}))
    differ = []
    started = time.perf_counter()
    try:
        for k, job in zip(keys, jobs):
            ours = job()
            theirs = cml._materialize_copy(slices[k], where[k], torch.float16)
            if not _same(ours, theirs):
                differ.append(k)
            del ours, theirs
    finally:
        copier.close()
        del jobs, slices
        for h in handles:
            h.__exit__(None, None, None)
    print(f"338 tensors through the copies (cast on the {cast}) and _materialize_copy in "
          f"{time.perf_counter() - started:.2f} s")
    assert not differ, differ[:10]
    assert len(keys) == 338 and copier.planned == copier.copied == 337 and copier.fallbacks == 0
    assert copier.reason is None and copier.slot_bytes == 8960 * 1536 * 2


@cuda
@pytest.mark.parametrize("cast", ["cpu", "gpu"])
def test_real_load_bit_equal(patched, real_dir, monkeypatch, cast):
    """The real model loaded in Soup's setup with load_stream.py's moves and the copies (default
    window and buffers) against transformers' own load: every packed weight, absmax, quant state,
    norm, bias and the embedding bit for bit."""
    from solution import load_pinned

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", cast)
    ours = _load_in_setup(patched, real_dir, _snapshot)
    copier = load_pinned.LAST["copier"]
    assert copier is not None and copier.copied == copier.planned == 337 and copier.reason is None
    assert copier.cast == cast
    plain = _load(real_dir)
    started = time.perf_counter()
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    torch.cuda.empty_cache()
    _compare(ours["result"], theirs)
    print(f"real load with the copies (cast on the {cast}) {ours['seconds']:.2f} s (GPU peak "
          f"{ours['peak'] / 2**20:.0f} MiB); "
          f"snapshot {time.perf_counter() - started:.2f} s")
