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
    assert load_pinned.enabled()  # on by default since the night integration
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", "0")
    assert not load_pinned.enabled() and load_pinned.new_copier("cpu") is None
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


def _tiny_checkpoint(path, edge_values):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=384, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(4.0)
        if edge_values:  # values fp16 overflows (inf), flushes and rounds: the load must keep their bits too
            model.model.layers[1].mlp.down_proj.weight[0, :8] = torch.tensor(
                [1e5, -7e4, 65520.0, 1e-8, 2.0 ** -25, 3 * 2.0 ** -26, -0.0, 1.0 + 2.0 ** -7])
    model.to(torch.bfloat16).save_pretrained(path)  # bf16 on disk, loaded in fp16, as the task's checkpoint
    return path


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return _tiny_checkpoint(tmp_path_factory.mktemp("tiny_qwen2_load_pinned"), edge_values=True)


@pytest.fixture(scope="module")
def tiny_train_dir(tmp_path_factory):
    """Without the overflowing values (an inf weight makes every step NaN)."""
    return _tiny_checkpoint(tmp_path_factory.mktemp("tiny_qwen2_load_pinned_train"), edge_values=False)


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


def _with_fill(model):
    """_snapshot of the model and what load_stream.py's EMBED_ASYNC did in its load."""
    from solution import load_stream

    load = load_stream.LAST["load"]
    fill = load.embedding_fill if load is not None else None
    outcome = dict(fill[3]) if fill else None
    return _snapshot(model), outcome


@cuda
@pytest.mark.parametrize("copies", ["1", "0"])
def test_tiny_load_embedding_fill_bit_equal(patched, tiny_dir, monkeypatch, copies):
    """SOUP_SOLUTION_EMBED_ASYNC=1 (load_stream.py): the embedding cast into its pinned buffer by a
    thread gives the load without it bit for bit (and transformers' own load), with and without the
    pinned copies of the other tensors, and no higher GPU peak."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED", copies)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", "0")
    without = _load_in_setup(patched, tiny_dir, _with_fill)  # first: the process's first load peaks higher
    assert without["result"][1] is None
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", "1")
    ours = _load_in_setup(patched, tiny_dir, _with_fill)
    snapshot, outcome = ours["result"]
    assert outcome is not None and outcome["error"] is None and outcome["seconds"] is not None
    _compare(snapshot, without["result"][0])
    plain = _load(tiny_dir)
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    _compare(snapshot, theirs)
    assert ours["peak"] <= without["peak"]


def _codes(model):
    """_snapshot and the identities of the 4-bit quant states' code tables."""
    import bitsandbytes as bnb

    ids = [id(m.weight.quant_state.code) for m in model.modules() if isinstance(m, bnb.nn.Linear4bit)]
    return _snapshot(model), ids


@cuda
def test_tiny_load_code_table_bit_equal(patched, tiny_dir, monkeypatch):
    """SOUP_SOLUTION_LOAD_CODE_TABLE=1 (load_stream.py): every quant state gets the same code values
    as bitsandbytes' own get_4bit_type gives it, in a tensor of its own; the function is bitsandbytes'
    own again after the load."""
    import bitsandbytes.functional as functional

    plain = functional.get_4bit_type
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_CODE_TABLE", "0")
    without = _load_in_setup(patched, tiny_dir, _codes)  # first: the process's first load peaks higher
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_CODE_TABLE", "1")
    ours = _load_in_setup(patched, tiny_dir, _codes)
    assert functional.get_4bit_type is plain
    snapshot, ids = ours["result"]
    assert len(ids) == LAYERS * 7 and len(set(ids)) == len(ids)
    _compare(snapshot, without["result"][0])
    assert ours["peak"] <= without["peak"] + 512  # the kept table: 16 floats in one 512-byte block


def _with_reads(model):
    """_snapshot of the model and how many tensors load_pinned.py's thread read from the file."""
    from solution import load_pinned

    copier = load_pinned.LAST["copier"]
    return _snapshot(model), (copier.read_from_file, copier.planned, copier.copied, copier.reason)


def _pread_on(monkeypatch):
    from transformers import modeling_utils

    from solution import load_stream

    monkeypatch.setattr(modeling_utils, "safe_open", modeling_utils.safe_open)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "1")
    load_stream.patch_loader()  # the safe_open hook (installed when the switch is on)
    assert getattr(modeling_utils.safe_open, "_soup_load_stream", False)


@cuda
def test_tiny_load_pread_bit_equal(patched, tiny_dir, monkeypatch):
    """SOUP_SOLUTION_LOAD_PREAD=1: every planned tensor staged with preadv from its file; the load is
    the one with the mmap copies and transformers' own, bit for bit."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    without = _load_in_setup(patched, tiny_dir, _with_reads)
    assert without["result"][1][0] == 0
    _pread_on(monkeypatch)
    ours = _load_in_setup(patched, tiny_dir, _with_reads)
    snapshot, (read, planned, copied, reason) = ours["result"]
    assert reason is None and planned == copied == read == LAYERS * 12 + 1
    _compare(snapshot, without["result"][0])
    plain = _load(tiny_dir)
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    _compare(snapshot, theirs)
    assert ours["peak"] <= without["peak"]


@cuda
def test_tiny_load_pread_not_for_the_cpu_cast(patched, tiny_dir, monkeypatch):
    """With the CPU cast the thread converts from the slice: nothing is read from the file."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "cpu")
    _pread_on(monkeypatch)
    ours = _load_in_setup(patched, tiny_dir, _with_reads)
    assert ours["result"][1][0] == 0 and ours["result"][1][3] is None


@cuda
def test_tiny_load_embedding_fill_thread_fails(patched, tiny_dir, monkeypatch):
    """A fill thread that fails: the join fills the buffer in the main thread, the same bytes."""
    from solution import load_stream

    original = load_stream._Load.embedding_job

    class Flaky:
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def __getitem__(self, key):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("fill failed on purpose")
            return self.inner[key]

    def embedding_job(self, job, spawn, args, kwargs):
        made = original(self, job, spawn, args, kwargs)
        if made is None:
            return None
        made_job, source, shape, dtype = made.args
        return type(made)(made.func, made_job, Flaky(source), shape, dtype)

    monkeypatch.setattr(load_stream._Load, "embedding_job", embedding_job)
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", "1")
    ours = _load_in_setup(patched, tiny_dir, _with_fill)
    snapshot, outcome = ours["result"]
    assert outcome is not None and outcome["error"] is None  # cleared by the main thread's fill
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", "0")
    without = _load_in_setup(patched, tiny_dir, _with_fill)
    _compare(snapshot, without["result"][0])


@cuda
@pytest.mark.parametrize("cast", ["cpu", "gpu"])
def test_tiny_training_step_is_the_same(patched, tiny_train_dir, monkeypatch, cast):
    """The whole setup (load with the copies, kbit preparation, peft, the streamers adopting the
    loaded weights) gives the same training step bit for bit as without the copies."""
    import peft
    from peft import LoraConfig, get_peft_model

    from solution import load_pinned
    from solution.head_stream import streamer_for
    from solution.weight_stream import streamer_of

    def setup():
        def fake_setup(self):
            model = peft.prepare_model_for_kbit_training(_load(tiny_train_dir), use_gradient_checkpointing=False)
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
    assert torch.isfinite(ours[0]) and len(ours) == len(theirs)
    assert all(torch.equal(a, b) for a, b in zip(ours, theirs))


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
def test_real_load_pread_bit_equal(patched, real_dir, monkeypatch):
    """The real model with SOUP_SOLUTION_LOAD_PREAD=1 (and the embedding filled by a thread):
    all 337 GPU tensors read from the file, the load bit for bit transformers' own."""
    _pread_on(monkeypatch)
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", "1")
    ours = _load_in_setup(patched, real_dir, _with_reads)
    snapshot, (read, planned, copied, reason) = ours["result"]
    assert reason is None and read == planned == copied == 337
    plain = _load(real_dir)
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    torch.cuda.empty_cache()
    _compare(snapshot, theirs)
    print(f"real load with preadv {ours['seconds']:.2f} s (GPU peak {ours['peak'] / 2**20:.0f} MiB)")


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


# ---- SOUP_SOLUTION_LOAD_EARLY (load_early.py) ---------------------------------------------------------


def _early_read(monkeypatch, path, chunk_bytes=64 << 10, fail=False, read_from=None):
    """load_early.py's reading thread started on the checkpoint in `path` (small chunks: several per
    file), its switch on, the safe_open hook installed; returns the module's state."""
    from transformers import modeling_utils

    from solution import load_early, load_stream

    monkeypatch.setattr(modeling_utils, "safe_open", modeling_utils.safe_open)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "1")
    monkeypatch.setattr(load_early, "CHUNK_BYTES", chunk_bytes)
    state = {"thread": None, "files": {}, "error": None, "seconds": None, "bytes": 0, "used": 0, "released": False}
    monkeypatch.setattr(load_early, "_STATE", state)
    files = [load_early._plan(p) for p in sorted(glob.glob(os.path.join(str(read_from or path), "*.safetensors")))]
    assert files and all(len(f.chunks) > 1 for f in files)
    state["files"] = {f.path: f for f in files}
    if fail:
        monkeypatch.setattr(load_early, "_read_into", None, raising=False)

        def broken(files_, device):
            for f in files_:
                f.failed = True
                for c in f.chunks:
                    c.ready.set()

        target = broken
    else:
        target = load_early._read
    state["thread"] = threading.Thread(target=target, args=(files, torch.cuda.current_device()), daemon=True)
    state["thread"].start()
    load_stream.patch_loader()  # the safe_open hook (installed when the switch is on)
    assert getattr(modeling_utils.safe_open, "_soup_load_stream", False)
    return state


def _with_early(model):
    from solution import load_pinned

    copier = load_pinned.LAST["copier"]
    return _snapshot(model), (copier.copied_early, copier.planned, copier.copied, copier.fallbacks, copier.reason)


@cuda
@pytest.mark.parametrize("embed_async", ["1", "0"])
def test_tiny_load_early_bit_equal(patched, tiny_dir, monkeypatch, embed_async):
    """SOUP_SOLUTION_LOAD_EARLY=1: every planned tensor (and the embedding's cast) from the checkpoint
    read ahead into pinned memory; the load is the plain one bit for bit; the memory released after."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_EMBED_ASYNC", embed_async)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_early)
    assert without["result"][1][0] == 0
    state = _early_read(monkeypatch, tiny_dir)
    ours = _load_in_setup(patched, tiny_dir, _with_early)
    snapshot, (early, planned, copied, fallbacks, reason) = ours["result"]
    assert reason is None and fallbacks == 0 and planned == copied == early == LAYERS * 12 + 1
    assert state["used"] == planned + (1 if embed_async == "1" else 0) and state["error"] is None
    assert state["released"]
    _compare(snapshot, without["result"][0])
    plain = _load(tiny_dir)
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    _compare(snapshot, theirs)


@cuda
def test_tiny_load_early_failed_read(patched, tiny_dir, monkeypatch):
    """A read that failed: every tensor takes the loop's own job (no early bytes), the same load."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_early)
    _early_read(monkeypatch, tiny_dir, fail=True)
    ours = _load_in_setup(patched, tiny_dir, _with_early)
    snapshot, (early, planned, copied, fallbacks, reason) = ours["result"]
    assert early == 0 and fallbacks == planned
    _compare(snapshot, without["result"][0])


def test_early_files_from_config_and_cache(tmp_path, monkeypatch):
    """The checkpoint files of Soup's config: a hub repo's snapshot (refs/main), a sharded index, a local
    directory; nothing without a config or a snapshot."""
    from solution import load_early

    hub = tmp_path / "hub"
    snap = hub / "models--Org--Model" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (hub / "models--Org--Model" / "refs").mkdir()
    (hub / "models--Org--Model" / "refs" / "main").write_text("abc123\n")
    (snap / "model.safetensors").write_bytes(b"")
    config = tmp_path / "task.yaml"
    config.write_text("# x\nbase: Org/Model\ntask: sft\n")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    base = load_early._config_base(["soup", "train", "--config", str(config), "--yes"])
    assert base == "Org/Model"
    assert load_early.checkpoint_files(base) == [str(snap / "model.safetensors")]
    (snap / "model.safetensors.index.json").write_text('{"weight_map": {"a": "m-2.safetensors", "b": "m-1.safetensors", "c": "m-2.safetensors"}}')
    assert load_early.checkpoint_files(base) == [str(snap / "m-1.safetensors"), str(snap / "m-2.safetensors")]
    assert load_early.checkpoint_files(str(snap)) == [str(snap / "m-1.safetensors"), str(snap / "m-2.safetensors")]
    assert load_early.checkpoint_files("Org/Other") == []  # no snapshot in the cache
    assert load_early._config_base(["soup", "train"]) is None
    monkeypatch.delenv("SOUP_SOLUTION_LOAD_EARLY", raising=False)
    assert not load_early.enabled()


@cuda
def test_tiny_load_early_other_files_dropped(patched, tiny_dir, monkeypatch, tmp_path):
    """The files read ahead are not the ones the load opens (another copy of the checkpoint): the
    buffers are dropped when the load starts, every tensor is staged the usual way, the same load."""
    import shutil

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_early)
    other = tmp_path / "copy"
    shutil.copytree(tiny_dir, other)
    state = _early_read(monkeypatch, tiny_dir, read_from=other)
    ours = _load_in_setup(patched, tiny_dir, _with_early)
    snapshot, (early, planned, copied, fallbacks, reason) = ours["result"]
    assert early == 0 and state["used"] == 0 and state["released"] and reason is None
    _compare(snapshot, without["result"][0])


# ---- SOUP_SOLUTION_LOAD_SPEC (load_spec.py) -----------------------------------------------------------


def _with_spec(model):
    from solution import load_pinned

    copier = load_pinned.LAST["copier"]
    return _snapshot(model), (copier.handed_over, copier.planned, copier.copied_early, copier.fallbacks, copier.reason)


def _spec_on(monkeypatch, half):
    from solution import load_spec

    monkeypatch.setitem(load_spec._STATE, "spec", None)
    monkeypatch.setitem(load_spec._STATE, "handover", {})
    for key, value in (("made", 0), ("handed", 0), ("dropped", None), ("waited", 0.0), ("first_wait", None)):
        monkeypatch.setitem(load_spec.LAST, key, value)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_SPEC", "1")
    spec = load_spec.start(half=half)
    assert spec is not None and len(spec.items) == LAYERS * 7
    return load_spec


@cuda
def test_tiny_load_spec_bit_equal(patched, tiny_dir, monkeypatch):
    """SOUP_SOLUTION_LOAD_SPEC=1: every 4-bit weight quantized ahead by the thread and handed over; the
    packed weights, every quant-state tensor and field, and every other tensor are the plain load's."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_spec)
    _early_read(monkeypatch, tiny_dir)
    load_spec = _spec_on(monkeypatch, torch.float16)
    ours = _load_in_setup(patched, tiny_dir, _with_spec)
    snapshot, (handed, planned, early, fallbacks, reason) = ours["result"]
    assert reason is None and fallbacks == 0 and handed == LAYERS * 7, load_spec.LAST
    assert early == planned - handed and load_spec.LAST["dropped"] is None
    _compare(snapshot, without["result"][0])
    plain = _load(tiny_dir)
    theirs = _snapshot(plain)
    del plain
    gc.collect()
    _compare(snapshot, theirs)
    assert not load_spec._STATE["handover"] and load_spec._STATE["spec"].stop


@cuda
def test_tiny_load_spec_mismatch_falls_back(patched, tiny_dir, monkeypatch):
    """Speculated in another dtype than the load's: dropped at the first weight, the loop does it all."""
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_spec)
    _early_read(monkeypatch, tiny_dir)
    load_spec = _spec_on(monkeypatch, torch.bfloat16)  # the load is in fp16
    ours = _load_in_setup(patched, tiny_dir, _with_spec)
    snapshot, (handed, planned, early, fallbacks, reason) = ours["result"]
    assert handed == 0 and "not the speculated" in (load_spec.LAST["dropped"] or "")
    _compare(snapshot, without["result"][0])


@cuda
def test_spec_quantize_4bit_equals_bitsandbytes():
    """load_spec._quantize_4bit with the table made once: bitsandbytes' quantize_4bit bit for bit (packed
    weight, absmax, offset, nested state, every field), also when the default dtype is fp16 meanwhile."""
    import bitsandbytes as bnb

    from solution import load_spec

    if bnb.__version__ != "0.50.2":
        pytest.skip("load_spec follows bitsandbytes 0.50.2")
    code = bnb.functional.get_4bit_type("nf4", device=torch.device("cuda", torch.cuda.current_device()))
    gen = torch.Generator(device="cuda").manual_seed(3)
    for shape in ((256, 1536), (8960, 1536), (1536, 8960)):
        a = (torch.randn(shape, device="cuda", generator=gen) * 0.05).half()
        w_ref, s_ref = bnb.functional.quantize_4bit(a, blocksize=64, compress_statistics=True, quant_type="nf4",
                                                   quant_storage=torch.uint8)
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)  # as transformers' local_torch_dtype on the main thread
        try:
            w, s = load_spec._quantize_4bit(a, code)
        finally:
            torch.set_default_dtype(previous)
        assert torch.equal(w, w_ref)
        for name in ("absmax", "code", "offset"):
            x, y = getattr(s, name), getattr(s_ref, name)
            assert x.dtype == y.dtype and torch.equal(x, y), name
        assert (s.shape, s.dtype, s.blocksize, s.quant_type, s.nested) == \
            (s_ref.shape, s_ref.dtype, s_ref.blocksize, s_ref.quant_type, s_ref.nested)
        assert torch.equal(s.state2.absmax, s_ref.state2.absmax) and torch.equal(s.state2.code, s_ref.state2.code)
        assert s.state2.blocksize == s_ref.state2.blocksize and s.code is not code


@cuda
def test_tiny_load_early_prepinned_embedding(patched, tiny_dir, monkeypatch):
    """SOUP_SOLUTION_LOAD_PREPIN=1: the embedding's pinned buffer made by load_early.py's thread is the one
    the load uses; the load is the plain one bit for bit."""
    from solution import load_early

    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PINNED_CAST", "gpu")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREAD", "0")
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_EARLY", "0")
    without = _load_in_setup(patched, tiny_dir, _with_early)
    monkeypatch.setenv("SOUP_SOLUTION_LOAD_PREPIN", "1")
    state = _early_read(monkeypatch, tiny_dir)
    state["thread"].join()
    made = state.get("prepinned", {}).get(VOCAB * HIDDEN * 2)
    assert made is not None and made.buffer is not None
    ours = _load_in_setup(patched, tiny_dir, _with_early)
    assert not state.get("prepinned")  # taken by the load's pinner
    _compare(ours["result"][0], without["result"][0])
    assert load_early._STATE is state
