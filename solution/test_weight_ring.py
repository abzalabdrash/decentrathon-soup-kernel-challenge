"""The type ring of weight_stream.py (SOUP_SOLUTION_WEIGHT_RING=type: the attention projections in a
ring of layers, one GPU slot per MLP matrix type) against the layer ring: a Qwen2 with the task's
layer shapes through the solution's layer patches with the night defaults (attention recompute, the
MLP offload, LoRA dropout), forward outputs, input gradients and LoRA gradients bit for bit, with
the GPU kept far behind the CPU at every dequantization; calls out of the training order; load and
wait counters; slots and memory; the switch and its fallbacks; an MLP that reads its matrices
without acquire()."""

import gc

import pytest
import torch

from solution.test_weight_stream import (  # noqa: F401 - layer_shaped_qwen_dir is a fixture
    HIDDEN, SLEEP_CYCLES, STEP_LENGTHS, _apply_layer_patches, _batch, _equal, _inputs, _quiet_gpu,
    _same, _soup_like, _step, layer_shaped_qwen_dir, make_stack)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _ring_step(model, lengths=STEP_LENGTHS):
    """One packed training step through unpad.varlen_model_forward with the MLP offload on: the
    decoder output, the gradient of the first layer's input and every LoRA gradient (cloned)."""
    from solution.offload import offload_saved_tensors
    from solution.unpad import pack_batch, varlen_model_forward

    ids, mask, _ = _batch(lengths=lengths)
    packed, pos, _, spans = pack_batch(ids, mask)
    decoder = model.get_base_model().model
    captured = {}

    def leaf(module, args, output):
        captured["x"] = output.detach().requires_grad_(True)
        return captured["x"]

    gen = torch.Generator().manual_seed(4)
    weight = torch.randn(1, packed.shape[1], decoder.config.hidden_size, generator=gen).cuda()
    model.zero_grad(set_to_none=True)
    handle = decoder.embed_tokens.register_forward_hook(leaf)
    torch.manual_seed(7)
    try:
        with offload_saved_tensors():
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = varlen_model_forward(decoder, packed, pos, spans)
            (hidden.float() * weight).sum().backward()
    finally:
        handle.remove()
    return ([hidden.detach().clone(), captured["x"].grad.clone()]
            + [p.grad.clone() for _, p in model.named_parameters() if p.requires_grad])


def _close(a, b, tol=1e-3):
    return len(a) == len(b) and all(
        (x.float() - y.float()).norm() <= tol * y.float().norm() for x, y in zip(a, b))


@pytest.fixture
def night(monkeypatch):
    """The night defaults that matter here (attention recompute of every layer), a bit-reproducible
    attention backward (one key split: dQ added in a fixed order), the layer patches of apply()."""
    from solution import fused_proj, unpad

    _apply_layer_patches()
    monkeypatch.setitem(fused_proj._RECOMPUTE, "mode", "all")
    monkeypatch.setitem(fused_proj._RECOMPUTE, "check", False)
    monkeypatch.setitem(unpad._STATE, "splits_key", 1)
    monkeypatch.delenv("SOUP_SOLUTION_WEIGHT_RING", raising=False)


@pytest.mark.parametrize("attention,mlp_recompute", [("recompute", False), ("math", False),
                                                     ("recompute", True)])
def test_type_ring_matches_layer_ring(layer_shaped_qwen_dir, night, monkeypatch, attention,
                                      mlp_recompute):
    """Two training steps with each ring equal the resident model's step bit for bit (and so each
    other), with every dequantization of fused_proj.py queued behind a ~3 ms sleep of the GPU: a
    copy into a matrix slot that did not wait for the read before it, or a read that did not wait
    for its copy, changes the result. `math`: SDPA's math backend per sequence (no recompute);
    `mlp_recompute`: SOUP_SOLUTION_MLP_MEMORY=recompute (up dequantized twice in the backward)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution import fused_proj
    from solution.weight_stream import stream_model_weights

    monkeypatch.setitem(fused_proj._MLP_MEMORY, "recompute", mlp_recompute)
    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    sleep = [0]
    rebuild = fused_proj._weights

    def slow_weights(bases, dtype, device, offset=0):
        if sleep[0]:
            torch.cuda._sleep(sleep[0])
        return rebuild(bases, dtype, device, offset)

    monkeypatch.setattr(fused_proj, "_weights", slow_weights)
    attn_rebuild = fused_proj._attn_weights  # q, k, v and o in one launch (SOUP_SOLUTION_CPU_DIET)

    def slow_attn_weights(plan, device):
        if sleep[0]:
            torch.cuda._sleep(sleep[0])
        return attn_rebuild(plan, device)

    monkeypatch.setattr(fused_proj, "_attn_weights", slow_attn_weights)

    def step():
        if attention == "math":
            with sdpa_kernel(SDPBackend.MATH):
                return _ring_step(model)
        return _ring_step(model)

    resident = step()
    exact = _same(resident, step())
    results = {}
    for ring in ("layer", "type"):
        streamer = stream_model_weights(model, slots=2, ring=ring)
        assert streamer is not None and streamer.ring == ring and streamer.ring_note is None
        sleep[0] = SLEEP_CYCLES // 3
        try:
            results[ring] = [step(), step()]
            assert streamer.late_loads == 0
        finally:
            sleep[0] = 0
            streamer.restore()
    compare = _same if exact else _close
    for ring, steps in results.items():
        assert all(compare(resident, got) for got in steps), ring
    if not exact:
        print(f"\nattention backward not reproducible here ({attention}): compared within 1e-3")
    assert compare(resident, step())  # restored


@pytest.mark.parametrize("timing", [True, False])
def test_type_ring_slots_counters_and_waits(layer_shaped_qwen_dir, night, monkeypatch, timing):
    """Every layer's MLP matrices (weight, absmax, second-level absmax) point into their type's one
    slot for good, its attention projections into slot j % 2; per step each MLP matrix is copied
    once per pass and waited for once per pass, nothing is loaded late; the copies and the waits of
    the attention group are timed, those of the MLP matrices with SOUP_SOLUTION_WEIGHT_RING_TIMING."""
    from solution import weight_stream
    from solution.weight_stream import find_decoder_layers, stream_model_weights

    monkeypatch.setitem(weight_stream._RING, "timing", timing)
    monkeypatch.setitem(weight_stream._RING, "stream_timing", True)  # the attention group's
    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    layers = find_decoder_layers(model)
    streamer = stream_model_weights(model, slots=2, ring="type")
    try:
        n = streamer.n
        assert [(g.name, g.count) for g in streamer._groups] == [("attn", 2), ("gate", 1), ("up", 1),
                                                                 ("down", 1)]
        attn, gate, up, down = streamer._groups
        # q, k, v, o: 5.5 M NF4 weights and their scales; gate, up, down: 13.8 M each
        assert 2_800_000 < attn.nbytes < 2_900_000
        assert all(7_000_000 < g.nbytes < 7_200_000 for g in (gate, up, down))
        assert len(streamer._slots) == 5
        assert streamer.ring_bytes == 2 * attn.nbytes + gate.nbytes + up.nbytes + down.nbytes
        assert streamer.ring_bytes < 0.6 * 2 * max(streamer.layer_bytes)

        def inside(t, slot):
            return slot.data_ptr() <= t.data_ptr() < slot.data_ptr() + slot.numel()

        def tensors(module):
            state = module.weight.quant_state
            return module.weight, state.absmax, state.state2.absmax

        for j, layer in enumerate(layers):
            for group, module in zip((gate, up, down), (layer.mlp.gate_proj, layer.mlp.up_proj,
                                                        layer.mlp.down_proj)):
                assert all(inside(t, streamer._slots[group.first]) for t in tensors(module))
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                module = getattr(layer.self_attn, name)
                module = getattr(module, "base_layer", module)
                assert all(inside(t, streamer._slots[attn.first + j % 2]) for t in tensors(module))
        assert streamer.loads == 2 + 3  # set up: attention of layers 0 and 1, layer 0's matrices
        _ring_step(model)
        _ring_step(model)
        stats = streamer.stats()
        assert streamer.late_loads == 0
        assert stats["group_loads"] == {"attn": 2 + 2 * 2 * (n - 2), "gate": 1 + 2 * 2 * (n - 1),
                                        "up": 1 + 2 * 2 * (n - 1), "down": 1 + 2 * 2 * (n - 1)}
        assert stats["matrix_timing"] is timing
        if timing:
            assert stats["copies_timed"] == stats["loads"] - 5
            # per step and pass: one wait per layer for its attention, one per MLP matrix
            assert stats["waits_forward"] == stats["waits_backward"] == 2 * 4 * n
            assert stats["matrix_waits_forward"] == stats["matrix_waits_backward"] == 2 * 3 * n
        else:  # the attention group's copies and waits only
            assert stats["copies_timed"] == stats["group_loads"]["attn"] - 2
            assert stats["waits_forward"] == stats["waits_backward"] == 2 * n
            assert stats["matrix_waits_forward"] == stats["matrix_waits_backward"] == 0
        assert stats["copy_gbps"] > 0.5
        print(f"\ntype ring {streamer.ring_bytes / 2**20:.1f} MiB; copies at {stats['copy_gbps']:.1f} "
              f"GB/s; waits {stats['wait_forward_ms']:.2f} ms forward (matrices "
              f"{stats['matrix_wait_forward_ms']:.2f}), {stats['wait_backward_ms']:.2f} ms backward "
              f"(matrices {stats['matrix_wait_backward_ms']:.2f})")
    finally:
        streamer.restore()


def _settled_requested():
    """(bytes the tensors on the GPU asked for, tolerance): the caching allocator's requested bytes
    (memory_allocated() counts whole cached blocks, up to 1 MiB more per large allocation when the
    rest of a block is too small to split), with the garbage of earlier tests collected first."""
    gc.collect()
    torch.cuda.synchronize()
    requested = torch.cuda.memory_stats().get("requested_bytes.all.current")
    if requested is None:
        return torch.cuda.memory_allocated(), 4 << 20
    return requested, 64 << 10


def test_type_ring_memory(layer_shaped_qwen_dir, night):
    """The type ring holds 2 x 2.84 + 3 x 7.1 MB of slots instead of 2 x 24.1 MB: what the GPU holds
    after the setup drops by the difference. Both rings are measured from the same state (the
    layers restored into tensors of their own by a first streaming)."""
    from solution.weight_stream import stream_model_weights

    model = _soup_like(layer_shaped_qwen_dir)
    _quiet_gpu()
    stream_model_weights(model, slots=2, ring="layer").restore()
    held = {}
    for ring in ("layer", "type"):
        before, tolerance = _settled_requested()
        streamer = stream_model_weights(model, slots=2, ring=ring)
        try:
            held[ring] = (before - _settled_requested()[0], streamer.ring_bytes)
        finally:
            streamer.restore()
    (layer_freed, layer_ring), (type_freed, type_ring) = held["layer"], held["type"]
    assert 48_000_000 < layer_ring < 48_600_000 and 26_800_000 < type_ring < 27_200_000
    saved = type_freed - layer_freed
    assert abs(saved - (layer_ring - type_ring)) <= tolerance, (saved, layer_ring - type_ring)
    print(f"\nring {layer_ring / 2**20:.1f} -> {type_ring / 2**20:.1f} MiB, allocated "
          f"{saved / 2**20:.1f} MiB less")


def test_type_ring_out_of_order(layer_shaped_qwen_dir, night):
    """Calls out of the training order load what they need on the spot: MLPs called directly in
    another order (the fused path, and with 8 tokens bitsandbytes' own 4-bit GEMM, which reads the
    matrices without acquire()), a forward and backward of one MLP alone, an eval forward of the
    whole model; then the training steps equal the reference bit for bit."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from solution.weight_stream import find_decoder_layers, stream_model_weights

    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    layers = find_decoder_layers(model)
    gen = torch.Generator().manual_seed(11)
    xs = [torch.randn(1, 200, HIDDEN, generator=gen).cuda() for _ in layers]
    small = [x[:, :8] for x in xs]
    ids, mask, _ = _batch()

    def mlp_calls():
        with torch.no_grad():
            out = [layers[j].mlp(xs[j]) for j in (3, 1, 2, 0, 0)]
            out += [layers[j].mlp(small[j]) for j in (2, 3, 0)]
        x = xs[1].clone().requires_grad_(True)
        y = layers[1].mlp(x)
        y.float().pow(2).sum().backward()
        out += [y.detach(), x.grad]
        xg = small[2].clone().requires_grad_(True)
        y = layers[2].mlp(xg)
        y.float().pow(2).sum().backward()
        out += [y.detach(), xg.grad]
        return [t.clone() for t in out]

    def evaluate():  # in training mode: the LoRA dropout draws its masks from a fixed seed
        torch.manual_seed(13)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return [model(input_ids=ids, attention_mask=mask).logits.float().clone()]

    def step():
        with sdpa_kernel(SDPBackend.MATH):
            return _ring_step(model)

    calls, logits, reference = mlp_calls(), evaluate(), step()
    streamer = stream_model_weights(model, slots=2, ring="type")
    assert streamer is not None and streamer.ring == "type"
    try:
        assert _same(calls, mlp_calls())
        late = streamer.late_loads
        assert late > 0
        assert _same(reference, step())
        assert _same(logits, evaluate())  # ends with the last layer's matrices in the slots
        assert _same(reference, step())
        assert _same(calls, mlp_calls())
        assert _same(reference, step())
        assert streamer.late_loads > late
    finally:
        streamer.restore()
    assert _same(reference, step())


def test_ring_switch_and_fallbacks(layer_shaped_qwen_dir, night, monkeypatch):
    """SOUP_SOLUTION_WEIGHT_RING: unset, `layer` or anything unknown is today's ring of whole layers
    (2 slots, no MLP matrix registered, fused_proj's MLP plans carry no ring); `type` the type ring;
    `only` declines the type ring with a note; restore() unregisters the matrices."""
    from solution import fused_proj, weight_stream
    from solution.weight_stream import find_decoder_layers, ring_from_env, stream_model_weights

    for value, expected in [(None, "type"), ("layer", "layer"), ("type", "type"), (" Type ", "type"),
                            ("matrix", "layer"), ("", "layer")]:
        if value is None:
            monkeypatch.delenv("SOUP_SOLUTION_WEIGHT_RING", raising=False)
        else:
            monkeypatch.setenv("SOUP_SOLUTION_WEIGHT_RING", value)
        assert ring_from_env() == expected, value
    model = _soup_like(layer_shaped_qwen_dir)
    layers = find_decoder_layers(model)
    x = torch.randn(1, 200, HIDDEN, device="cuda")

    monkeypatch.setenv("SOUP_SOLUTION_WEIGHT_RING", "layer")  # the type ring is the default
    streamer = stream_model_weights(model, slots=2)
    try:
        assert streamer.ring == "layer" and streamer.ring_note is None
        assert len(streamer._slots) == 2 and streamer.slot_bytes == max(streamer.layer_bytes)
        # this model's matrices are not registered (other tests' type-ring streamers may still be)
        assert not streamer._matrices
        assert not any(m in weight_stream._MATRICES for m in model.modules())
        assert fused_proj._mlp_plan(layers[0].mlp, x).ring is None
    finally:
        streamer.restore()

    monkeypatch.setenv("SOUP_SOLUTION_WEIGHT_RING", "type")
    streamer = stream_model_weights(model, slots=2)
    try:
        assert streamer.ring == "type" and len(streamer._slots) == 5
        # counted on this model (other tests' models may still have registered matrices)
        assert sum(m in weight_stream._MATRICES for m in model.modules()) == 3 * len(layers)
        assert fused_proj._mlp_plan(layers[0].mlp, x).ring is streamer
        mlp = layers[1].mlp
        assert weight_stream.mlp_ring(mlp.gate_proj, mlp.up_proj, mlp.down_proj) is streamer
        assert weight_stream.mlp_ring(mlp.gate_proj, layers[2].mlp.up_proj) is False
        assert weight_stream.mlp_ring(layers[1].self_attn.k_proj) is None
    finally:
        streamer.restore()
    assert not any(m in weight_stream._MATRICES for m in model.modules())

    streamer = stream_model_weights(model, slots=2, only={"mlp"}, ring="type")
    try:
        assert streamer.ring == "layer" and "only" in streamer.ring_note
    finally:
        streamer.restore()


@pytest.mark.parametrize("sleep", [0, SLEEP_CYCLES])
def test_type_ring_unmanaged_mlp(sleep, monkeypatch):
    """A layer whose MLP reads its matrices without acquire() (test_weight_stream's Block has no
    fused_proj MLP): the type ring loads the three matrices at the layer's forward and again when
    its backward starts; bit for bit, also with the GPU far behind the CPU."""
    from solution import weight_stream
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.weight_stream import WeightStreamer

    monkeypatch.setitem(weight_stream._RING, "timing", True)  # counts the matrix waits
    patch_bnb_matmul4bit()
    blocks = make_stack(sleep=sleep)
    x, weight = _inputs()
    reference = _step(blocks, x, weight)
    with torch.no_grad():
        alone = [blocks[j](x) for j in (3, 1)]
    streamer = WeightStreamer(blocks, slots=2, ring="type")
    try:
        assert streamer.ring == "type"
        assert _equal(reference, _step(blocks, x, weight))
        assert _equal(reference, _step(blocks, x, weight))
        n = len(blocks)
        stats = streamer.stats()
        assert stats["matrix_waits_forward"] == stats["matrix_waits_backward"] == 2 * 3 * n
        with torch.no_grad():  # blocks out of order: loaded on the spot by their hooks
            assert all(torch.equal(blocks[j](x), want) for j, want in zip((3, 1), alone))
        assert _equal(reference, _step(blocks, x, weight))
    finally:
        streamer.restore()
    assert _equal(reference, _step(blocks, x, weight))


@pytest.mark.parametrize("early", [True, False])
def test_type_ring_int8_backward_early_release(layer_shaped_qwen_dir, night, monkeypatch, early):
    """The int8 MLP backward (int8_bwd.py) with the type ring: with SOUP_SOLUTION_RING_EARLY_RELEASE
    each matrix's slot is released right after its int8 decode, so the next layer's copy into it
    starts while this backward still runs. Every decode is queued behind a GPU sleep: a copy that did
    not wait for its decode changes the result. Two steps equal the resident model's bit for bit."""
    from solution import int8_bwd, weight_stream
    from solution.weight_stream import stream_model_weights

    monkeypatch.setitem(int8_bwd._STATE, "on", True)
    int8_bwd.compile_now()
    monkeypatch.setattr(weight_stream, "EARLY_RELEASE", early)
    model = _soup_like(layer_shaped_qwen_dir, dropout=0.05)
    sleep = [0]
    decode = int8_bwd._decode_t

    def slow_decode(*args, **kwargs):
        if sleep[0]:
            torch.cuda._sleep(sleep[0])
        return decode(*args, **kwargs)

    monkeypatch.setattr(int8_bwd, "_decode_t", slow_decode)
    calls = []
    backward = int8_bwd.mlp_backward
    monkeypatch.setattr(int8_bwd, "mlp_backward",
                        lambda *a, **k: calls.append(a[4] is not None) or backward(*a, **k))
    resident = _ring_step(model)
    exact = _same(resident, _ring_step(model))
    streamer = stream_model_weights(model, slots=2, ring="type")
    assert streamer is not None and streamer.ring == "type"
    calls.clear()
    sleep[0] = SLEEP_CYCLES // 3
    try:
        steps = [_ring_step(model), _ring_step(model)]
        assert streamer.late_loads == 0
    finally:
        sleep[0] = 0
        streamer.restore()
    assert calls and all(c == early for c in calls)  # the int8 path ran, with the callback or not
    compare = _same if exact else _close
    assert all(compare(resident, got) for got in steps)
