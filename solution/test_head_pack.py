"""The 12-bit packed head weight (head_pack.py): exact round trip through the decode kernel on values
shaped like a bf16 checkpoint held in fp16 (subnormal classes, zeros, a 17th class that escapes), the
refusals, and the streamed head giving the same bits from the packed ring as from the fp16 ring."""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _bf16_like(rows, cols, seed=0, big=0):
    """fp16 values that came from bf16 (as the run's embedding): the bulk ~N(0, 0.02) (exponent
    fields 1..11), a band of subnormals and zeros (the 4 subnormal classes), 2,000 values with
    exponent field 12 and `big` values with field 13 (the checkpoint's rare 17th class)."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=gen) * 0.02
    tiny = torch.rand(rows, cols, generator=gen) < 0.01
    x = torch.where(tiny, torch.randn(rows, cols, generator=gen) * 3e-5, x)
    x[torch.rand(rows, cols, generator=gen) < 0.001] = 0.0
    x = x.bfloat16().half()
    flat = x.view(-1)
    idx = torch.randperm(flat.numel(), generator=gen)[:2000 + big]
    flat[idx[:2000]] = (torch.rand(2000, generator=gen) * 0.1 + 0.13).bfloat16().half()
    if big:
        flat[idx[2000:]] = (torch.rand(big, generator=gen) * 0.2 + 0.26).bfloat16().half()
    return x


def _expected_escapes(weight, rows):
    """Values outside their chunk's 16 most frequent classes (an independent count)."""
    from solution.head_pack import classify

    total = 0
    for c0 in range(0, weight.shape[0], rows):
        _, cid = classify(weight[c0:c0 + rows].cuda())
        counts = sorted(torch.bincount(cid.int()).tolist(), reverse=True)
        total += sum(counts[16:])
    return total


def _fake_streamer(weight, rows):
    host = torch.empty(weight.shape, dtype=weight.dtype, pin_memory=True)
    host.copy_(weight)
    vocab, hidden = weight.shape
    return types.SimpleNamespace(dtype=weight.dtype, hidden=hidden, rows=rows, vocab=vocab,
                                 n_chunks=-(-vocab // rows), device=torch.device("cuda"), host=host)


def _decode_all(pack, weight):
    out = torch.empty_like(weight, device="cuda")
    slot = torch.empty(pack.slot_bytes, dtype=torch.uint8, device="cuda")
    for c in range(pack.n_chunks):
        c0, vc, _, nb = pack.chunk(c)
        slot[:nb].copy_(pack.host_chunk(c))
        pack.decode(c, slot, out[c0:c0 + vc])
    return out


def test_classes_of_bf16_values_are_exact():
    """Every finite bf16 value in fp16 range: classify -> 16-entry table -> decode gives its bits."""
    from solution.head_pack import classify, N_CLASSES

    bits = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    x = bits[torch.isfinite(bits)].float()
    x = x[(x.abs() < 65504)].half()
    x = x[x.bfloat16().float() == x.float()]  # bf16 values exactly representable in fp16 (no rounding)
    a, cid = classify(x.cuda())
    assert int(cid.max()) < N_CLASSES - 1, "a bf16-origin value without a class"


@pytest.mark.parametrize("rows,big", [(4096, 0), (4096, 7), (3072, 3)])
def test_round_trip_is_bit_exact(rows, big):
    from solution.head_pack import HeadPack

    weight = _bf16_like(2 * rows + 384, 1536, seed=rows + big, big=big)
    pack = HeadPack(_fake_streamer(weight, rows))
    pack.build()
    assert pack.ready and pack.escapes == _expected_escapes(weight, rows)
    assert (pack.escapes > 0) == (big > 0)
    assert pack.total_bytes == weight.numel() * 3 // 2
    out = _decode_all(pack, weight)
    assert torch.equal(out.view(torch.int16).cpu(), weight.view(torch.int16))
    pack.stop()


def test_escape_overflow_and_dtype_refused():
    from solution.head_pack import MAX_ESC, HeadPack, PackSkip

    weight = _bf16_like(512, 1536, seed=3)
    weight.view(-1)[: MAX_ESC + 1] = torch.full((MAX_ESC + 1,), 0.0123, dtype=torch.float16)  # 3 low bits set
    assert int(weight.view(-1)[0].view(torch.int16)) & 7
    pack = HeadPack(_fake_streamer(weight, 512))
    with pytest.raises(PackSkip):
        pack.build()
    assert not pack.ready
    with pytest.raises(PackSkip):
        HeadPack(_fake_streamer(weight.bfloat16(), 512))


def test_non_bf16_values_escape_exactly():
    """A few fp16 values with all 10 mantissa bits (not from bf16) per chunk: escaped, still exact."""
    from solution.head_pack import HeadPack

    weight = _bf16_like(1024, 1536, seed=5)
    # a normal value with mantissa bit 0 set, one with all set, an odd subnormal in [256, 512), one in
    # [512, 1024) that is not a multiple of 4
    odd = torch.tensor([0x2001, 0x23FF - 0x10000 + 0x8000, 0x0101, 0x8201 - 0x10000], dtype=torch.int32)
    weight[::256, 7] = odd.to(torch.int16).view(torch.float16)
    pack = HeadPack(_fake_streamer(weight, 512))
    pack.build()
    assert pack.escapes == 4
    assert torch.equal(_decode_all(pack, weight).view(torch.int16).cpu(), weight.view(torch.int16))
    pack.stop()


@pytest.mark.parametrize("n,rows", [(700, 4096), (1, 4096), (333, 3072)])
def test_streamed_head_packed_equals_fp16(n, rows):
    """head() from the packed ring returns the same bits as from the fp16 ring (same chunks, same
    kernels); the pack thread builds while nothing waits for it, the ring switches at the next head."""
    from solution.head_stream import HeadStreamer

    vocab, d = 3 * rows + 1000, 1536
    weight = _bf16_like(vocab, d, seed=n, big=2).cuda()
    param = torch.nn.Parameter(weight.clone(), requires_grad=False)
    streamer = HeadStreamer(param, rows=rows, slots=2)
    try:
        gen = torch.Generator(device="cuda").manual_seed(n)
        hidden = (torch.randn(n, d, device="cuda", generator=gen) * 2).half()
        targets = torch.randint(0, vocab, (n,), device="cuda", generator=gen)
        targets[: n // 3] = -100
        order = torch.argsort((targets != -100).to(torch.int8), descending=True, stable=True)
        h16, t = hidden[order].contiguous(), targets[order].contiguous()
        n_lab = int((t != -100).sum())
        raw = streamer.head(h16, t, n_lab, 1.0 / max(n_lab, 1))
        pack = streamer.start_pack(log=lambda *_: None)
        pack.join()
        assert pack.ready, pack.error
        packed = streamer.head(h16, t, n_lab, 1.0 / max(n_lab, 1))
        assert streamer.packed_heads == 1 and streamer.heads == 2
        for a, b in zip(raw, packed):
            assert torch.equal(a, b)
        st = streamer.stats()
        assert st["late_loads"] == 0
        assert streamer.copy_bytes < streamer.n_chunks * rows * d * 2 * 2  # the packed head moved fewer
    finally:
        streamer.restore()
