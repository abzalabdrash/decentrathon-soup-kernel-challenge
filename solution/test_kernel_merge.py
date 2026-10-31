"""SOUP_SOLUTION_KERNEL_MERGE: head_chunk_kernel's first chunk through the FIRST=False variant with the
runtime flag `first` (MERGE=True) gives the FIRST=True variant's results bit for bit."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def _run(merge, logits, targets, weight, n_lab, chunk, acc_fill):
    from solution.head_stream import BLOCK_D, IGNORE_INDEX, _num_warps, head_chunk_kernel

    rows, vocab = logits.shape
    d = weight.shape[1]
    state = torch.full((5, rows), 7.0, device="cuda")  # garbage the first chunk must not read
    idx = torch.full((rows,), 3, dtype=torch.int32, device="cuda")
    acc = torch.full((n_lab, d), acc_fill, device="cuda")
    wt = torch.zeros(n_lab, d, device="cuda", dtype=weight.dtype)
    parts = []
    block = 1 << (chunk - 1).bit_length()
    for c, c0 in enumerate(range(0, vocab, chunk)):
        vc = min(chunk, vocab - c0)
        part = logits[:, c0:c0 + vc].contiguous()
        w = weight[c0:c0 + vc]
        head_chunk_kernel[(rows,)](part, part.stride(0), targets, *state, idx, acc, acc.stride(0), w,
                                   w.stride(0), wt, wt.stride(0), c0, vc, n_lab, d,
                                   FIRST=c == 0 and not merge, IGNORE=IGNORE_INDEX, BLOCK=block,
                                   BLOCK_D=BLOCK_D, first=int(c == 0), MERGE=merge,
                                   num_warps=_num_warps(block))
        parts.append(part)
        if c == 0:
            acc_after_first = acc.clone()
        acc.mul_(1.0).add_(torch.ones_like(acc))  # what the head's addmm would change between chunks
    return state, idx, acc, acc_after_first, wt, torch.cat(parts, 1)


@pytest.mark.parametrize("acc_fill", [0.0, float("nan")])
def test_merged_first_chunk_is_bitwise(acc_fill):
    torch.manual_seed(0)
    rows, vocab, d, chunk, n_lab = 37, 3000, 96, 1024, 29
    logits = (torch.randn(rows, vocab, device="cuda") * 4).half()
    logits[5, 17] = logits[5, 2000] = 30.0  # a tie across chunks
    targets = torch.randint(0, vocab, (rows,), device="cuda")
    targets[3] = targets[30] = -100
    weight = torch.randn(vocab, d, device="cuda").half()
    ref = _run(False, logits, targets, weight, n_lab, chunk, acc_fill)
    got = _run(True, logits, targets, weight, n_lab, chunk, acc_fill)
    for a, b in zip(ref, got):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))  # NaN patterns too


def test_warm_compiles_one_variant_with_merge(monkeypatch):
    from solution import head_stream

    seen = []
    monkeypatch.setattr(head_stream, "MERGE", True)
    import solution.warmup as warmup

    monkeypatch.setattr(warmup, "compile_kernel", lambda kernel, *a, **k: seen.append((kernel, k)))
    head_stream._warm_kernels(head_stream._config()[0])
    chunk = [k for kernel, k in seen if kernel is head_stream.head_chunk_kernel]
    assert len(chunk) == 1 and chunk[0]["FIRST"] is False and chunk[0]["MERGE"] is True
