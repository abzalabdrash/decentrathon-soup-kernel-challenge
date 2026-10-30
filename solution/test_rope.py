"""The Triton RoPE against transformers' apply_rotary_pos_emb, forward and backward, and the patch.

Microbenchmark against the original on the task's shapes: `python -m solution.test_rope`.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

D = 128
TOLERANCE = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=1e-2, atol=1e-2),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}
# (batch, q heads, kv heads, tokens, lengths of the packed rows or None for padded rows)
CASES = [(1, 12, 2, 826, (300, 211, 190, 125)), (4, 12, 2, 512, None)]


def _reference():
    from transformers.models.qwen2 import modeling_qwen2 as mq

    return getattr(mq, "_soup_original_apply_rotary_pos_emb", mq.apply_rotary_pos_emb)


def _positions(batch, n, lengths):
    """Per-row positions 0..len-1, as unpad.pack_batch makes them (packed) or as the padded base."""
    if lengths is None:
        return torch.arange(n, device="cuda").expand(batch, n)
    return torch.cat([torch.arange(length, device="cuda") for length in lengths]).unsqueeze(0)


def _tables(position_ids, dtype):
    """cos, sin from the real Qwen2RotaryEmbedding with Qwen2.5-1.5B's rotary settings."""
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

    config = Qwen2Config(hidden_size=1536, num_attention_heads=12, num_key_value_heads=2,
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    rotary = Qwen2RotaryEmbedding(config).cuda()
    return rotary(torch.empty(1, device="cuda", dtype=dtype), position_ids)


def _projection(batch, heads, n, dtype):
    """A (B, N, H*D) leaf as q_proj makes it; the attention sees it as a transposed (B, H, N, D)
    view."""
    return torch.randn(batch, n, heads * D, device="cuda", dtype=dtype, requires_grad=True)


def _heads(flat):
    batch, n, _ = flat.shape
    return flat.view(batch, n, -1, D).transpose(1, 2)


def _run(fn, flat_q, flat_k, cos, sin, grad_q, grad_k):
    q_embed, k_embed = fn(_heads(flat_q), _heads(flat_k), cos, sin)
    torch.autograd.backward((q_embed, k_embed), (grad_q, grad_k))
    out = (q_embed.detach(), k_embed.detach(), flat_q.grad, flat_k.grad)
    flat_q.grad = flat_k.grad = None
    return out


def _compare(cos, sin, batch, q_heads, kv_heads, n, dtype, seed):
    from solution.rope import apply_rotary_pos_emb_triton

    torch.manual_seed(seed)
    flat_q = _projection(batch, q_heads, n, dtype)
    flat_k = _projection(batch, kv_heads, n, dtype)
    out_dtype = torch.promote_types(dtype, cos.dtype)
    grad_q = torch.randn(batch, q_heads, n, D, device="cuda", dtype=out_dtype)
    grad_k = torch.randn(batch, kv_heads, n, D, device="cuda", dtype=out_dtype)

    ref = _run(_reference(), flat_q, flat_k, cos, sin, grad_q, grad_k)
    out = _run(apply_rotary_pos_emb_triton, flat_q, flat_k, cos, sin, grad_q, grad_k)
    for got, want, name in zip(out, ref, ("q_embed", "k_embed", "dq", "dk")):
        assert got.dtype == want.dtype and got.shape == want.shape, name
        torch.testing.assert_close(got, want, **TOLERANCE[dtype], msg=name)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch,q_heads,kv_heads,n,lengths", CASES)
def test_rope_matches_original(batch, q_heads, kv_heads, n, lengths, dtype):
    cos, sin = _tables(_positions(batch, n, lengths), dtype)
    assert cos.shape == (1 if lengths else batch, n, D)
    _compare(cos, sin, batch, q_heads, kv_heads, n, dtype, seed=n)


def test_rope_keeps_the_transposed_layout():
    """Outputs and gradients come in the layout of the (B, N, H, D) projection, so the backward of
    transpose + view needs no copy."""
    from solution.rope import apply_rotary_pos_emb_triton

    cos, sin = _tables(_positions(1, 826, (826,)), torch.float32)
    q = _heads(_projection(1, 12, 826, torch.float32))
    k = _heads(_projection(1, 2, 826, torch.float32))
    q_embed, k_embed = apply_rotary_pos_emb_triton(q, k, cos, sin)
    dq, dk = torch.autograd.grad((q_embed, k_embed), (q, k),
                                 (torch.randn(q.shape, device="cuda"),
                                  torch.randn(k.shape, device="cuda")))
    assert q_embed.stride() == dq.stride() == q.stride() == (826 * 12 * D, D, 12 * D, 1)
    assert k_embed.stride() == dk.stride() == k.stride()


def test_rope_half_inputs_with_fp32_tables():
    """q, k in fp16 with fp32 cos, sin: the original promotes the result to fp32."""
    cos, sin = _tables(_positions(1, 826, (826,)), torch.float32)
    q_embed, *_ = _compare(cos, sin, 1, 12, 2, 826, torch.float16, seed=1)
    assert q_embed.dtype == torch.float32


def test_rope_tables_indexed_per_batch_row():
    """Random cos, sin that differ per batch row and per half, as a non-contiguous view."""
    torch.manual_seed(2)
    cos = torch.randn(5, 3, 77, D, device="cuda")[:, 1]
    sin = torch.randn(5, 3, 77, D, device="cuda")[:, 2]
    _compare(cos, sin, 5, 6, 3, 77, torch.float32, seed=3)


def test_rope_falls_back_where_unsupported():
    from solution import rope

    calls = []
    launch = rope._launch
    rope._launch = lambda *args, **kwargs: (calls.append(1), launch(*args, **kwargs))[1]
    try:
        q, k = torch.randn(2, 4, 9, 64), torch.randn(2, 2, 9, 64)
        cos, sin = torch.randn(2, 9, 64), torch.randn(2, 9, 64)
        cases = [
            (q, k, cos, sin, 1),  # CPU
            (q.cuda().transpose(1, 2), k.cuda().transpose(1, 2), cos.cuda(), sin.cuda(), 2),
            (q.cuda()[..., :48], k.cuda()[..., :48], cos.cuda()[..., :48], sin.cuda()[..., :48], 1),
            (q.cuda(), k.cuda(), cos.cuda().requires_grad_(True), sin.cuda(), 1),
            (q.cuda(), k.cuda(), cos.cuda(), sin.cuda()[:1], 1),  # cos and sin of different shapes
        ]
        for q_, k_, cos_, sin_, dim in cases:
            got = rope.apply_rotary_pos_emb_triton(q_, k_, cos_, sin_, dim)
            want = _reference()(q_, k_, cos_, sin_, dim)
            for a, b in zip(got, want):
                torch.testing.assert_close(a, b)
        assert not calls
    finally:
        rope._launch = launch


def test_patch_routes_qwen2_attention_through_kernel(monkeypatch):
    from transformers import Qwen2Config
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import rope
    from solution.unpad import patch_qwen2_attention, varlen

    # both patches are undone when the test ends, so other tests in the process see the module as
    # they found it
    for owner, names in ((mq, ("apply_rotary_pos_emb", "_soup_rope",
                               "_soup_original_apply_rotary_pos_emb")),
                         (mq.Qwen2Attention, ("forward", "_soup_varlen", "_soup_original_forward"))):
        for name in names:
            if name in vars(owner):
                monkeypatch.setattr(owner, name, vars(owner)[name])
            else:
                monkeypatch.setattr(owner, name, False, raising=False)
                monkeypatch.delattr(owner, name)

    rope.patch_qwen2_rope()
    original = mq._soup_original_apply_rotary_pos_emb
    rope.patch_qwen2_rope()  # a second call changes nothing
    assert mq.apply_rotary_pos_emb is rope.apply_rotary_pos_emb_triton
    assert mq._soup_original_apply_rotary_pos_emb is original is not rope.apply_rotary_pos_emb_triton
    patch_qwen2_attention()

    calls = []
    launch = rope._launch
    monkeypatch.setattr(rope, "_launch", lambda *args, backward: (
        calls.append(backward), launch(*args, backward=backward))[1])

    torch.manual_seed(4)
    config = Qwen2Config(hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
                         intermediate_size=512, num_hidden_layers=1,
                         rope_parameters={"rope_type": "default", "rope_theta": 1000000.0})
    config._attn_implementation = "sdpa"
    attention = mq.Qwen2Attention(config, layer_idx=0).cuda()
    rotary = mq.Qwen2RotaryEmbedding(config).cuda()

    def step(x, position_ids, spans=None):
        x = x.detach().clone().requires_grad_(True)
        cos, sin = rotary(x, position_ids)
        if spans is None:
            out, _ = attention(x, (cos, sin), attention_mask=None)
        else:
            with varlen(spans):
                out, _ = attention(x, (cos, sin), attention_mask=None)
        out.square().sum().backward()
        return out.detach(), x.grad

    padded = torch.randn(2, 37, 256, device="cuda")
    packed = torch.randn(1, 37, 256, device="cuda")
    packed_positions = torch.cat([torch.arange(20), torch.arange(17)]).unsqueeze(0).cuda()
    runs = [(padded, torch.arange(37, device="cuda").expand(2, -1), None),
            (packed, packed_positions, [(0, 20), (20, 37)])]
    for x, position_ids, spans in runs:
        calls.clear()
        out, grad = step(x, position_ids, spans)
        assert calls == [False, True]  # the kernel ran forward and backward
        monkeypatch.setattr(mq, "apply_rotary_pos_emb", original)
        ref_out, ref_grad = step(x, position_ids, spans)
        monkeypatch.setattr(mq, "apply_rotary_pos_emb", rope.apply_rotary_pos_emb_triton)
        torch.testing.assert_close(out, ref_out, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(grad, ref_grad, rtol=1e-4, atol=1e-5)


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _kernel_ms(fn, iters=50):
    """GPU time of all kernels one call launches (torch.profiler), without the CPU-side cost."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    return sum(event.self_device_time_total for event in prof.key_averages()) / 1000 / iters


def bench():
    """Original vs Triton, forward and backward (the gradient down to the (B, N, H*D) projection),
    ms per call: wall time by CUDA events over 50 calls after warm-up, and the GPU time of the
    kernels alone by torch.profiler (the wall time also holds autograd's and the launch's CPU work
    when that is longer than the kernels)."""
    from solution.rope import apply_rotary_pos_emb_triton

    print(f"{torch.cuda.get_device_name()}; fp32 q, k, cos, sin; ms per call, mean of 50")
    print("q, k                          | timer   | fwd eager | fwd triton | bwd eager | bwd triton")
    for batch, q_heads, kv_heads, n, lengths in CASES:
        cos, sin = _tables(_positions(batch, n, lengths), torch.float32)
        flat_q = _projection(batch, q_heads, n, torch.float32)
        flat_k = _projection(batch, kv_heads, n, torch.float32)
        grad_q = torch.randn(batch, q_heads, n, D, device="cuda")
        grad_k = torch.randn(batch, kv_heads, n, D, device="cuda")
        times = {"events": [], "kernels": []}
        for fn in (_reference(), apply_rotary_pos_emb_triton):
            q, k = _heads(flat_q), _heads(flat_k)
            outs = fn(q, k, cos, sin)

            def forward():
                with torch.no_grad():
                    fn(q, k, cos, sin)

            def backward():
                torch.autograd.grad(outs, (flat_q, flat_k), (grad_q, grad_k), retain_graph=True)

            times["events"].append((_time_ms(forward), _time_ms(backward)))
            times["kernels"].append((_kernel_ms(forward), _kernel_ms(backward)))
        shapes = f"({batch},{q_heads},{n},{D})+({batch},{kv_heads},{n},{D})"
        for timer, ((fe, be), (ft, bt)) in times.items():
            print(f"{shapes:30s}| {timer:7s} | {fe:9.3f} | {ft:10.3f} | {be:9.3f} | {bt:10.3f}")


if __name__ == "__main__":
    bench()
