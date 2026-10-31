"""Pre-test of "one autograd.Function per decoder block" (external review 1, item 2; task C1): what an
autograd node costs the host, against what a layer's host work costs.

    python -m solution.bench_nodes                  # 300 and 830 tokens, 60 iterations
    python -m solution.bench_nodes --iters 100 --tokens 300/830/1440

1. A chain of 6 custom autograd Functions (one small op each, two saved tensors, ctx attributes:
   a decoder layer has 6 nodes, two norms, q/k/v, attention, o_proj, MLP) against one Function that
   runs the same 6 ops each way: host us per call, forward and backward apart; the difference over 5
   is the host cost of one node.
2. One Qwen2.5-1.5B attention block with the recompute (the real fused code, 4 rows) as three nodes,
   as two (SOUP_SOLUTION_ATTN_TWO_NODES) and as one (_AttnBlock), and the fused MLP: host ms to
   issue the forward and the backward from an idle GPU (median), kernel launches.

The modes alternate call by call, so clock changes hit all alike.
"""

import argparse
import statistics
import time

import torch

K = 6


class _Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        out = x.add(w)
        ctx.save_for_backward(x, out)
        ctx.shape, ctx.flag = x.shape, True
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x, out = ctx.saved_tensors
        return grad.mul(1.0), None


class _Ops(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        saved = []
        for _ in range(K):
            out = x.add(w)
            saved += [x, out]
            x = out
        ctx.save_for_backward(*saved)
        ctx.shape, ctx.flag = x.shape, True
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        ctx.saved_tensors  # noqa: B018 - unpacked as the chain's nodes unpack theirs
        for _ in range(K):
            grad = grad.mul(1.0)
        return grad, None


def _issue(fn):
    """Host seconds to issue `fn` from an idle GPU."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed, result


def _chain(n, iters):
    x0 = torch.randn(n, 1536, device="cuda", dtype=torch.float16)
    w = torch.zeros((), device="cuda", dtype=torch.float16)
    grad = torch.randn_like(x0)
    times = {"nodes": ([], []), "one": ([], [])}

    def nodes(x):
        for _ in range(K):
            x = _Op.apply(x, w)
        return x

    for i in range(iters + 5):
        for name, fn in (("nodes", nodes), ("one", lambda x: _Ops.apply(x, w))):
            x = x0.detach().requires_grad_(True)
            fwd, out = _issue(lambda: fn(x))
            bwd, _ = _issue(lambda: out.backward(grad))
            if i >= 5:
                times[name][0].append(fwd)
                times[name][1].append(bwd)
    med = {k: (statistics.median(f) * 1e6, statistics.median(b) * 1e6) for k, (f, b) in times.items()}
    per_node = ((med["nodes"][0] - med["one"][0]) / (K - 1), (med["nodes"][1] - med["one"][1]) / (K - 1))
    print(f"chain of {K} nodes vs one node, {n} x 1536 half: forward {med['nodes'][0]:.0f} vs "
          f"{med['one'][0]:.0f} us, backward {med['nodes'][1]:.0f} vs {med['one'][1]:.0f} us; "
          f"one node costs the host {per_node[0]:.1f} us forward + {per_node[1]:.1f} us backward",
          flush=True)
    return per_node


def _launches(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return len([e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA])


def _blocks(tokens, iters):
    from transformers.models.qwen2 import modeling_qwen2 as mq

    from solution import fused_proj, unpad
    from solution import test_fused_proj as t
    from solution.mlp16 import patch_qwen2_mlp_half
    from solution.nf4 import patch_bnb_matmul4bit
    from solution.rope import patch_qwen2_rope

    unpad.patch_qwen2_attention()
    patch_qwen2_rope()
    patch_bnb_matmul4bit()
    patch_qwen2_mlp_half()
    fused_proj.patch_qwen2_attention_fused()
    fused_proj.patch_qwen2_mlp_fused()
    fused_proj._RECOMPUTE["mode"] = "all"
    config = t._config()
    attn = t._quantize_attention(mq.Qwen2Attention(config, 0), seed=3, dropout=0.05).cuda().train()
    mlp = t._quantize_mlp(mq.Qwen2MLP(config), seed=20).cuda().train()
    rotary = mq.Qwen2RotaryEmbedding(config=config).cuda()
    modes = {"three": (False, False), "two": (False, True), "one": (True, False)}
    for n in tokens:
        lengths = [n // 4] * 3 + [n - 3 * (n // 4)]
        x, pe, grad = t._attention_inputs(rotary, seed=5, lengths=lengths)
        spans = t._spans(lengths)

        def attention_fwd():
            with torch.autocast("cuda", dtype=torch.float16), unpad.varlen(spans):
                return attn(xr, pe, None)[0]

        def mlp_fwd():
            with torch.autocast("cuda", dtype=torch.float16):
                return mlp(xr)

        times = {name: ([], []) for name in (*modes, "mlp")}
        counts = {}
        for i in range(iters + 5):
            for name in (*modes, "mlp"):
                if name in modes:
                    fused_proj._ATTN["one_function"], fused_proj._ATTN["two_nodes"] = modes[name]
                xr = x.detach().requires_grad_(True)
                for p in (*attn.parameters(), *mlp.parameters()):
                    p.grad = None
                fwd, out = _issue(attention_fwd if name in modes else mlp_fwd)
                bwd, _ = _issue(lambda: out.backward(grad))
                if i >= 5:
                    times[name][0].append(fwd)
                    times[name][1].append(bwd)
                if i == 0:
                    xr = x.detach().requires_grad_(True)
                    out = None
                    holder = {}

                    def forward():
                        holder["out"] = (attention_fwd if name in modes else mlp_fwd)()

                    counts[name] = (_launches(forward), _launches(lambda: holder["out"].backward(grad)))
        print(f"{n} tokens in 4 rows (host ms to issue from an idle GPU, median of {iters}; launches):",
              flush=True)
        for name, (f, b) in times.items():
            print(f"  {name:6s} forward {statistics.median(f) * 1e3:.3f} ms backward "
                  f"{statistics.median(b) * 1e3:.3f} ms; launches {counts[name][0]} / {counts[name][1]}",
                  flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iters", type=int, default=60)
    parser.add_argument("--tokens", default="300/830")
    args = parser.parse_args()
    print(torch.cuda.get_device_name(), torch.__version__, flush=True)
    tokens = [int(t) for t in args.tokens.split("/")]
    for n in tokens:
        _chain(n, args.iters)
    _blocks(tokens, args.iters)


if __name__ == "__main__":
    main()
