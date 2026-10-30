"""The half-storage MLP against the original Qwen2MLP under autocast."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_mlp_half_matches_original():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

    from solution.mlp16 import patch_qwen2_mlp_half

    patch_qwen2_mlp_half()
    original = Qwen2MLP._soup_original_forward
    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=96, intermediate_size=224)
    mlp = Qwen2MLP(config).cuda().half()
    x = torch.randn(2, 37, 96, device="cuda", dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    with torch.autocast("cuda", dtype=torch.float16):
        ref = original(mlp, x_ref)
        ref.float().sum().backward()
        out = mlp(x)
        out.float().sum().backward()
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=5e-2, atol=5e-2)


def test_shrink_embeddings_is_exact_only():
    from solution.embed16 import shrink_embeddings

    emb = torch.nn.Embedding(50, 8).cuda()
    emb.weight.requires_grad_(False)
    emb.weight.data = emb.weight.data.half().float()  # exact in fp16

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = emb

        def get_input_embeddings(self):
            return self.e

    m = M()
    assert shrink_embeddings(m) == 1
    assert emb.weight.dtype == torch.float16
    out = emb(torch.tensor([1, 2], device="cuda"))
    assert out.dtype == torch.float32
    emb2 = torch.nn.Embedding(50, 8).cuda()
    emb2.weight.requires_grad_(False)
    emb2.weight.data += 1e-9  # not representable in fp16
    m2 = M()
    m2.e = emb2
    assert shrink_embeddings(m2) == 0 and emb2.weight.dtype == torch.float32
