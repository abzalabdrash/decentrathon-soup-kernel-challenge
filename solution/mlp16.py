"""The MLP's big intermediates in fp16 instead of fp32 (lossless) and one cast instead of two.

In the base the residual stream is fp32 (prepare_model_for_kbit_training casts the norm weights to
fp32, and `fp32 weight * fp16 tensor` promotes), so the normed input of the MLP is fp32. bitsandbytes'
Linear4bit casts it to fp16 for the GEMM and casts the fp16 result back to fp32: `gate` and `up` are
fp32 tensors (2048 x 8960 x 4 bytes = 73 MB each on this task) that hold fp16 values, and SiLU and
the product run and save in fp32: three 73 MB tensors per layer stay alive for the backward, about
6 GB over 28 layers.

Here the input is cast to half once, the three projections run on half in and half out (no casts
inside bitsandbytes), the SwiGLU is the kit's example kernel (fp32 math, half storage), and only the
down projection's output goes back to the residual dtype. What is saved per layer: `gate` and `up`
in half, 73 MB instead of 219 MB. The values are the same as the base's: the GEMM outputs were
half before the base upcast them, and the base casts the fp32 product to half for the down GEMM.
"""

import torch

from .swiglu import swiglu


def half_dtype():
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return torch.float16


def patch_qwen2_mlp_half():
    from transformers.models.qwen2 import modeling_qwen2

    cls = modeling_qwen2.Qwen2MLP
    if getattr(cls, "_soup_mlp16", False):
        return
    original = getattr(cls, "_soup_original_forward", None) or cls.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if (act != "silu" or not torch.is_tensor(x) or not x.is_cuda
                or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
            return original(self, x)
        half = x.dtype if x.dtype != torch.float32 else half_dtype()
        x_half = x if x.dtype == half else x.to(half)
        gate = self.gate_proj(x_half)
        up = self.up_proj(x_half)
        hidden = swiglu(gate.to(half), up.to(half))
        out = self.down_proj(hidden)
        return out if out.dtype == x.dtype else out.to(x.dtype)

    cls.forward = forward
    cls._soup_original_forward = original
    cls._soup_mlp16 = True
