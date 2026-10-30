"""The tied embedding / lm_head weight stored in fp16 instead of fp32 (lossless on this run).

The checkpoint is fp16 and Soup loads it in fp16; peft's prepare_model_for_kbit_training then casts
every non-4-bit parameter to fp32, the 151,936 x 1536 tied embedding included: 933 MB that hold fp16
values. The base pays for it twice more: autocast makes an fp16 copy of the lm_head weight every
step (467 MB), and the embedding lookup is an fp32 gather.

After Soup has built the model (SFTTrainerWrapper._setup_transformers), the parameter goes back to
fp16 when the round trip is exact element for element (it is for an fp16 checkpoint; a bf16-loaded
model on an Ampere card is left alone). A forward hook casts the embedding output to fp32, so the
residual stream and the rotary tables keep the base's dtype; the fused head (fused_ce.py) uses the
fp16 weight directly. The parameter is frozen, so nothing else sees the change.

With head_stream.py on (the default), the weight is in pinned host memory before peft's cast and
never comes back to the GPU: this module then leaves it alone and only serves as the fallback.
"""

import torch


ROWS_PER_CHUNK = 8192  # 8192 x 1536 fp32 = 48 MiB per comparison chunk


def _round_trip_exact(half, full):
    """True when `half` cast back to fp32 equals `full` element for element. Compared in row chunks:
    one `half.to(fp32)` of the whole embedding would add a third 933 MiB tensor next to the fp32
    weight and its fp16 copy, a setup spike that nvidia-smi (100 ms sampling) misses while torch's
    reserved peak records it, which fails check 9 (peaks must agree within 3%)."""
    rows = full.shape[0] if full.dim() > 1 else 1
    half2, full2 = half.reshape(rows, -1), full.reshape(rows, -1)
    for start in range(0, rows, ROWS_PER_CHUNK):
        end = min(start + ROWS_PER_CHUNK, rows)
        if not torch.equal(half2[start:end].to(torch.float32), full2[start:end]):
            return False
    return True


def _streamed(weight):
    from .head_stream import streamer_for

    return streamer_for(weight) is not None


def shrink_embeddings(model):
    """Returns the number of embedding parameters moved to fp16 (0, 1 or 2)."""
    if not isinstance(model, torch.nn.Module):
        return 0
    done = 0
    seen = set()
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        get = getattr(model, getter, None)
        try:
            module = get() if callable(get) else None
        except Exception:  # noqa: BLE001 - a model without that accessor
            module = None
        if not isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
            continue
        weight = module.weight
        if id(weight) in seen:
            continue
        seen.add(id(weight))
        if weight.dtype != torch.float32 or not weight.is_cuda or weight.requires_grad:
            continue
        if _streamed(weight):
            continue  # its values live in host memory (head_stream.py); the GPU holds 4 bytes
        half = weight.data.to(torch.float16)
        if not _round_trip_exact(half, weight.data):
            del half
            continue
        weight.data = half
        done += 1
        if isinstance(module, torch.nn.Embedding) and not getattr(module, "_soup_fp32_out", False):
            module.register_forward_hook(lambda mod, inputs, output: output.to(torch.float32))
            module._soup_fp32_out = True
    if done:
        torch.cuda.empty_cache()
    return done


def patch_soup_setup():
    """After SFTTrainerWrapper._setup_transformers returns, shrink the embeddings of the real model.
    Soup's regression tests call this with fake modules: they are left untouched."""
    from soup_cli.trainer import sft as soup_sft

    cls = soup_sft.SFTTrainerWrapper
    if getattr(cls, "_soup_embed16", False):
        return
    original = cls._setup_transformers

    def _setup_transformers(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        model = getattr(self, "model", None)
        if isinstance(model, torch.nn.Module):
            try:
                moved = shrink_embeddings(model)
                if moved:
                    print(f"[solution] embedding weights stored in fp16 ({moved} tensor(s))")
            except Exception as exc:  # noqa: BLE001 - an optimisation must not break setup
                print(f"[solution] embedding fp16 skipped: {exc!r}")
        return result

    cls._setup_transformers = _setup_transformers
    cls._soup_embed16 = True
