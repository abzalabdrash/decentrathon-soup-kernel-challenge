"""CPU coverage for child methods which fused projections would otherwise bypass."""

import torch


def test_fused_child_guard_rejects_instance_forward(monkeypatch):
    from solution.fused_proj import _no_hooks

    child = torch.nn.Linear(4, 4)
    assert _no_hooks(child)
    with monkeypatch.context() as patch:
        patch.setattr(child, "forward", lambda value: value + 1)
        assert not _no_hooks(child)
    assert _no_hooks(child)


def test_fused_child_guard_rejects_identity_hooks():
    from solution.fused_proj import _no_hooks

    child = torch.nn.Identity()
    assert _no_hooks(child)
    handle = child.register_forward_hook(lambda module, inputs, output: output + 1)
    try:
        assert not _no_hooks(child)
    finally:
        handle.remove()
    assert _no_hooks(child)
