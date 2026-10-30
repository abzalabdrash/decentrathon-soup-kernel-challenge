"""offload._gc_paused (SOUP_SOLUTION_GC_PAUSE): the collector's state inside Soup's training and
after it (startup.py may have frozen objects before: the counts only grow)."""

import gc

import pytest


@pytest.mark.parametrize("mode", ["0", "freeze", "off"])
def test_gc_state_inside_and_after(monkeypatch, mode):
    from solution import offload

    monkeypatch.setattr(offload, "GC_PAUSE", mode)
    assert gc.isenabled()
    before = gc.get_freeze_count()
    keep = [[] for _ in range(100)]  # objects made after the last freeze
    with offload._gc_paused():
        assert gc.isenabled() == (mode != "off")
        inside = gc.get_freeze_count()
        assert inside >= before + (100 if mode != "0" else 0)
    # nothing is unfrozen (frozen objects can still die by reference counting: the context's own)
    assert gc.isenabled() and before <= gc.get_freeze_count() <= inside
    del keep


def test_gc_enabled_again_after_an_error(monkeypatch):
    from solution import offload

    monkeypatch.setattr(offload, "GC_PAUSE", "off")
    with pytest.raises(RuntimeError), offload._gc_paused():
        raise RuntimeError("training failed")
    assert gc.isenabled()
