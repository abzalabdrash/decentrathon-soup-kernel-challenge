"""offload._gc_paused (SOUP_SOLUTION_GC_PAUSE): the collector's state inside Soup's training and
after it."""

import gc

import pytest


@pytest.mark.parametrize("mode", ["0", "freeze", "off"])
def test_gc_state_inside_and_after(monkeypatch, mode):
    from solution import offload

    monkeypatch.setattr(offload, "GC_PAUSE", mode)
    assert gc.isenabled() and gc.get_freeze_count() == 0
    with offload._gc_paused():
        assert gc.isenabled() == (mode != "off")
        assert (gc.get_freeze_count() > 0) == (mode != "0")
    assert gc.isenabled() and gc.get_freeze_count() == 0


def test_gc_enabled_again_after_an_error(monkeypatch):
    from solution import offload

    monkeypatch.setattr(offload, "GC_PAUSE", "off")
    with pytest.raises(RuntimeError), offload._gc_paused():
        raise RuntimeError("training failed")
    assert gc.isenabled() and gc.get_freeze_count() == 0
