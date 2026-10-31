"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward on int8 tensor cores by default, in every process. The bitwise
tests of the fp16 path (test_fused_proj.py and the tests of the modules built on it) compare against
plain PyTorch, so every test starts with the int8 backward off; the int8 path has its own tests
(test_int8_bwd.py, fixture `int8_on`), which switch it on explicitly. The same for the int8 head
(head_int8.py, off by default, on with SOUP_SOLUTION_HEAD_INT8=1): every test starts with the fp16 streamed head (test_head_stream.py
compares it with the base's head), and test_head_int8.py switches the int8 head on."""

import pytest


@pytest.fixture(autouse=True)
def _fp16_mlp_backward():
    try:
        from solution import int8_bwd
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield
        return
    saved = int8_bwd._STATE["on"]
    int8_bwd._STATE["on"] = False
    yield
    int8_bwd._STATE["on"] = saved


@pytest.fixture(autouse=True)
def _fp16_head():
    try:
        from solution import head_int8
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield
        return
    saved = head_int8._STATE["on"]
    head_int8._STATE["on"] = False
    yield
    head_int8._STATE["on"] = saved
