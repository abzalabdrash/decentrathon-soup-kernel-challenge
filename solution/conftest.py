"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward on int8 tensor cores by default, in every process. The bitwise
tests of the fp16 path (test_fused_proj.py and the tests of the modules built on it) compare against
plain PyTorch, so every test starts with the int8 backward off; the int8 path has its own tests
(test_int8_bwd.py, fixture `int8_on`), which switch it on explicitly."""

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
