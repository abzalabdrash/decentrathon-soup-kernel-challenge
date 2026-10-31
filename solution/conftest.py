"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward and int8_fwd.py the MLP's gate|up forward on int8 tensor cores by
default, in every process. The bitwise tests of the fp16 path (test_fused_proj.py and the tests of
the modules built on it) compare against plain PyTorch, so every test starts with both int8 paths
off; the int8 paths have their own tests (test_int8_bwd.py, test_int8_fwd.py, fixture `int8_on`),
which switch them on explicitly."""

import pytest


@pytest.fixture(autouse=True)
def _fp16_mlp_backward():
    try:
        from solution import int8_bwd, int8_fwd
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield
        return
    saved = int8_bwd._STATE["on"], int8_fwd._STATE["on"]
    int8_bwd._STATE["on"] = int8_fwd._STATE["on"] = False
    yield
    int8_bwd._STATE["on"], int8_fwd._STATE["on"] = saved
