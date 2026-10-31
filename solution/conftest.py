"""pytest setup of solution/'s own tests (Soup's regression tests in tests/ do not load it).

int8_bwd.py runs the MLP backward on int8 tensor cores by default, in every process, and
offload_int8.py sends the MLP's (gate|up) to the host as int8. The bitwise tests of the fp16 path
(test_fused_proj.py and the tests of the modules built on it) compare against plain PyTorch, so every
test starts with both off; the int8 paths have their own tests (test_int8_bwd.py,
test_offload_int8.py, fixture `int8_on`), which switch them on explicitly."""

import pytest


@pytest.fixture(autouse=True)
def _fp16_mlp_backward():
    try:
        from solution import int8_bwd
    except Exception:  # noqa: BLE001 - no triton or no GPU stack: nothing to pin
        yield
        return
    from solution import offload_int8

    saved = int8_bwd._STATE["on"], offload_int8._STATE["on"]
    int8_bwd._STATE["on"] = offload_int8._STATE["on"] = False
    yield
    int8_bwd._STATE["on"], offload_int8._STATE["on"] = saved
