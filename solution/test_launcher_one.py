"""SOUP_SOLUTION_LAUNCHERS_ONE (launcher_one.py): kernels compiled by the warm-up launch through
launchers built in one compiler run, with the same results as through triton's own launchers."""

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

import triton.language as tl  # noqa: E402

from solution import launcher_one, warmup  # noqa: E402


@triton.jit
def _axpy(x_ptr, y_ptr, out_ptr, alpha, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, (alpha * x + y).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _scale_rows(src_ptr, dst_ptr, scale_ptr, rows, cols, stride: tl.int64, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    s = tl.load(scale_ptr + row)
    v = tl.load(src_ptr + row * stride + offs, mask=mask)
    tl.store(dst_ptr + row * stride + offs, v * s, mask=mask)


@triton.jit
def _negate(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, -tl.load(src_ptr + offs, mask=mask), mask=mask)


@pytest.fixture
def fresh_server(monkeypatch):
    from triton.backends.nvidia import driver as nvidia

    current = nvidia.compile_module_from_src
    monkeypatch.setattr(nvidia, "compile_module_from_src", getattr(current, "_soup_original", current))
    saved = dict(launcher_one._STATE)
    for key in ("served", "fallbacks", "sources", "waited"):
        launcher_one.LAST[key] = 0
    launcher_one.LAST["error"] = None
    yield nvidia
    launcher_one._STATE.update(saved)


def _registrations(n_float):
    def warm_axpy():
        warmup.compile_kernel(_axpy, torch.float16, torch.float32, torch.float16, 0.5, 4096, BLOCK=1024)
        warmup.compile_kernel(_axpy, torch.float32, torch.float32, torch.float32, 0.5, 4097, BLOCK=256)

    def warm_scale():
        warmup.compile_kernel(_scale_rows, torch.float32, torch.float32, torch.float32, 16, 100, 128, BLOCK=128)

    return [("axpy", warm_axpy), ("scale", warm_scale)]


def test_combined_launchers_launch(fresh_server, tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    thread = launcher_one.start(_registrations(2))
    assert thread is not None, launcher_one.LAST
    thread.join()
    assert launcher_one.LAST["error"] is None, launcher_one.LAST["error"]
    assert launcher_one.LAST["sources"] >= 2
    for _, fn in _registrations(2):
        fn()  # compiles, and builds each launcher from the combined module
    assert launcher_one.LAST["served"] >= 3 and launcher_one.LAST["fallbacks"] == 0

    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4096, device="cuda", generator=gen).half()
    y = torch.randn(4096, device="cuda", generator=gen)
    out = torch.empty(4096, device="cuda", dtype=torch.float16)
    _axpy[(4,)](x, y, out, 0.5, 4096, BLOCK=1024)
    torch.testing.assert_close(out, (0.5 * x.float() + y).half(), rtol=0, atol=0)

    x32 = torch.randn(4097, device="cuda", generator=gen)
    out32 = torch.empty_like(x32)
    _axpy[(triton.cdiv(4097, 256),)](x32, torch.ones(4097, device="cuda"), out32, 0.5, 4097, BLOCK=256)
    torch.testing.assert_close(out32, 0.5 * x32 + 1.0, rtol=0, atol=0)

    src = torch.randn(16, 128, device="cuda", generator=gen)
    scale = torch.randn(16, device="cuda", generator=gen)
    dst = torch.zeros_like(src)
    _scale_rows[(16,)](src, dst, scale, 16, 100, 128, BLOCK=128)
    torch.testing.assert_close(dst[:, :100], src[:, :100] * scale[:, None], rtol=0, atol=0)
    assert (dst[:, 100:] == 0).all()


def test_unknown_source_takes_triton_build(fresh_server):
    nvidia = fresh_server
    assert launcher_one.install_server()
    launcher_one._STATE["sources"] = {}
    launcher_one._STATE["done"] = None
    text = nvidia.make_launcher({}, {0: "*fp32", 1: "i64", 2: "fp32", 3: "i32"}, None)
    mod = nvidia.compile_module_from_src(src=text, name=launcher_one.LAUNCHER, library_dirs=nvidia.library_dirs(),
                                         include_dirs=nvidia.include_dirs, libraries=nvidia.libraries)
    assert callable(mod.launch)
    assert launcher_one.LAST["fallbacks"] == 1 and launcher_one.LAST["served"] == 0


def test_failed_build_falls_back(fresh_server, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("no compiler")

    broken._soup_launcher_build = ("0", 0)

    def warm_negate():
        warmup.compile_kernel(_negate, torch.float32, torch.float32, 1000, BLOCK=256)

    thread = launcher_one.start([("negate", warm_negate)], build=broken)
    assert thread is not None
    thread.join()
    assert "no compiler" in launcher_one.LAST["error"]
    warm_negate()  # its launcher from triton's own build
    assert launcher_one.LAST["served"] == 0 and launcher_one.LAST["fallbacks"] == 1
    src = torch.arange(1000, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)
    _negate[(4,)](src, dst, 1000, BLOCK=256)
    assert torch.equal(dst, -src)


def test_file_scope_names_cover_template():
    from triton.backends.nvidia import driver as nvidia

    text = nvidia.make_launcher({}, {0: "*fp16", 1: "i32", 2: "fp32", 3: "constexpr"}, None)
    names = launcher_one.file_scope_names(text)
    for name in ("launch", "_launch", "getPointer", "ModuleDef", "ModuleMethods", "PyInit___triton_launcher",
                 "pack_fp16", "DevicePtrInfo", "_DevicePtrInfo", "PyCUtensorMapObject", "data_ptr_str"):
        assert name in names


def test_switch_off_by_default(monkeypatch):
    monkeypatch.delenv("SOUP_SOLUTION_LAUNCHERS_ONE", raising=False)
    assert not launcher_one.enabled()
    monkeypatch.setenv("SOUP_SOLUTION_LAUNCHERS_ONE", "1")
    assert launcher_one.enabled()


def test_probe_takes_the_kernel_modules_warmups_only():
    import functools

    from solution import fused_proj, head_stream, warmup as warmup_module

    assert launcher_one.probe_safe(warmup_module.warm_rmsnorm_fwd)
    assert launcher_one.probe_safe(fused_proj._warm_masked_pair_sum)
    assert launcher_one.probe_safe(functools.partial(head_stream._warm_kernels, 4096))

    def elsewhere():
        pass

    assert not launcher_one.probe_safe(elsewhere)  # this module: not one of the kernel modules
