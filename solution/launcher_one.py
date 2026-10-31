"""The warm-up's kernel launchers built by one C compiler run instead of one per launcher
(SOUP_SOLUTION_LAUNCHERS_ONE=1, off by default until measured).

triton 3.6 launches a compiled kernel through a small C extension module generated for the kernel's
signature (triton/backends/nvidia/driver.py `make_launcher`: argument parsing with the Python C API,
then cuLaunchKernelEx) and builds it on first use with the system C compiler, one `cc ... -shared`
process per distinct signature (`CudaLauncher.__init__` -> `compile_module_from_src`, build.py
`_build`). On the T4 the warm-up's ~13 launcher builds were the largest single part of its time:
in the startup profile of soup-l1-p0 the two workers spent 2.1 s of the model setup alone waiting
for these compiler processes, about 0.2 CPU-s each on a 4-vCPU box whose main thread loads the
model at the same time. Most of a build is per process (driver, assembler, linker of a shared
object): on the WSL box one launcher takes 0.22 s, its preprocessing 0.04 s, and 13 launchers in
one translation unit 0.34 s.

Here, when the warm-up starts (warmup.py `_run`, after triton's driver and cuda_utils are set up):
- every queued registration that compiles Triton kernels runs once with triton's own
  `knobs.runtime.jit_cache_hook` answering "skip" for this thread only (jit.py `_do_compile`: the
  hook sees the launch's signature; nothing is compiled, cached or launched), which gives the
  signatures the warm-up will compile; their launcher sources come from triton's `make_launcher`;
- a thread writes one C file: the headers once, then each distinct source with its file-scope names
  renamed by `#define` (`launch` -> `launch_3`, `PyInit___triton_launcher` -> `..._3`, ...: the
  template's static functions, variables and typedefs), and a module init that calls each renamed
  init and keeps the module it returns; it builds that file with the warm-up's launcher build (the
  same compiler, flags, include and library directories) and imports it;
- `compile_module_from_src` in triton's CUDA driver module serves a launcher whose source text is
  one of those with the module its own init created (each launcher keeps its own module object,
  method table and statics; triton only reads `mod.launch`), waiting for the build if it is still
  running. Any other source, another module name, or a build that failed (or does not compile) take
  triton's own path, as does everything when the switch is off.

The kernels (cubins from ptxas) and the launch code are triton's; only the number of compiler runs
changes. The probe runs the registrations' Python once more (a few milliseconds each).
"""

import importlib.machinery
import importlib.util
import os
import atexit
import re
import tempfile
import threading
import time

LAST = {"sources": 0, "seconds": None, "served": 0, "fallbacks": 0, "error": None, "waited": 0.0}
_STATE = {"sources": {}, "done": None, "module": None, "dir": None, "builds": 0, "original": None}
_LOCK = threading.Lock()
# file-scope names of triton 3.6's launcher template (driver.py make_launcher); the regexes below add
# any the template of another version defines
_KNOWN = ("PyCUtensorMapObject", "gpuAssert", "cuLaunchKernelEx_t", "getLaunchKernelExHandle", "_launch",
          "_DevicePtrInfo", "DevicePtrInfo", "data_ptr_str", "py_tensor_map_type", "getPointer", "getTmaDesc",
          "ensureCudaContext", "pack_fp16", "pack_bf16", "pack_fp32", "pack_fp64", "launch", "ModuleMethods",
          "ModuleDef", "PyInit___triton_launcher")
_NAME_PATTERNS = (
    re.compile(r"^static\b[^(=;{\[]*?\b([A-Za-z_]\w*)\s*[(=;\[]", re.M),  # static functions and variables
    re.compile(r"^}\s*([A-Za-z_]\w*)\s*;", re.M),  # typedef struct { ... } Name;
    re.compile(r"\btypedef\s+struct\s+([A-Za-z_]\w*)"),  # struct tags
    re.compile(r"\btypedef\s+\w+\s*\(\s*\*\s*([A-Za-z_]\w*)\s*\)"),  # function pointer typedefs
    re.compile(r"^PyMODINIT_FUNC\s+([A-Za-z_]\w*)\s*\(", re.M),
)
LAUNCHER = "__triton_launcher"


def enabled():
    return os.environ.get("SOUP_SOLUTION_LAUNCHERS_ONE", "0").strip().lower() not in ("0", "false", "no", "")


def _key(text):
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the probe

# the modules whose warm-up registrations only compile (compile_kernel with dtypes and sizes: nothing
# allocated, launched or drawn), so running one once more in the probe changes nothing
COMPILE_ONLY_MODULES = frozenset(f"solution.{name}" for name in (
    "warmup", "rmsnorm", "nf4", "rope", "swiglu", "fused_ce", "fused_proj", "head_stream", "head_pack", "int8_bwd"))


def probe_safe(fn):
    """Whether the probe may run the registration `fn` once more: a warm-up defined in one of
    COMPILE_ONLY_MODULES (cublas and step_ops, which launch, are left out by warmup.py)."""
    import functools

    while isinstance(fn, functools.partial):
        fn = fn.func
    return (getattr(fn, "__module__", "") or "") in COMPILE_ONLY_MODULES


def probe_sources(registrations):
    """{source key: launcher source} of the launches `registrations` ([(name, fn)]) compile, from one
    run of each with triton's jit_cache_hook skipping the compile for this thread."""
    from triton import knobs
    from triton.backends.nvidia import driver as nvidia

    ident = threading.get_ident()
    signatures = []
    previous = knobs.runtime.jit_cache_hook

    def hook(**kwargs):
        if threading.get_ident() == ident:
            info = kwargs.get("compile") or {}
            signature = info.get("signature")
            if isinstance(signature, dict):
                signatures.append(dict(signature))
            return True  # skip: nothing compiled, cached or launched
        return previous(**kwargs) if previous is not None else None

    knobs.runtime.jit_cache_hook = hook
    try:
        for _, fn in registrations:
            try:
                fn()
            except Exception:  # noqa: BLE001 - its launchers take triton's own build then
                pass
    finally:
        if knobs.runtime.jit_cache_hook is hook:
            knobs.runtime.jit_cache_hook = previous
    sources = {}
    for signature in signatures:
        if any(isinstance(ty, str) and ty.startswith("tensordesc") for ty in signature.values()):
            continue  # tensor descriptors need the compiled kernel's metadata
        text = nvidia.make_launcher({}, signature, None)
        sources.setdefault(_key(text), text)
    return sources


# ---------------------------------------------------------------- the combined source

def file_scope_names(text):
    names = set(_KNOWN)
    for pattern in _NAME_PATTERNS:
        names.update(pattern.findall(text))
    return sorted(names)


def combined_source(texts, module):
    """One translation unit: the template's headers, each launcher source with its file-scope names
    renamed, and `PyInit_<module>` returning a module whose attribute `l<i>` is the module the i-th
    launcher's own init created."""
    head = ('#include "cuda.h"\n#include <dlfcn.h>\n#include <stdbool.h>\n#include <stdlib.h>\n'
            '#define PY_SSIZE_T_CLEAN\n#include <Python.h>\n')
    parts = [head]
    for i, text in enumerate(texts):
        names = file_scope_names(text)
        parts.append("".join(f"#define {name} {name}_soup{i}\n" for name in names))
        parts.append(text)
        parts.append("".join(f"#undef {name}\n" for name in names))
    inits = "".join(
        f"  sub = PyInit___triton_launcher_soup{i}();\n"
        f"  if (sub == NULL || PyModule_AddObject(m, \"l{i}\", sub) < 0) {{ Py_XDECREF(sub); Py_DECREF(m); return NULL; }}\n"
        for i in range(len(texts)))
    parts.append(
        f"\nstatic struct PyModuleDef SoupLaunchersDef = {{ PyModuleDef_HEAD_INIT, \"{module}\", NULL, -1, NULL }};\n"
        f"PyMODINIT_FUNC PyInit_{module}(void) {{\n"
        f"  PyObject *m = PyModule_Create(&SoupLaunchersDef);\n  PyObject *sub;\n  if (m == NULL) return NULL;\n"
        f"{inits}  return m;\n}}\n")
    return "".join(parts)


# ---------------------------------------------------------------- build and serve

def _build_and_load(texts, build):
    from triton.backends.nvidia import driver as nvidia

    with _LOCK:
        _STATE["builds"] += 1
        module = f"soup_launchers_{os.getpid()}_{_STATE['builds']}"
    directory = _STATE["dir"] = tempfile.mkdtemp(prefix="soup-launchers-one-")
    src = os.path.join(directory, f"{module}.c")
    with open(src, "w", encoding="utf-8") as f:
        f.write(combined_source(texts, module))
    args = (module, src, directory, list(nvidia.library_dirs()), list(nvidia.include_dirs), list(nvidia.libraries))
    if build is None:
        from triton.runtime.build import _build  # triton's own -O3 build (the launchers pass no ccflags)

        so = _build(*args, [])
    elif getattr(build, "_soup_launcher_build", None):
        so = build(*args, ahead=False)  # warmup.py's launcher build (-O0 by default)
    else:
        so = build(*args)
    loader = importlib.machinery.ExtensionFileLoader(module, so)
    spec = importlib.util.spec_from_file_location(module, so, loader=loader)
    combined = importlib.util.module_from_spec(spec)
    loader.exec_module(combined)
    return combined


def _serve(original):
    def compile_module_from_src(src, name, *args, **kwargs):
        if name == LAUNCHER and isinstance(src, str):
            index = _STATE["sources"].get(_key(src))
            done = _STATE["done"]
            if index is not None and done is not None:
                started = time.perf_counter()
                done.wait()
                LAST["waited"] += time.perf_counter() - started
                combined = _STATE["module"]
                launcher = getattr(combined, f"l{index}", None) if combined is not None else None
                if launcher is not None:
                    LAST["served"] += 1
                    return launcher
            LAST["fallbacks"] += 1
        return original(src, name, *args, **kwargs)

    compile_module_from_src._soup_launcher_one = True
    compile_module_from_src._soup_original = original
    return compile_module_from_src


def install_server():
    """triton's CUDA driver module calls `compile_module_from_src` through ours (once). Returns
    whether it is in place."""
    try:
        from triton.backends.nvidia import driver as nvidia

        current = nvidia.compile_module_from_src
        if getattr(current, "_soup_launcher_one", False):
            return True
        _STATE["original"] = current
        nvidia.compile_module_from_src = _serve(current)
        return True
    except Exception as exc:  # noqa: BLE001 - triton's own builds then
        LAST["error"] = f"install: {exc!r}"
        return False


def start(registrations, build=None):
    """Probes `registrations` for their launchers and starts the thread that builds them in one
    compiler run; launchers asked for meanwhile wait for it. Returns the thread, or None."""
    started = time.perf_counter()
    try:
        if not install_server():
            return None
        sources = probe_sources(registrations)
    except Exception as exc:  # noqa: BLE001 - triton's own builds then
        LAST["error"] = f"probe: {exc!r}"
        return None
    if not sources:
        return None
    texts = list(sources.values())
    done = threading.Event()
    with _LOCK:
        _STATE["module"] = None
        _STATE["done"] = done
        _STATE["sources"] = {key: i for i, key in enumerate(sources)}
    LAST["sources"] = len(texts)

    def run():
        try:
            _STATE["module"] = _build_and_load(texts, build)
        except Exception as exc:  # noqa: BLE001 - every launcher takes triton's own build
            LAST["error"] = f"build: {exc!r}"[:500]
        finally:
            LAST["seconds"] = round(time.perf_counter() - started, 3)
            done.set()

    thread = threading.Thread(target=run, name="solution-launchers-one", daemon=True)
    atexit.register(remove_build_dir)
    thread.start()
    return thread


def remove_build_dir():
    path = _STATE.get("dir")
    if path:
        import shutil

        shutil.rmtree(path, ignore_errors=True)


def report():
    """One line for the warm-up's log."""
    if not LAST["sources"] and not LAST["error"]:
        return ""
    return (f"launchers in one build: {LAST['sources']} sources in {LAST['seconds'] or 0:.2f} s from the warm-up's "
            f"start, {LAST['served']} served, {LAST['fallbacks']} by triton's own build, waited {LAST['waited']:.2f} s"
            + (f", {LAST['error']}" if LAST["error"] else ""))

