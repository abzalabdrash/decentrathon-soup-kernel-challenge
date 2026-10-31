"""Timestamps of a process's exit stages, for bench_exit.py (diagnostics only; nothing imports this).

    python solution/exit_probe.py harness/_train.py --config ... [--solution]

runs the script as `python harness/_train.py ...` would (sys.path[0] is the script's directory, the
script runs as __main__), and writes one line to stderr at the very end of the interpreter's
finalization:

    [exit-probe] start=<t> train_begin=<t> train_end=<t> save_end=<t> main_end=<t>
                 atexit:<name>=<t0>+<dt> ... atexit_end=<t> gc:<gen>=<t0>+<dt> modules_cleared=<t>
                 finalize_late=<t>

with every time CLOCK_MONOTONIC (time.monotonic(), the same clock in every process of the machine),
so the parent can put them on one line with its own Popen and wait() times. What is timed:
- train_begin / train_end / save_end: transformers.Trainer.train and save_model (patched as soon as
  transformers.trainer is imported, through the audit hook's "import" event): the same points in the
  base and in our runs;
- main_end: the script's main() returning or raising SystemExit (after metrics.json is written);
- every atexit callback (atexit.register and threading._register_atexit are wrapped from the start,
  so every later registration is timed; the earliest registration runs last) and threading._shutdown
  (the join of non-daemon threads before the atexit callbacks);
- every garbage collection after main_end (gc.callbacks), i.e. the finalization's full collections;
- modules_cleared / finalize_late: the __del__ of sentinel objects in the dicts of the threading and
  sys modules, i.e. the moments those dicts are cleared in the module teardown (sys is wiped
  last, at the end of finalize_modules): what follows is C (static destructors, the
  CUDA runtime) and the kernel's exit (bench_exit.py polls /proc for that part). The audit event
  cpython._PySys_ClearAuditHooks would be later, but it reaches C hooks only.
Everything the late code uses is bound in closures: module dicts and builtins are gone by then.
The atexit part is also written at the end of the atexit callbacks ([exit-probe-atexit] line).
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0] or os.curdir) == _HERE:
    sys.path.pop(0)  # as `python <script>`: the script's directory first, not solution/

import atexit  # noqa: E402
import gc  # noqa: E402
import runpy  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402


def _install():
    now, records, modules = time.monotonic, [], sys.modules
    add, write, join, error = records.append, os.write, " ".join, Exception
    state = {"after_main": False, "patched": False, "gc_t0": 0.0}
    add("start=%.6f" % now())

    def timed(kind, func):
        name = "%s.%s" % (getattr(func, "__module__", None) or "?",
                          getattr(func, "__qualname__", None) or type(func).__name__)
        name = name.replace(" ", "_")

        def run(*args, **kwargs):
            t0 = now()
            try:
                return func(*args, **kwargs)
            finally:
                add("%s:%s=%.6f+%.6f" % (kind, name, t0, now() - t0))

        run.probe_of = func
        return run

    original_register, original_unregister = atexit.register, atexit.unregister
    registered = []

    def register(func, *args, **kwargs):
        wrapper = timed("atexit", func)
        registered.append(wrapper)
        original_register(wrapper, *args, **kwargs)
        return func

    def unregister(func):
        for wrapper in [w for w in registered if w.probe_of == func]:
            original_unregister(wrapper)
            registered.remove(wrapper)
        original_unregister(func)

    atexit.register, atexit.unregister = register, unregister
    if hasattr(threading, "_register_atexit"):
        original_thread_atexit = threading._register_atexit

        def thread_atexit(func, *args, **kwargs):
            return original_thread_atexit(timed("thread_atexit", func), *args, **kwargs)

        threading._register_atexit = thread_atexit
    if hasattr(threading, "_shutdown"):
        threading._shutdown = timed("threads", threading._shutdown)

    def on_gc(phase, info):
        if state["after_main"]:
            if phase == "start":
                state["gc_t0"] = now()
            else:
                add("gc:%d=%.6f+%.6f" % (info["generation"], state["gc_t0"], now() - state["gc_t0"]))

    gc.callbacks.append(on_gc)

    def patch_trainer():
        state["patched"] = True
        cls = modules["transformers.trainer"].Trainer
        for method, begin, end in (("train", "train_begin", "train_end"), ("save_model", None, "save_end")):
            original = getattr(cls, method)

            def wrapper(self, *args, _original=original, _begin=begin, _end=end, **kwargs):
                if _begin:
                    add("%s=%.6f" % (_begin, now()))
                try:
                    return _original(self, *args, **kwargs)
                finally:
                    add("%s=%.6f" % (_end, now()))

            setattr(cls, method, wrapper)

    def hook(event, args):
        try:
            if event == "import":
                if not state["patched"] and "transformers.trainer" in modules:
                    trainer = modules["transformers.trainer"]
                    if trainer is not None and hasattr(trainer, "Trainer"):
                        patch_trainer()
        except error:  # a probe must not break the audited operation
            pass

    sys.addaudithook(hook)

    def main_end():
        state["after_main"] = True
        add("main_end=%.6f" % now())
        write(2, b"[exit-probe-main-end]\n")  # the parent starts polling /proc
        if os.environ.get("EXIT_PROBE_FINALIZERS"):  # what weakref.finalize will run at exit, by callback
            import collections
            import weakref

            counts = collections.Counter(
                "%s.%s" % (getattr(info.func, "__module__", "?"), getattr(info.func, "__qualname__", "?"))
                for info in list(weakref.finalize._registry.values()) if info.atexit)
            write(2, ("[exit-probe-finalizers] %s\n" % sorted(counts.items(), key=lambda kv: -kv[1])).encode())

    def atexit_end():
        add("atexit_end=%.6f" % now())
        write(2, ("\n[exit-probe-atexit] " + join(records) + "\n").encode())

    original_register(atexit_end)  # the first registration runs last

    class Sentinel:
        """Its __del__ stamps the moment the dict holding it is cleared in the module teardown."""

        def __init__(self, label, final=False):
            self.label, self.final = label, final

        def __del__(self):
            add("%s=%.6f" % (self.label, now()))
            if self.final:
                write(2, ("\n[exit-probe] " + join(records) + "\n").encode())

    # threading's dict is cleared when the module goes (most modules are gone by then); sys's dict
    # is wiped at the very end of the module teardown (finalize_clear_sys_builtins_dict). A sentinel
    # in builtins' dict never fires once torch is imported.
    threading.__dict__["_probe_threading_dict"] = Sentinel("modules_cleared")
    sys.__dict__["_probe_sys_dict"] = Sentinel("finalize_late", final=True)
    return main_end


def main():
    main_end = _install()
    script = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(script))
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        main_end()


if __name__ == "__main__":
    main()
