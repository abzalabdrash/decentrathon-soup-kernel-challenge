"""Measurements behind warmup.py. Every number comes from fresh processes with an empty triton cache
(a new TRITON_CACHE_DIR each), as in a measured run.

    python -m solution.bench_warmup imports     # import cost of triton, transformers, peft, trl, bnb
    python -m solution.bench_warmup compile     # compile time of every registered warm-up
    python -m solution.bench_warmup gil         # how much of the main thread the compiling thread takes
    python -m solution.bench_warmup first-call  # first training-shape launch, without / with warm-up
    python -m solution.bench_warmup first-step  # first steps of a one-layer 4-bit model after apply()
    python -m solution.bench_warmup apply       # warm-up started before / after apply()
    python -m solution.bench_warmup load        # the real model load next to the warm-up / imports
    python -m solution.bench_warmup workers     # the warm-up with 1 to 4 worker threads
    python -m solution.bench_warmup child-exit early|thread  # exit right after starting the warm-up
    python -m solution.bench_warmup all

Kernels are compiled for fp16, the T4's dtype, on any card.
"""

import os
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMPORTS = ("triton", "transformers", "peft", "trl", "bitsandbytes")


# ---------------------------------------------------------------- training-shape launches

def fused_head_call(n, n_labelled, num_items, chunk, d=64):
    """`fused_lm_head_loss` forward and backward as training calls it (hidden size `d`: the kernel
    never sees it)."""
    import torch

    from solution.fused_ce import fused_lm_head_loss

    v = 151936
    weight = (torch.randn(v, d, device="cuda") * 0.02).half()
    hidden = torch.randn(n, d, device="cuda", requires_grad=True)
    targets = torch.randint(0, v, (n,), device="cuda")
    targets[torch.randperm(n, device="cuda")[n_labelled:]] = -100
    torch.cuda.synchronize()
    start = time.perf_counter()
    loss, *_ = fused_lm_head_loss(hidden, weight, targets, num_items, chunk)
    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def swiglu_call(tokens):
    """The example SwiGLU forward and backward on a packed batch, fp16."""
    import torch

    from solution.swiglu import swiglu

    gate = torch.randn(1, tokens, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    up = torch.randn(1, tokens, 8960, device="cuda", dtype=torch.float16, requires_grad=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    swiglu(gate, up).float().sum().backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def one_layer_model(vocab=640, seed=0):
    """One decoder layer with Qwen2.5-1.5B's shapes (hidden 1536, MLP 8960, 12 / 2 heads of 128),
    a small vocabulary, random weights, built the way Soup's model ends up: 4-bit NF4 linear layers
    with double quant and fp16 compute, `prepare_model_for_kbit_training`, LoRA r=16 on q_proj and
    v_proj. The kernels see the same dtypes, strides and constants as in the real run."""
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab, hidden_size=1536, intermediate_size=8960,
                         num_hidden_layers=1, num_attention_heads=12, num_key_value_heads=2,
                         max_position_embeddings=4096, tie_word_embeddings=True,
                         attn_implementation="sdpa")
    model = Qwen2ForCausalLM(config).half()
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if not isinstance(child, torch.nn.Linear):
                    continue
                q = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                      bias=child.bias is not None, compute_dtype=torch.float16,
                                      compress_statistics=True, quant_type="nf4")
                q.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False,
                                             compress_statistics=True, quant_type="nf4")
                if child.bias is not None:
                    q.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
                setattr(parent, name, q)
    model = model.cuda()  # quantizes the Params4bit
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "v_proj"],
                                             task_type="CAUSAL_LM"))
    model.train()
    return model


def model_step(model, lengths, seed=1):
    """One forward and backward under fp16 autocast on a right-padded batch with these row lengths,
    labels on the last two thirds of every row (the prompt part has none). Returns seconds."""
    import torch

    torch.manual_seed(seed)
    b, t = len(lengths), max(lengths)
    ids = torch.randint(0, model.config.vocab_size, (b, t), device="cuda")
    mask = (torch.arange(t, device="cuda")[None, :]
            < torch.tensor(lengths, device="cuda")[:, None]).long()
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, : t // 3] = -100
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
    loss.backward()
    torch.cuda.synchronize()
    return time.perf_counter() - start


STEP_LENGTHS = ((300, 212, 190, 124), (512, 77, 40, 9))  # 826 and 638 real tokens


# ---------------------------------------------------------------- child processes

def _key(name):
    """A registration name as a key=value key."""
    return re.sub(r"\W", "_", name)


def _child_env(cache_root):
    return dict(os.environ, TRITON_CACHE_DIR=tempfile.mkdtemp(prefix="triton-", dir=cache_root))


def _child(args, cache_root, **env):
    out = subprocess.run([sys.executable, "-m", "solution.bench_warmup", *args], cwd=REPO,
                         env=dict(_child_env(cache_root), **env), capture_output=True, text=True,
                         timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"{args}: {out.stderr[-3000:]}")
    return dict(line.split("=", 1) for line in out.stdout.split("\n") if "=" in line)


def _prelude():
    """What harness/_train.py does before apply(): torch, the CUDA context, transformers."""
    import torch

    torch.empty(1, device="cuda")
    import transformers  # noqa: F401


def _setup_warmup():
    import torch

    from solution import fused_ce, swiglu, warmup  # noqa: F401 - what apply() has imported

    warmup.training_half_dtype = lambda: torch.float16
    if os.environ.get("BENCH_WARMUP_WORKERS"):  # report_workers
        warmup.WORKERS = int(os.environ["BENCH_WARMUP_WORKERS"])
    return warmup


def child_import(ctx, name):
    import importlib

    if ctx == "torch":
        import torch  # noqa: F401
    else:
        _prelude()
    before = set(sys.modules)
    start = time.perf_counter()
    importlib.import_module(name)
    seconds = time.perf_counter() - start
    pulled = [m for m in IMPORTS if m != name and m not in before and m in sys.modules]
    print(f"seconds={seconds:.3f}\npulled={','.join(pulled)}")


def child_prelude():
    start = time.perf_counter()
    import torch

    torch_s = time.perf_counter() - start
    torch.empty(1, device="cuda")
    cuda_s = time.perf_counter() - start - torch_s
    import transformers  # noqa: F401

    print(f"torch={torch_s:.3f}\ncuda={cuda_s:.3f}\ntransformers="
          f"{time.perf_counter() - start - torch_s - cuda_s:.3f}")


def child_compile():
    """Every registered warm-up in the main thread, one after another, with the compile to cubin
    (`kernel.warmup`) and the launcher build + cubin load (`_init_handles`) timed apart."""
    _prelude()
    start = time.perf_counter()
    import triton  # noqa: F401

    print(f"import_triton={time.perf_counter() - start:.3f}")
    warmup = _setup_warmup()
    from triton.runtime.driver import driver

    start = time.perf_counter()
    driver.active.utils  # noqa: B018 - builds and loads triton's cuda_utils module
    print(f"cuda_utils={time.perf_counter() - start:.3f}")
    parts = {}
    original = warmup.compile_kernel

    def timed(kernel, *args, **kwargs):
        start = time.perf_counter()
        compiled = kernel.warmup(*args, grid=(1,), **kwargs)
        mid = time.perf_counter()
        compiled._init_handles()
        parts.setdefault("compile", []).append(mid - start)
        parts.setdefault("launcher", []).append(time.perf_counter() - mid)
        return compiled

    warmup.compile_kernel = timed
    for name, fn in warmup.WARMUPS:
        parts.clear()
        start = time.perf_counter()
        fn()
        total = time.perf_counter() - start
        print(f"{_key(name)}={total:.3f}|{len(parts.get('compile', []))}|"
              f"{sum(parts.get('compile', [])):.3f}|{sum(parts.get('launcher', [])):.3f}")
    warmup.compile_kernel = original


def _spin(stop, samples):
    """Counts in the main thread; samples (time, count) every 4096 iterations until `stop(t)`."""
    n = 0
    while True:
        n += 1
        if not n & 4095:
            t = time.perf_counter()
            samples.append((t, n))
            if stop(t):
                return


def _rate(samples, t0, t1):
    inside = [(t, n) for t, n in samples if t0 <= t <= t1]
    if len(inside) < 2:
        return None
    return (inside[-1][1] - inside[0][1]) / (inside[-1][0] - inside[0][0])


def child_gil(mode):
    """The warm-up thread while the main thread just waits (mode wait) or spins a Python counter,
    with the warm-up's own GIL switch interval (spin) or another one (spin-<ms>). Per
    registration: its wall time and, when spinning, the counter's rate against the counter alone."""
    _prelude()
    import triton  # noqa: F401

    warmup = _setup_warmup()
    if mode.startswith("spin-"):  # spin-<ms>: that switch interval instead of the warm-up's
        warmup.SWITCH_INTERVAL = float(mode[5:]) / 1000
    windows = {}

    def wrap(name, fn):
        def run():
            start = time.perf_counter()
            try:
                fn()
            finally:
                windows[name] = (start, time.perf_counter())
        return run

    warmup.WARMUPS[:] = [(name, wrap(name, fn)) for name, fn in warmup.WARMUPS]
    spinning = mode != "wait"
    samples = []
    if spinning:
        t_end = time.perf_counter() + 1.0
        _spin(lambda t: t >= t_end, samples)
        base = _rate(samples, samples[0][0], samples[-1][0])
        samples.clear()
    start = time.perf_counter()
    warmup.start_background_warmup(log=False)
    if spinning:
        _spin(lambda t: warmup._DONE.is_set(), samples)
    warmup.wait_for_warmup()
    print(f"total={time.perf_counter() - start:.3f}")
    for name, (t0, t1) in windows.items():
        line = f"{_key(name)}={t1 - t0:.3f}"
        if spinning:
            rate = _rate(samples, t0, t1)
            line += f"|{rate / base:.3f}" if rate else "|nan"
        print(line)
    if spinning:
        t0 = min(t for t, _ in windows.values())
        t1 = max(t for _, t in windows.values())
        print(f"all={t1 - t0:.3f}|{_rate(samples, t0, t1) / base:.3f}")


def child_first_call(mode):
    """First and second training-shape launches in this fresh process, after a warm-up (warm) or
    without one (cold)."""
    _prelude()
    import triton  # noqa: F401

    warmup = _setup_warmup()
    if mode == "warm":
        start = time.perf_counter()
        warmup.start_background_warmup(log=False)
        warmup.wait_for_warmup()
        print(f"warmup_s={time.perf_counter() - start:.3f}")
    print(f"head_first={fused_head_call(826, 600, 900.0, 512):.4f}")
    print(f"head_second={fused_head_call(826, 600, 900.0, 512):.4f}")
    print(f"swiglu_first={swiglu_call(826):.4f}")
    print(f"swiglu_second={swiglu_call(826):.4f}")


def child_first_step(mode):
    """The one-layer model's first and second training steps after apply(), after a warm-up (warm)
    or without one (cold)."""
    _prelude()
    from triton.runtime.jit import JITFunction  # noqa: F401 - the harness imports it before apply()

    import solution

    warmup = _setup_warmup()
    solution.apply()
    model = one_layer_model()
    if mode == "warm":
        start = time.perf_counter()
        warmup.start_background_warmup(log=False)
        warmup.wait_for_warmup()
        print(f"warmup_s={time.perf_counter() - start:.3f}")
    print(f"step_first={model_step(model, STEP_LENGTHS[0]):.4f}")
    print(f"step_second={model_step(model, STEP_LENGTHS[1]):.4f}")
    print(f"step_third={model_step(model, STEP_LENGTHS[0]):.4f}")


def child_apply(mode):
    """apply() with the warm-up started after it (late) or before it (early), the main thread
    then waiting: apply's own time and when the warm-up is done, from the start of apply()."""
    _prelude()
    from triton.runtime.jit import JITFunction  # noqa: F401 - the harness imports it before apply()

    start = time.perf_counter()
    if mode == "early":
        from solution import warmup

        warmup.start_background_warmup(log=False)
    import solution

    solution.apply()
    apply_end = time.perf_counter()
    from solution import warmup

    warmup.start_background_warmup(log=False)
    warmup.wait_for_warmup()
    done = time.perf_counter()
    print(f"apply_s={apply_end - start:.3f}\nwarmup_done_s={done - start:.3f}\n"
          f"after_apply_s={done - apply_end:.3f}")


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LOAD_IMPORTS = ("trl", "datasets")  # what Soup imports later and loading the model does not need


def _load_model():
    """The model as Soup's `_setup_transformers` builds it: 4-bit NF4 with double quant and fp16
    compute, `prepare_model_for_kbit_training`, LoRA r=16 on q_proj / v_proj."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant,
                                                 dtype=torch.float16, device_map={"": 0})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    return get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                            target_modules=["q_proj", "v_proj"],
                                            task_type="CAUSAL_LM"))


def child_load(mode):
    """Loading the real model (from the local cache) alone, with the warm-up running next to it
    (warmup, after apply()), or with `preload_in_background(LOAD_IMPORTS)` next to it (imports,
    without apply(), so that those modules are not imported yet; `seq` imports them first)."""
    _prelude()
    from triton.runtime.jit import JITFunction  # noqa: F401 - the harness imports it before apply()

    import importlib

    import peft  # noqa: F401 - Soup imports it at the start of _setup_transformers (sft.py:1613)

    if mode in ("alone", "warmup"):
        import solution

        warmup = _setup_warmup()
        solution.apply()
    start = time.perf_counter()
    if mode == "warmup":
        warmup.start_background_warmup(log=False)
    if mode == "imports":
        from solution.warmup import preload_in_background

        ready = preload_in_background(LOAD_IMPORTS, log=False)
    if mode == "seq":
        for name in LOAD_IMPORTS:
            importlib.import_module(name)
        print(f"imports_s={time.perf_counter() - start:.3f}")
    _load_model()
    print(f"load_end_s={time.perf_counter() - start:.3f}")
    if mode == "warmup":
        warmup.wait_for_warmup()
        print(f"warmup_s={warmup.warmup_report()['seconds']:.3f}")
    if mode == "imports":
        ready.wait()
        print(f"imports_s={sum(ready.seconds.values()):.3f}")
    print(f"all_done_s={time.perf_counter() - start:.3f}")


# ---------------------------------------------------------------- reports

def report_imports(cache_root, repeats=3):
    pre = [_child(["child-prelude"], cache_root) for _ in range(repeats)]
    print("harness prelude, s (median of %d): import torch %.2f, CUDA context %.2f, "
          "import transformers %.2f" % (repeats, *(statistics.median(float(p[k]) for p in pre)
                                                   for k in ("torch", "cuda", "transformers"))))
    print(f"{'module':14s} {'after torch, s':>15s} {'after prelude, s':>17s}  also loads")
    for name in IMPORTS:
        row = []
        pulled = ""
        for ctx in ("torch", "harness"):
            runs = [_child(["child-import", ctx, name], cache_root) for _ in range(repeats)]
            row.append(statistics.median(float(r["seconds"]) for r in runs))
            pulled = runs[-1]["pulled"] if ctx == "harness" else pulled
        print(f"{name:14s} {row[0]:15.2f} {row[1]:17.2f}  {pulled or '-'}")


def report_compile(cache_root):
    res = _child(["child-compile"], cache_root)
    print(f"import triton {float(res.pop('import_triton')):.2f} s (after the prelude); "
          f"triton's cuda_utils build {float(res.pop('cuda_utils')):.2f} s")
    print(f"{'warm-up':38s} {'total, s':>9s} {'compiles':>9s} {'to cubin, s':>12s} "
          f"{'launcher+load, s':>17s}")
    for name, value in res.items():
        total, count, comp, launch = value.split("|")
        print(f"{name:38s} {float(total):9.2f} {count:>9s} {float(comp):12.2f} {float(launch):17.2f}")


def report_gil(cache_root, repeats=3):
    """Every mode `repeats` times, interleaved (the machine's other load drifts); medians."""
    modes = {"waiting": "wait", "spin 5 ms": "spin-5", "spin 1 ms": "spin-1",
             "spin 0.2 ms": "spin-0.2"}
    runs = {label: [] for label in modes}
    for _ in range(repeats):
        for label, mode in modes.items():
            runs[label].append(_child(["child-gil", mode], cache_root))
    print(f"warm-up thread wall time, s, while the main thread waits or spins a Python counter "
          f"with that GIL switch interval; rate: the counter's rate during it against the counter "
          f"alone (medians of {repeats}; every run of the whole warm-up below)")
    names = [name for name in runs["waiting"][0] if name != "total"] + ["all"]
    print(f"{'warm-up':36s} {'waiting':>8s}" + "".join(f" {label:>12s} {'rate':>5s}"
                                                      for label in modes if label != "waiting"))
    for name in names:
        key = "total" if name == "all" else name
        waits = sorted(float(r[key]) for r in runs["waiting"])
        line = f"{name:36s} {statistics.median(waits):8.2f}"
        for label in modes:
            if label == "waiting":
                continue
            pairs = [r[name].split("|") for r in runs[label]]
            secs = sorted(float(a) for a, _ in pairs)
            rates = [float(b) for _, b in pairs if b != "nan"]
            line += f" {statistics.median(secs):12.2f} {statistics.median(rates):5.2f}"
        print(line)
    for label in modes:
        key = "total" if label == "waiting" else "all"
        secs = sorted(float(r[key].split("|")[0]) for r in runs[label])
        print(f"  {label}: all {', '.join(f'{x:.2f}' for x in secs)} s")


def report_first_call(cache_root):
    cold = _child(["child-first-call", "cold"], cache_root)
    warm = _child(["child-first-call", "warm"], cache_root)
    cold_first = float(cold["head_first"]) + float(cold["swiglu_first"])
    warm_first = float(warm["head_first"]) + float(warm["swiglu_first"])
    steady = float(warm["head_second"]) + float(warm["swiglu_second"])
    print(f"fused head (826 tokens, 512-row chunks) first call: cold {float(cold['head_first']):.3f} s, "
          f"after warm-up {float(warm['head_first']):.3f} s, steady {float(warm['head_second']):.3f} s")
    print(f"SwiGLU fwd+bwd (826 x 8960) first call: cold {float(cold['swiglu_first']):.3f} s, "
          f"after warm-up {float(warm['swiglu_first']):.3f} s, steady {float(warm['swiglu_second']):.3f} s")
    print(f"warm-up thread took {float(warm['warmup_s']):.2f} s")
    print(f"cold_first_s={cold_first:.4f} warm_first_s={warm_first:.4f} steady_s={steady:.4f}")


def report_first_step(cache_root):
    cold = _child(["child-first-step", "cold"], cache_root)
    warm = _child(["child-first-step", "warm"], cache_root)
    print("one-layer 4-bit model after apply(), steps on 826 / 638 / 826 tokens, s:")
    for label, res in (("cold", cold), ("after warm-up", warm)):
        print(f"  {label:14s} first {float(res['step_first']):.3f}, second "
              f"{float(res['step_second']):.3f}, third {float(res['step_third']):.3f}")
    print(f"  warm-up thread took {float(warm['warmup_s']):.2f} s")


def report_apply(cache_root, repeats=3):
    runs = {mode: [_child(["child-apply", mode], cache_root) for _ in range(repeats)]
            for mode in ("late", "early")}
    print(f"apply() and the warm-up, main thread waiting after apply(); medians of {repeats}, s:")
    for mode, res in runs.items():
        med = {k: statistics.median(float(r[k]) for r in res)
               for k in ("apply_s", "warmup_done_s", "after_apply_s")}
        print(f"  warm-up started {mode:5s}: apply {med['apply_s']:.2f}, warm-up done at "
              f"{med['warmup_done_s']:.2f}, {med['after_apply_s']:.2f} after apply()")


def report_load(cache_root, repeats=2):
    """Needs the model in the local Hugging Face cache and about 1.5 GB of free GPU memory."""
    modes = ("alone", "warmup", "seq", "imports")
    runs = {mode: [] for mode in modes}
    for _ in range(repeats):
        for mode in modes:
            runs[mode].append(_child(["child-load", mode], cache_root))

    def med(mode, key):
        return statistics.median(float(r[key]) for r in runs[mode])

    print(f"loading {MODEL} in 4 bits + LoRA, medians of {repeats}, s from the start of the load:")
    print(f"  after apply(), alone: load {med('alone', 'load_end_s'):.2f}")
    print(f"  after apply(), warm-up next to it: load {med('warmup', 'load_end_s'):.2f}, warm-up "
          f"{med('warmup', 'warmup_s'):.2f}, both done {med('warmup', 'all_done_s'):.2f}")
    print(f"  {', '.join(LOAD_IMPORTS)} imported first ({med('seq', 'imports_s'):.2f}), then the "
          f"load: both done {med('seq', 'all_done_s'):.2f}")
    print(f"  {', '.join(LOAD_IMPORTS)} imported next to the load: load "
          f"{med('imports', 'load_end_s'):.2f}, imports {med('imports', 'imports_s'):.2f}, both "
          f"done {med('imports', 'all_done_s'):.2f}")


def report_workers(cache_root, repeats=2):
    """The whole warm-up with 1 to 4 worker threads, main thread waiting or spinning a counter."""
    print(f"whole warm-up by worker threads, medians of {repeats}, s (counter rate against alone):")
    for workers in (1, 2, 3, 4):
        env = {"BENCH_WARMUP_WORKERS": str(workers)}
        wait, spin, rate = [], [], []
        for _ in range(repeats):
            wait.append(float(_child(["child-gil", "wait"], cache_root, **env)["total"]))
            a, b = _child(["child-gil", "spin"], cache_root, **env)["all"].split("|")
            spin.append(float(a))
            rate.append(float(b))
        print(f"  {workers} worker(s): waiting {statistics.median(wait):.2f}, spinning "
              f"{statistics.median(spin):.2f} (rate {statistics.median(rate):.2f})")


def child_exit(mode):
    """Starts the warm-up as apply() does (`early`: started at the beginning of apply() and closed
    at its end; `thread`: started at the end) and returns at once: the process must exit with 0
    while the warm-up is still compiling (see warmup._shutdown)."""
    _prelude()
    warmup = _setup_warmup()
    if mode == "early":
        warmup.start_background_warmup(log=False, final=False)
    warmup.start_background_warmup(log=False)
    time.sleep(1.0)  # into the compiles (without warmup._shutdown: exit code 134 in 1 of 2 runs)
    print(f"done_at_exit={warmup.wait_for_warmup(0)}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    children = {"child-import": child_import, "child-prelude": child_prelude,
                "child-compile": child_compile, "child-gil": child_gil,
                "child-first-call": child_first_call, "child-first-step": child_first_step,
                "child-apply": child_apply, "child-load": child_load,
                "child-exit": child_exit}
    if cmd in children:
        children[cmd](*sys.argv[2:])
        return
    reports = {"imports": report_imports, "compile": report_compile, "gil": report_gil,
               "first-call": report_first_call, "first-step": report_first_step,
               "apply": report_apply, "load": report_load, "workers": report_workers}
    root = os.environ.get("TRITON_CACHE_DIR") or tempfile.mkdtemp(prefix="bench-warmup-")
    os.makedirs(root, exist_ok=True)
    for name in ([r for r in reports if r != "load"] if cmd == "all" else [cmd]):
        print(f"== {name}")
        reports[name](root)


if __name__ == "__main__":
    main()
