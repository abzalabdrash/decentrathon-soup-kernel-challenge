"""Runs several benchmark modules one after the other, each in its own fresh process (one Kaggle
bench session for several measurements; a module that reads nvidia-smi sees only its own process):

    python -m solution.bench_seq solution.bench_pack_mem solution.bench_attn_bwd:30

`module:arg1:arg2` passes arguments to that module.
"""

import subprocess
import sys


def main():
    for spec in sys.argv[1:]:
        module, *args = spec.split(":")
        print(f"===== python -m {module} {' '.join(args)}", flush=True)
        code = subprocess.call([sys.executable, "-m", module, *args])
        print(f"===== {module} exit {code}", flush=True)


if __name__ == "__main__":
    main()
