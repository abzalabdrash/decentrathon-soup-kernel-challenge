# T13 finding: check 10 can fail at random on main (b5259a7): the warmup thread aborts the pytest process at exit

`SOUP_APPLY_SOLUTION=1 pytest tests -q` (what harness/check_rules.py run_tests() runs; it requires
returncode == 0 AND "79 passed"), WSL, clean `git archive HEAD` copy of b5259a7, 6 runs each:

- default switches: rc=134 in 3 of 6 runs, every time after "79 passed", stderr
  "terminate called without an active exception" (a joinable std::thread destroyed at exit);
- SOUP_SOLUTION_WARMUP=0: rc=0 in 6 of 6.

The 79 tests take 0.6 s; apply() starts warmup.start_background_warmup(), which is still compiling
Triton kernels (5+ s) when the interpreter finalizes. Fix belongs in warmup.py (T15): e.g. do not
start the background warmup unless a real CUDA training run is coming (the tests never build a real
model), or join/cancel it at exit (atexit that waits for the compile thread or skips the rest).
check 10 = 0 points for the whole correctness block (20) on the jury's run if it hits.

Script used: scratchpad soup_rc.sh: `out=$(bash tools/wsl_pytest.sh soup 2>&1); rc=$?`.
