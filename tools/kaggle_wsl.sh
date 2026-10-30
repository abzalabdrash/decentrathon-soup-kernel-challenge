#!/bin/bash
# Local helper (this team's Windows laptop): run tools/kaggle_run.py from WSL under one of the two
# team accounts. Run it from the root of the checkout you want measured, in Git Bash:
#
#   export MSYS_NO_PATHCONV=1
#   wsl -d Ubuntu -- bash tools/kaggle_wsl.sh <1|2> --mode solution --slug soup-tNN --commit <sha> [--env K=V ...]
#
# 1 = account namedesign (token file kaggle1.token), 2 = account peagverr (kaggle2.token); both have the
# private dataset soup-secrets with the GitHub read token. The commit must be pushed to origin first.
# Modes: solution (~7 min, our run only), diag (solution run with SOUP_SOLUTION_DIAG=1: memory snapshot,
# chrome trace, import times), full (base + solution + the 10 checks, ~10 min), profile.
# Results: runs_kaggle/<stamp>-<mode>-<slug>/output/ (gitignored). The CLI needs Python >= 3.11, hence WSL.
acct=$1; shift
case "$acct" in
  1) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle1.token; user=namedesign ;;
  2) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle2.token; user=peagverr ;;
  *) echo "first argument: account 1 (namedesign) or 2 (peagverr)"; exit 2 ;;
esac
export KAGGLE_API_TOKEN=$(tr -d '\r\n ' < "$tok")
export PATH=$HOME/miniforge3/envs/soup/bin:$PATH
python tools/kaggle_run.py --user "$user" --repo https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers.git --poll 30 "$@"
echo "KAGGLE_RUN_DONE $user $*"
