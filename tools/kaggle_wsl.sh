#!/bin/bash
# Local helper (this team's Windows laptop): run tools/kaggle_run.py from WSL under one of the two
# team accounts. Run it from the root of the checkout you want measured, in Git Bash:
#
#   export MSYS_NO_PATHCONV=1
#   wsl -d Ubuntu -- bash tools/kaggle_wsl.sh <1|2> --mode solution --slug soup-tNN --commit <sha> [--env K=V ...]
#   wsl -d Ubuntu -- bash tools/kaggle_wsl.sh <1|2> --attach --slug soup-tNN --mode solution   # re-join a running kernel
#
# 1 = account namedesign (token file kaggle1.token), 2 = account peagverr (kaggle2.token); both have the
# private dataset soup-secrets with the GitHub read token. The commit must be pushed to origin first.
# Modes: solution (~7 min, our run only), diag (solution run with SOUP_SOLUTION_DIAG=1: memory snapshot,
# chrome trace, import times), full (base + solution + the 10 checks, ~10 min), profile.
# Results: runs_kaggle/<stamp>-<mode>-<slug>/output/ (gitignored). The CLI needs Python >= 3.11, hence WSL.
#
# Kaggle runs at most two GPU notebooks per account at a time (a third push fails or the kernel dies at
# once). Several sessions on this laptop share the accounts, so this script takes one of MAX_SLOTS slot
# files in a directory shared by every worktree and waits while both are taken. A slot is stale when
# its holder is gone (its pid at this VM boot no longer exists: WSL restarted or the watcher was
# killed; the Kaggle kernel itself keeps running) or after 50 minutes.
acct=$1; shift
case "$acct" in
  1) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle1.token; user=namedesign ;;
  2) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle2.token; user=peagverr ;;
  3) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle3.token; user=slerfas ;;
  4) tok=/mnt/c/Users/abdra/OneDrive/Desktop/kaggle7.token; user=dinarakoshkaeva ;;
  *) echo "first argument: account 1 (namedesign), 2 (peagverr), 3 (slerfas) or 4 (dinarakoshkaeva)"; exit 2 ;;
esac
slug=soup-kernel-run
prev=""
for arg in "$@"; do
  if [ "$prev" = "--slug" ]; then slug=$arg; fi
  prev=$arg
done
MAX_SLOTS=2
LOCKDIR=/mnt/c/Users/abdra/Projects/soup/runs_kaggle/.kaggle_slots_$user
mkdir -p "$LOCKDIR"
BOOT_ID=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)
stale_locks() {
  find "$LOCKDIR" -name '*.lock' -mmin +50 -delete 2>/dev/null
  for f in "$LOCKDIR"/*.lock; do
    [ -e "$f" ] || continue
    read -r pid boot _ < "$f"
    if [ "$boot" != "$BOOT_ID" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; fi
  done
}
stale_locks
sleep $((RANDOM % 7))
while [ "$(ls "$LOCKDIR"/*.lock 2>/dev/null | wc -l)" -ge "$MAX_SLOTS" ]; do
  echo "[kaggle_wsl] both GPU slots of $user are taken ($(ls "$LOCKDIR" | tr '\n' ' ')); waiting 60 s"
  sleep 60
  stale_locks
done
echo "$$ $BOOT_ID $(date) $slug" > "$LOCKDIR/$slug.lock"
trap 'rm -f "$LOCKDIR/$slug.lock"' EXIT
export KAGGLE_API_TOKEN=$(tr -d '\r\n ' < "$tok")
export PATH=$HOME/miniforge3/envs/soup/bin:$PATH
python tools/kaggle_run.py --user "$user" --repo https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers.git --poll 30 "$@"
echo "KAGGLE_RUN_DONE $user $*"
