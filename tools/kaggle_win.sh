#!/bin/bash
# Local helper (Yerassyl's Windows laptop, from 31 Oct): tools/kaggle_run.py from Git Bash with the
# Windows Python 3.12 (kaggle 2.2.5 is installed there) under one of the two team accounts. Same
# arguments as tools/kaggle_wsl.sh (Abzal's laptop); run it from the root of the checkout or worktree
# you want measured:
#
#   bash tools/kaggle_win.sh <1|2> --mode variants --slug soup-xx --commit <sha> --variant a: --variant b:K=1 --repeats 2 --abba --warm-run --skip-pytest
#   bash tools/kaggle_win.sh <1|2> --mode full --slug soup-xx --commit <sha>
#   bash tools/kaggle_win.sh <1|2> --attach --slug soup-xx --mode variants      # re-join a running kernel
#
# 1 = account namedesign (kaggle1.token), 2 = account peagverr (kaggle2.token); the token files live in
# "C:\Users\User\Downloads\Telegram Desktop" (never committed). Both accounts have the private dataset
# soup-secrets with the GitHub read token. The commit must be pushed to origin first.
#
# Kaggle runs at most two GPU notebooks per account at a time. Several sessions (worktrees) on this
# laptop share the accounts, so this script takes one of MAX_SLOTS slot files in a directory shared by
# every worktree and waits while both are taken; a slot is stale when its holder's pid is gone or
# after 50 minutes (the Kaggle kernel itself keeps running). --attach takes no slot.
acct=$1; shift
TOKDIR="/c/Users/User/Downloads/Telegram Desktop"
case "$acct" in
  1) tok="$TOKDIR/kaggle1.token"; user=namedesign ;;
  2) tok="$TOKDIR/kaggle2.token"; user=peagverr ;;
  *) echo "first argument: account 1 (namedesign) or 2 (peagverr)"; exit 2 ;;
esac
[ -r "$tok" ] || { echo "token file $tok not found"; exit 2; }
PYDIR=/c/Users/User/AppData/Local/Programs/Python/Python312
export PATH="$PYDIR/Scripts:$PYDIR:$PATH"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export KAGGLE_API_TOKEN=$(tr -d '\r\n ' < "$tok")
slug=soup-kernel-run; attach=0; prev=""
for arg in "$@"; do
  [ "$prev" = "--slug" ] && slug=$arg
  [ "$arg" = "--attach" ] && attach=1
  prev=$arg
done
run() {
  "$PYDIR/python.exe" tools/kaggle_run.py --user "$user" --repo https://github.com/BAITC-Hacks/hack-78c6e9c3-null-pointers.git --poll 30 "$@"
  echo "KAGGLE_RUN_DONE $user $*"
}
if [ "$attach" = 1 ]; then run "$@"; exit; fi
MAX_SLOTS=2
LOCKDIR="/c/Users/User/Desktop/децентратон/soup/runs_kaggle/.kaggle_slots_$user"
mkdir -p "$LOCKDIR"
stale_locks() {
  find "$LOCKDIR" -name '*.lock' -mmin +50 -delete 2>/dev/null
  for f in "$LOCKDIR"/*.lock; do
    [ -e "$f" ] || continue
    read -r pid _ < "$f"
    kill -0 "$pid" 2>/dev/null || rm -f "$f"
  done
}
stale_locks
sleep $((RANDOM % 7))
while [ "$(ls "$LOCKDIR"/*.lock 2>/dev/null | wc -l)" -ge "$MAX_SLOTS" ]; do
  echo "[kaggle_win] both GPU slots of $user are taken ($(ls "$LOCKDIR" | tr '\n' ' ')); waiting 60 s"
  sleep 60
  stale_locks
done
echo "$$ $(date) $slug" > "$LOCKDIR/$slug.lock"
trap 'rm -f "$LOCKDIR/$slug.lock"' EXIT
run "$@"
