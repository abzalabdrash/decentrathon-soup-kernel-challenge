#!/bin/bash
# Local helper: the test commands in the WSL conda environment, from the checkout root in Git Bash:
#   export MSYS_NO_PATHCONV=1
#   wsl -d Ubuntu -- bash tools/wsl_pytest.sh solution            # our tests (pytest solution)
#   wsl -d Ubuntu -- bash tools/wsl_pytest.sh soup                # Soup's 79 tests with apply() (check 10)
#   wsl -d Ubuntu -- bash tools/wsl_pytest.sh solution -k nf4     # extra pytest arguments pass through
PY=$HOME/miniforge3/envs/soup/bin/python
what=$1; shift
case "$what" in
  solution) timeout 1800 $PY -m pytest solution -q -p no:cacheprovider "$@" ;;
  soup) SOUP_APPLY_SOLUTION=1 timeout 900 $PY -m pytest tests -q -p no:cacheprovider "$@" ;;
  *) echo "usage: wsl_pytest.sh solution|soup [pytest args]"; exit 2 ;;
esac
