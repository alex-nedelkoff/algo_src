#!/bin/zsh
P=~/.venvs/aigp-rl/bin/python
POL=$1; LAGS=$2   # LAGS: "0" or "1"
cd ~/Documents/drone-ai-grand-prix/algo_src
tot=0; n=0; fin=0
for tn in -0.25 -0.2 -0.15 0.15 0.2 0.25; do
  for sp in 12 14 16; do
    if [ "$LAGS" = "1" ]; then
      g=$(PYTHONPATH=. $P scripts/sysid/closed_loop_policy.py --policy $POL --maxw 6 --thrmax 0.6 --space $sp --turn $tn --lag 0.085 --lat 0.019 2>/dev/null | tail -1 | sed 's/.*reached \([0-9]*\).*/\1/')
    else
      g=$(PYTHONPATH=. $P scripts/sysid/closed_loop_policy.py --policy $POL --maxw 6 --thrmax 0.6 --space $sp --turn $tn 2>/dev/null | tail -1 | sed 's/.*reached \([0-9]*\).*/\1/')
    fi
    tot=$((tot+g)); n=$((n+1)); [ "$g" = "6" ] && fin=$((fin+1))
    printf "%s " $g
  done
done
echo ""
echo "$POL lag=$LAGS: mean $(echo "scale=2; $tot/$n" | bc)/6  finished $fin/$n"
