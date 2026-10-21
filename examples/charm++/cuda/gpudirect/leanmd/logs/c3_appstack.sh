#!/bin/bash
# c3_appstack.sh <jobid> <tag-prefix>: once that run's rank 0 has printed step 30, wait 40 s and take the stacks of
# every leanmd process on the node (pgrep -x leanmd: the application, not nsys, whose command line also carries the
# application's arguments).
JID=$1; PFX=$2; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
RL=""; until [ -n "$RL" ] && grep -aq '^Step 30 ' $RL/rank_0.log 2>/dev/null; do RL=$(ls -td $L/ranklogs/${PFX}_* 2>/dev/null | head -1); sleep 5; done
echo "step 30 seen in $RL at $(date +%T); stacks in 40 s"; sleep 40
srun --jobid=$JID --overlap -N 1 -n 1 --exact bash -c 'n=0; for p in $(pgrep -x leanmd); do n=$((n+1)); echo "== leanmd pid $p at $(date +%T)"; gstack $p; done; [ $n = 0 ] && { echo "no leanmd process at $(date +%T); command lines with leanmd:"; pgrep -af leanmd | cut -c1-120; }'
