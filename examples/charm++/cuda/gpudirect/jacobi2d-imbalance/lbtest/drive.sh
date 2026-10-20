#!/bin/bash
# Wait for allocation 21958361, run the LB battery, then ALWAYS release the node.
set -u
JOB=21958361
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
cd "$D" || exit 1
export JOBID=$JOB

# 1. Wait for the allocation.
while true; do
  st=$(squeue -j $JOB -h -o %T 2>/dev/null)
  if [ -z "$st" ]; then
    echo "JOB GONE before any run: $(sacct -j $JOB -n -o State 2>/dev/null | head -1)"; exit 1
  fi
  [ "$st" = "RUNNING" ] && break
  sleep 15
done
# Node is ours: from here on, release it however this script ends.
trap 'echo "=== releasing allocation $JOB at $(date +%T)"; scancel $JOB; sleep 2; squeue -j $JOB 2>&1 | tail -2' EXIT
echo "=== GRANTED node $(squeue -j $JOB -h -o %N) at $(date +%T)"

# 2. Battery. 256 chares on 32 PEs = 8 objects/PE; -m 5 = diagonal load gradient.
APP="-W 8192 -H 8192 -w 512 -h 512 -u 10 -m 5"
POOL="+gpushm +gpupool +gpuipceventpool 256"

echo "### smoke: 20 iters, no LB"
./lbtest/run_lb.sh smoke $APP -i 20 -f 100 -l 100 +ppn 8 $POOL || exit 1

for cfg in "nolb:" "greedy:+balancer GreedyRefineCentralGPULB" "metis:+balancer MetisLB"; do
  tag=${cfg%%:*}; bal=${cfg#*:}
  echo "### $tag  ${bal:-(no balancer)}"
  ./lbtest/run_lb.sh "$tag" $APP -i 100 -f 20 -l 20 +ppn 8 $POOL $bal || {
    echo "### STOPPING: $tag failed"; exit 1; }
done

echo "=== BATTERY COMPLETE at $(date +%T)"
