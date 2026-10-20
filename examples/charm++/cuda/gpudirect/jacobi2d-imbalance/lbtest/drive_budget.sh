#!/bin/bash
# Does clamping the move budget cut migrations without losing balance?
set -u
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
cd "$D" || exit 1
export JOBID=$(cat lbtest/jobid)
APP="-W 4096 -H 4096 -w 256 -h 256 -i 30 -u 5 -f 10 -l 10 -m 5"
BASE="+ppn 8 +gpushm +gpuipceventpool 256 +LBDebug 1 +LBPrintSummary"
for pct in 100 50 20 10; do
  echo "### +LBPercentMovesAllowed $pct"
  ./lbtest/run_load.sh budget_$pct $APP $BASE \
      +balancer GreedyRefineCentralGPULB +LBPercentMovesAllowed $pct \
    | grep -vE "Load Summary \(before" || { echo "### FAILED at pct=$pct"; break; }
done
echo "=== SWEEP DONE at $(date +%T). ALLOCATION LEFT UP."
squeue -j $JOBID -h -o "    job %i on %N, time left %L"
