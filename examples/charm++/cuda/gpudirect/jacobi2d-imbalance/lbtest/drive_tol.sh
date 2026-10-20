#!/bin/bash
# Does +LBGreedyRefineTolerance buy fewer migrations? DOES NOT CANCEL.
set -u
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
cd "$D" || exit 1
JOB=$(cat lbtest/jobid); export JOBID=$JOB
while true; do
  st=$(squeue -j $JOB -h -o %T 2>/dev/null)
  [ -z "$st" ] && { echo "JOB GONE: $(sacct -j $JOB -n -o State 2>/dev/null|head -1)"; exit 1; }
  [ "$st" = "RUNNING" ] && break
  sleep 15
done
trap 'echo "=== releasing $JOB at $(date +%T)"; scancel $JOB; sleep 2; sacct -j $JOB -n -o JobID,State 2>/dev/null|head -2' EXIT
echo "=== GRANTED $(squeue -j $JOB -h -o %N) at $(date +%T)"
APP="-W 4096 -H 4096 -w 256 -h 256 -i 30 -u 5 -f 10 -l 10 -m 5"
BASE="+ppn 8 +gpushm +gpuipceventpool 256 +LBDebug 1 +LBPrintSummary +balancer GreedyRefineCentralGPULB"
echo "### baseline: no tolerance (searches, = old behaviour)"
./lbtest/run_tol.sh tol_search $APP $BASE || exit 1
for t in 1.05 1.1 1.2 1.5; do
  echo "### +LBGreedyRefineTolerance $t"
  ./lbtest/run_tol.sh tol_$t $APP $BASE +LBGreedyRefineTolerance $t || { echo "### FAILED at $t"; break; }
done
echo "=== SWEEP DONE at $(date +%T)"
