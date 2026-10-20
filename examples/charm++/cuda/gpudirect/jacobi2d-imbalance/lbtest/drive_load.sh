#!/bin/bash
# What loads did the balancer see? Complete batch. DOES NOT CANCEL the allocation.
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
echo "=== GRANTED $(squeue -j $JOB -h -o %N) at $(date +%T)"

APP="-W 4096 -H 4096 -w 256 -h 256 -i 30 -u 5 -f 10 -l 10 -m 5"
BASE="+ppn 8 +gpushm +gpuipceventpool 256 +LBDebug 1 +LBPrintSummary"

echo "### 1/3 GreedyRefineCentralGPULB, load summary"
./lbtest/run_load.sh load_greedy $APP $BASE +balancer GreedyRefineCentralGPULB || exit 1
echo "### 2/3 MetisLB, load summary"
./lbtest/run_load.sh load_metis  $APP $BASE +balancer MetisLB || exit 1
echo "### 3/3 GreedyRefineCentralGPULB + per-object stats dump"
rm -f lbtest/lbstats*
./lbtest/run_load.sh load_dump $APP $BASE +balancer GreedyRefineCentralGPULB \
    +LBDump 1 +LBDumpSteps 1 +LBDumpFile lbtest/lbstats || exit 1
echo "--- dump files:"; ls -la lbtest/lbstats* 2>/dev/null || echo "    (none written)"

echo "=== BATCH DONE at $(date +%T). ALLOCATION $JOB LEFT UP."
squeue -j $JOB -h -o "    job %i on %N, time left %L"
