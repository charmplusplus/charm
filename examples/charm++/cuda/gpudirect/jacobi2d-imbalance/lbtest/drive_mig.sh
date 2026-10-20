#!/bin/bash
# Did the balancer migrate anything? Small, fast, +LBDebug 1 to print the count.
set -u
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
cd "$D" || exit 1
JOB=$(cat lbtest/jobid)
export JOBID=$JOB
while true; do
  st=$(squeue -j $JOB -h -o %T 2>/dev/null)
  [ -z "$st" ] && { echo "JOB GONE before any run: $(sacct -j $JOB -n -o State 2>/dev/null|head -1)"; exit 1; }
  [ "$st" = "RUNNING" ] && break
  sleep 15
done
trap 'echo "=== releasing $JOB at $(date +%T)"; scancel $JOB; sleep 2; sacct -j $JOB -n -o JobID,State 2>/dev/null|head -2' EXIT
echo "=== GRANTED $(squeue -j $JOB -h -o %N) at $(date +%T)"

# 256 chares on 32 PEs = 8 obj/PE; -m 5 diagonal gradient; LB at iters 10 and 20.
APP="-W 4096 -H 4096 -w 256 -h 256 -i 30 -u 5 -f 10 -l 10 -m 5"
FLAGS="+ppn 8 +gpushm +gpuipceventpool 256 +LBDebug 1"

echo "### control: no balancer (expect zero migration lines)"
./lbtest/run_mig.sh mig_nolb $APP $FLAGS || exit 1
echo "### GreedyRefineCentralGPULB"
./lbtest/run_mig.sh mig_greedy $APP $FLAGS +balancer GreedyRefineCentralGPULB || exit 1
echo "### MetisLB"
./lbtest/run_mig.sh mig_metis $APP $FLAGS +balancer MetisLB || exit 1
echo "=== DONE at $(date +%T)"
