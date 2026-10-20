#!/bin/bash
set -u
ROOT=/u/bhosale/charm-reconverse
OUT=$ROOT/examples/charm++/cuda/gpudirect/leanmd/measurement-test
JOB=$1; TAG=$2; BIN=$3; PROCS=$4; SPLIT=$5
export LD_LIBRARY_PATH="$ROOT/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024 CHARM_LB_NO_CUPTI=1
ulimit -c 0
EXTRA=()
[ "$SPLIT" = 0 ] || EXTRA=(+LBAsync)
timeout 35 srun --jobid="$JOB" --mpi=cray_shasta -N 1 -n "$PROCS" \
    --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 \
    stdbuf -oL -eL "$BIN" 4 "$SPLIT" +pe "$((PROCS*4))" \
    +balancer GreedyCentralLB +gpupool "${EXTRA[@]}" > "$OUT/$TAG.log" 2>&1
RC=$?
echo "$TAG exit=$RC"
tail -20 "$OUT/$TAG.log"
exit "$RC"
