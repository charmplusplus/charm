#!/bin/bash
# One jacobi2d-imbalance run inside an existing allocation.
# Usage: JOBID=<id> run_lb.sh <tag> <all app+charm args...>
set -u
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
BIN=$D/jacobi2d
WRAP=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d/numa_wrap.sh
LOG=$D/lbtest/$1.log; shift

export LD_LIBRARY_PATH=/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export FI_MR_CACHE_MONITOR=disabled      # CUDA/libfabric ABBA deadlock on migration
export MODE=match PES_PER_PROC=8         # NUMA pemap: local rank r -> NUMA 3-r
ulimit -c 0

srun --jobid="$JOBID" --unbuffered -N 1 -n 4 --ntasks-per-node=4 \
     --cpus-per-task=8 --cpu-bind=none --exact \
     "$WRAP" "$BIN" "$@" > "$LOG" 2>&1
rc=$?
echo "=== $(basename "$LOG") rc=$rc"
grep -E "Chares:|Total time|Average iteration time" "$LOG"
grep -E "outstanding for|Objects migrating|migrated|Migrat" "$LOG" | head -4
echo "--- last line: $(tail -1 "$LOG")"
if [ $rc -ne 0 ] || ! grep -q "Total time" "$LOG"; then
  echo "!!! FAILED OR NO RESULT -- log tail:"; tail -25 "$LOG"; exit 1
fi
exit 0
