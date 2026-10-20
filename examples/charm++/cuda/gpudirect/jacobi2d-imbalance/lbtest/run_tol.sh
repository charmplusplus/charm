#!/bin/bash
# One run, reporting what the tolerance bought.
set -u
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance
WRAP=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d/numa_wrap.sh
LOG=$D/lbtest/$1.log; shift
export LD_LIBRARY_PATH=/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID FI_MR_CACHE_MONITOR=disabled
export MODE=match PES_PER_PROC=8
ulimit -c 0
srun --jobid="$JOBID" --unbuffered -N 1 -n 4 --ntasks-per-node=4 \
     --cpus-per-task=8 --cpu-bind=none --exact \
     "$WRAP" "$D/jacobi2d" "$@" > "$LOG" 2>&1
rc=$?
echo "=== $(basename "$LOG") rc=$rc"
echo "--- migrations:"; grep -o "#Objects migrating: [0-9]*" "$LOG" | sed 's/^/    /'
echo "    total moved: $(grep -o '#Objects migrating: [0-9]*' "$LOG" | awk -F': ' '{s+=$2} END{print s+0}')"
echo "--- new diagnostic (single-candidate mode only):"; grep "ratioToGreedy" "$LOG" | sed 's/^/    /'
echo "--- final balance:"; grep "Load Summary (after" "$LOG" | tail -1 | sed 's/^/    /'
echo "--- timing:"; grep -E "Total time|Average iteration" "$LOG" | sed 's/^/    /'
grep -E "outstanding for" "$LOG" | head -2
[ $rc -ne 0 ] && { echo "!!! FAILED:"; tail -20 "$LOG"; exit 1; }
grep -q "Total time" "$LOG" || { echo "!!! NO RESULT:"; tail -20 "$LOG"; exit 1; }
exit 0
