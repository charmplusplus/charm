#!/bin/bash
set -eu
ROOT=/u/bhosale/charm-reconverse
APP=$ROOT/examples/charm++/cuda/gpudirect/leanmd
OUT=$APP/measurement-test
JOB=$1
export LD_LIBRARY_PATH="$ROOT/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
ulimit -c 0
while :; do
    STATE=$(squeue -h -j "$JOB" -o '%T')
    [ "$STATE" = RUNNING ] && break
    [ "$STATE" = PENDING ] || { echo "Allocation unavailable: $STATE"; exit 1; }
    sleep 10
done
date
srun --jobid="$JOB" -N 1 -n 1 --cpus-per-task=8 --exact nvidia-smi topo -m > "$OUT/topology.log" 2>&1
cat "$OUT/topology.log"
TEST=$ROOT/tests/charm++/load_balancing/async_arrival/async_arrival
for PROCS in 1 4; do
    for SPLIT in 0 1; do
        EXTRA=()
        [ "$SPLIT" = 0 ] || EXTRA=(+LBAsync)
        LOG=$OUT/regression-${PROCS}proc-${SPLIT}split.log
        set +e
        CHARM_LB_NO_CUPTI=1 timeout 60 srun --jobid="$JOB" --mpi=cray_shasta -N 1 -n "$PROCS" \
            --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 \
            "$TEST" 4 "$SPLIT" +pe "$((PROCS*4))" +balancer GreedyCentralLB +gpupool "${EXTRA[@]}" > "$LOG" 2>&1
        RC=$?
        set -e
        echo "regression procs=$PROCS split=$SPLIT exit=$RC"
        tail -15 "$LOG"
        [ "$RC" = 0 ] || exit "$RC"
        grep -q '\[TEST\] PASS' "$LOG"
    done
done
bash "$OUT/run.sh" "$JOB" fixed-small "$APP/leanmd" async 4 40 8 8 6
bash "$OUT/run.sh" "$JOB" before-async "$OUT/leanmd.before" async
bash "$OUT/run.sh" "$JOB" fixed-async "$APP/leanmd" async
bash "$OUT/run.sh" "$JOB" fixed-sync "$APP/leanmd" sync
python3 "$OUT/summarize.py" "$OUT/before-async" "$OUT/fixed-async" "$OUT/fixed-sync"
echo 'CAMPAIGN COMPLETE'
