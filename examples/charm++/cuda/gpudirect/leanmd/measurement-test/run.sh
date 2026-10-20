#!/bin/bash
set -eu
ROOT=/u/bhosale/charm-reconverse
APP=$ROOT/examples/charm++/cuda/gpudirect/leanmd
JOB=$1
TAG=$2
BIN=$3
MODE=$4
GRID=${5:-8}
STEPS=${6:-100}
FIRST=${7:-20}
PERIOD=${8:-20}
LAG=${9:-16}
export RANKLOG_DIR=$APP/measurement-test/$TAG
mkdir -p "$RANKLOG_DIR"
ulimit -c 0
export LD_LIBRARY_PATH="$ROOT/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
export CK_GPU_ARENA_MB=2048 CHARM_LB_RAWLOAD=1 CHARM_GPU_LOAD_AUDIT=1
ARGS=("$GRID" "$GRID" "$GRID" "$STEPS" "$FIRST" "$PERIOD"
      -computemap local -density gradient +pe 32 +setcpuaffinity
      +gpushm +gpuipceventpool 256 +gpupool
      +balancer MetisLB +balancer DiffusionLB
      +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 1)
if [ "$MODE" = async ]; then ARGS+=(-lbasync -lblag "$LAG" +LBAsync); fi
printf '%q ' "$BIN" "${ARGS[@]}" > "$RANKLOG_DIR/command.txt"
printf '\n' >> "$RANKLOG_DIR/command.txt"
set +e
timeout 240 srun --jobid="$JOB" --mpi=cray_shasta -N 1 -n 4 \
    --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact \
    --kill-on-bad-exit=1 stdbuf -oL -eL "$APP/ranklogs/rankwrap.sh" \
    "$BIN" "${ARGS[@]}" > "$RANKLOG_DIR/launcher.log" 2>&1
RC=$?
set -e
echo "$TAG exit=$RC"
tail -12 "$RANKLOG_DIR/launcher.log"
for LOG in "$RANKLOG_DIR"/rank_*.log; do
    echo "$LOG"
    tail -8 "$LOG"
done
exit "$RC"
