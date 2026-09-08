#!/bin/bash
# Controlled test: is lag 16 bad because of the LAG, or because of the
# MEASUREMENT WINDOW it leaves? Instrumentation is off from the step's start
# until AtSyncWait returns (cklocation.C:2578), so the window is period - lag.
#   period 20, lag 16 -> 4 instrumented steps
#   period 40, lag 16 -> 24 instrumented steps   (same lag, 6x the window)
# If lag16 converges at period 40 like sync does, the window is the cause.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs
cd $L || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
J=$1; REPS=${2:-3}
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
# 140 steps, first LB at 20, period 40 -> LB at 20, 60, 100: three decisions.
C="8 8 8 140 20 40 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool"
run(){ tag=$1; shift
  for r in $(seq 1 $REPS); do
    export RANKLOG_DIR=$RL/wt_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
    env X=1 timeout 240 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 \
        --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd $C "$@" > /dev/null 2>&1
    rc=$?; cp -f $RANKLOG_DIR/rank_0.log $L/wt_${tag}_$r.log 2>/dev/null
    echo "$tag rep$r rc=$rc steps=$(grep -ac 'Benchmark Time' $L/wt_${tag}_$r.log 2>/dev/null)"
  done; }
run p40sync  $D
run p40lag16 -lbasync -lblag 16 +LBAsync $D
echo "WINDOWTEST DONE"
