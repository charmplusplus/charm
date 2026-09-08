#!/bin/bash
# run_lagsweep.sh <jobid> <reps>
# -lblag is the distance between AtSyncStart and AtSyncWait, in steps. With
# -ldbPeriod 20 the natural setting is a fraction of that period, not a fixed
# small number: lag 16 is 80% of the window, so the strategy and the migrations
# overlap almost the whole interval between balancing steps.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs
cd $L || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
J=$1; REPS=${2:-3}; CELLS=8
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
C="$CELLS $CELLS $CELLS 100 20 20 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool"
run(){ tag=$1; shift
  for r in $(seq 1 $REPS); do
    export RANKLOG_DIR=$RL/ls_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
    env X=1 timeout 180 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 \
        --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd $C "$@" > /dev/null 2>&1
    rc=$?; cp -f $RANKLOG_DIR/rank_0.log $L/ls_${tag}_$r.log 2>/dev/null
    echo "$tag rep$r rc=$rc steps=$(grep -ac 'Benchmark Time' $L/ls_${tag}_$r.log 2>/dev/null) energy=$(grep -ho -P 'final \K[0-9.E+-]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)"
  done; }
run sync   $D
run lag2   -lbasync -lblag 2  +LBAsync $D
run lag16  -lbasync -lblag 16 +LBAsync $D
echo "LAGSWEEP DONE"
