#!/bin/bash
# run_timeline.sh <jobid> [reps]
# noLB / sync / async on the imbalanced configuration (-computemap local
# -density gradient), 100 steps with LB at 20/40/60/80 -- long stretches
# between balancing steps so the timeline is readable.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs
cd $L || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
J=$1; REPS=${2:-2}; CELLS=${CELLS:-6}
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
C="$CELLS $CELLS $CELLS 100 20 20 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool"
for r in $(seq 1 $REPS); do for arm in noLB sync async; do
  case $arm in noLB) LB="";; sync) LB="$D";; async) LB="-lbasync -lblag ${LAG:-2} +LBAsync $D";; esac
  export RANKLOG_DIR=$RL/tl${CELLS}_${arm}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  env X=1 timeout 180 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 \
      --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd $C $LB > /dev/null 2>&1
  rc=$?
  cp -f $RANKLOG_DIR/rank_0.log $L/tl${CELLS}_${arm}_$r.log 2>/dev/null
  echo "$arm rep$r rc=$rc steps=$(grep -ac 'Benchmark Time' $L/tl${CELLS}_${arm}_$r.log 2>/dev/null) energy=$(grep -ho -P 'final \K[0-9.E+-]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)"
done; done
echo "TIMELINE DONE"
