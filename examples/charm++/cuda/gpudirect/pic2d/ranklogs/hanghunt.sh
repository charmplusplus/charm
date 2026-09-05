#!/bin/bash
# Launch plain 2-node pic2d async runs; on a stall (>35 s) attach gdb to every pic2d process on both nodes, dump stacks, kill.
J=$1; N=$2; P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; RL=$P/ranklogs; cd $P || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_PIC2D_CHECKSUM=1 PMI_MAX_KVS_ENTRIES=1024
unset CHARM_DEBUG_MIGRATE CHARM_DEVICE_MR_CACHE
export CHARM_ZC_STALL_SECS=10 CHARM_ZC_RESTAGE_DEBUG=1
NODES=$(squeue -j $J --noheader -o "%N"); 
for r in $(seq 1 $N); do
  export RANKLOG_DIR=$RL/hunt_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  srun --jobid=$J --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/pcwrap.sh $P/pic2d -W 1024 -H 1024 -w 128 -h 128 -p 4 -i 30 -u 3 -d 1 -c 5 -f 5 -b 5 -a +balancer DiffusionLB +LBDiffusionCommOn +LBAsync +gpushm +gpupool +gpuipceventpool 256 >/dev/null 2>&1 &
  pid=$!; t=0
  while kill -0 $pid 2>/dev/null && [ $t -lt 35 ]; do sleep 1; t=$((t+1)); done
  if kill -0 $pid 2>/dev/null; then
    echo "run $r: STALLED after ${t}s; last: $(tail -1 $RANKLOG_DIR/rank_0.log | cut -c1-80)"
    for h in $(scontrol show hostnames $NODES); do
      srun --jobid=$J --overlap -N1 -n1 -w $h bash -c 'for p in $(pgrep -x pic2d); do echo "=== host $(hostname) pid $p"; gdb -p $p -batch -ex "thread apply all bt 10" 2>&1 | grep -E "^Thread|^#" ; done' > $RANKLOG_DIR/stacks_$h.txt 2>&1
    done
    echo "stacks: $(cat $RANKLOG_DIR/stacks_*.txt | wc -l) lines"; echo "--- stall/restage/forward lines:"; grep -ah "STALL\|ZC RESTAGE\|FORWARD" $RANKLOG_DIR/rank_*.log | head -12 | cut -c1-220
    pkill -P $pid; kill $pid 2>/dev/null; scancel --jobid=$J --signal=KILL --steps 2>/dev/null; sleep 3
    exit 2
  fi
  wait $pid; rc=$?; tot=$(grep -ho -P 'Total time: \K[0-9.]+' $RANKLOG_DIR/rank_0.log); echo "run $r: rc=$rc total=$tot restages=$(grep -ah "ZC RESTAGE" $RANKLOG_DIR/rank_*.log | wc -l) repairs_rdma=$(grep -ah "FORWARD-REPAIR.*rdma" $RANKLOG_DIR/rank_*.log | wc -l) repairs_direct=$(grep -ah "FORWARD-REPAIR.*direct" $RANKLOG_DIR/rank_*.log | wc -l) skips=$(grep -ah "FORWARD-SKIP" $RANKLOG_DIR/rank_*.log | wc -l)"
done
echo "no stall in $N runs"
