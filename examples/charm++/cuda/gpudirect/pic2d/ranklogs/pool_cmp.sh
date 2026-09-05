#!/bin/bash
# pool_cmp.sh <jobid> <reps> [tags...]   tags: <lb>_<mode>, lb in noLB|sync|async, mode in nopool|pool
J=$1; REPS=$2; shift 2
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; RL=$P/ranklogs; cd $P || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_ZC_STATS=1 CHARM_PIC2D_CHECKSUM=1
BASE="-W 1024 -H 1024 -w 128 -h 128 -p 4 -i 30 -u 3 -d 1 -v 2 -T 0.05 -c 10"
NOPOOL="+gpushm +gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 512 +gpuipceventpool 256"
POOL="+gpushm +gpupool +gpuipceventpool 256"
OUT=$RL/pool_cmp_results.txt; : > $OUT
for r in $(seq 1 $REPS); do for tag in "$@"; do
  lb=${tag%_*}; mode=${tag#*_}
  case $lb in noLB) LB="-f 999";; sync) LB="-b 5 +balancer DiffusionLB +LBDiffusionCommOn";; async) LB="-b 5 -a +balancer DiffusionLB +LBDiffusionCommOn +LBAsync";; esac
  [ $mode = pool ] && IPC="$POOL" || IPC="$NOPOOL"
  export RANKLOG_DIR=$RL/pc_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  timeout 120 srun --jobid=$J --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/pcwrap.sh $P/pic2d $BASE $LB $IPC >/dev/null 2>&1; rc=$?; sleep 1
  tot=$(grep -ho -P 'Total time: \K[0-9.]+' $RANKLOG_DIR/rank_*.log | head -1)
  phi=$(grep -ho -P 'phi2 \K[0-9.e+-]+' $RANKLOG_DIR/rank_*.log | tail -1)
  mig=$(grep -ho -P 'Objects migrating: \K[0-9]+' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)
  staged=$(grep -ah ipc-stats $RANKLOG_DIR/rank_*.log | grep -o 'staged=[0-9]*' | cut -d= -f2 | paste -sd,)
  ok=1; [ "$rc" = 0 ] && [ -n "$tot" ] && [ -n "$phi" ] || ok=0
  echo "$tag rep$r ok=$ok rc=$rc total=${tot:-NONE} phi2_final=${phi:-NONE} migrated=${mig:-0} staged=${staged:-?}" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag rep$r"; grep -ahnE 'Reason:|Fatal|Abort|outstanding for|WARNING|malloc\(\)|Segmentation' $RANKLOG_DIR/rank_*.log | head -6 | cut -c1-200; for f in $RANKLOG_DIR/rank_*.log; do echo "== $(basename $f): $(tail -1 $f | cut -c1-120)"; done; exit 1; fi
done; done
echo "ALL PASSED"
