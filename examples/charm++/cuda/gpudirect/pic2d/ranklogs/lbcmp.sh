#!/bin/bash
# lbcmp.sh <jobid> <reps> [tags...]   tags: <lb>_<mode>, lb in noLB|sync|async, mode in nopool|pool
# Matrix config (real imbalance): -d 1, LB every 5 from 5, checksum every 5.
J=$1; REPS=$2; shift 2
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; RL=$P/ranklogs; cd $P || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_ZC_STATS=1 CHARM_PIC2D_CHECKSUM=1 CHARM_LB_MIGSTATS=1 PMI_MAX_KVS_ENTRIES=1024
NODES=${NODES:-1}; NT=$((NODES * 4))
BASE="-W 1024 -H 1024 -w 128 -h 128 -p 4 -i 30 -u 3 -d 1 -c 5"
NOPOOL="+gpushm +gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 512 +gpuipceventpool 256"
POOL="+gpushm +gpupool +gpuipceventpool 256"
REF=4.778313883736840e+04
OUT=$RL/lbcmp_results.txt; : > $OUT
for r in $(seq 1 $REPS); do for tag in "$@"; do
  lb=${tag%_*}; mode=${tag#*_}
  case $lb in noLB) LB="-f 999";; sync) LB="-f 5 -b 5 +balancer DiffusionLB +LBDiffusionCommOn";; async) LB="-f 5 -b 5 -a +balancer DiffusionLB +LBDiffusionCommOn +LBAsync ${EXTRA:-}";; esac
  [ $mode = pool ] && IPC="$POOL" || IPC="$NOPOOL"
  export RANKLOG_DIR=$RL/lbc_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  timeout ${TMO:-120} srun --jobid=$J --mpi=cray_shasta -N $NODES -n $NT --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/pcwrap.sh $P/pic2d $BASE $LB $IPC >/dev/null 2>&1; rc=$?; sleep 1
  tot=$(grep -ho -P 'Total time: \K[0-9.]+' $RANKLOG_DIR/rank_*.log | head -1)
  phi=$(grep -ho -P 'phi2 \K[0-9.e+-]+' $RANKLOG_DIR/rank_0.log | tail -1)
  exact=$(python3 -c "import sys; a=float('${phi:-0}'); b=$REF; print(1 if abs(a-b)/b < 2e-7 else 0)")
  emig=$(grep -aho -P 'emigrate n=\K[0-9]+' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)
  pseudo=$(grep -aho -P 'Pseudo LB: \K[0-9.]+' $RANKLOG_DIR/rank_0.log); across=$(grep -aho -P 'Across Node: \K[0-9.]+' $RANKLOG_DIR/rank_0.log)
  ok=1; [ "$rc" = 0 ] && [ -n "$tot" ] && [ "$exact" = 1 ] || ok=0
  echo "$tag rep$r ok=$ok rc=$rc total=${tot:-NONE} phi2=${phi:-NONE} exact=$exact emigrated=${emig:-0} pseudo=${pseudo:-?} across=${across:-?}" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag rep$r"; grep -ahnE 'Reason:|Fatal|Abort|outstanding for|WARNING|malloc\(\)|Segmentation|conserved' $RANKLOG_DIR/rank_*.log | head -6 | cut -c1-200; for f in $RANKLOG_DIR/rank_*.log; do echo "== $(basename $f): $(tail -1 $f | cut -c1-120)"; done; exit 1; fi
done; done
echo "ALL PASSED"
