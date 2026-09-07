#!/bin/bash
# lbcmp.sh <jobid> <reps> [tags...]   tags: <lb>_<mode>, lb in noLB|sync|async, mode in pool|nopool
# Runs the tags in order, REPS times each, and checks every run's final
# checksums (out2, w2) against the first run's: the routing is a pure function
# of the arguments, so every placement must produce the same bits. Stops at
# the first failure and dumps that run's log tail.
#   BASE=  the application arguments (default: 64 experts, 2048x8192, 8192 tokens/PE, zipf 1)
#   BAL=   balancer (DiffusionLB); BALFLAGS= its flags, set empty for a non-diffusion one
#   LBF=   extra balancer flags (e.g. "+LBDiffusionMaxMoveFrac 1")
#   NODES= node count (1); TMO= per-run timeout in seconds (300)
J=$1; REPS=$2; shift 2
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/moe; RL=$P/ranklogs; cd $P || exit 1; ulimit -c 0
MATHLIBS=$(sed -n 's/^CUDA_DIR="\(.*\)"/\1/p' /u/bhosale/charm-reconverse/include/conv-mach-opt.sh | sed 's#/cuda/#/math_libs/#')
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$MATHLIBS/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 PMI_MAX_KVS_ENTRIES=1024
NODES=${NODES:-1}; NT=$((NODES * 4))
BASE=${BASE:-"-e 64 -m 2048 -h 8192 -t 8192 -i 30 -u 3 -z 1.0 -p 10 -C 5"}
NOPOOL="+gpushm +gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 1024 +gpuipceventpool 256"
POOL="+gpushm +gpupool +gpupoolsize 1024 +gpuipceventpool 256"
LBFLAGS="+balancer ${BAL:-DiffusionLB} ${BALFLAGS-+LBDiffusionCommOn +LBDiffusionGpuDim} ${LBF:-}"
REF_OUT=""; REF_W=""
OUT=$RL/lbcmp_results.txt; : > $OUT
for r in $(seq 1 $REPS); do for tag in "$@"; do
  lb=${tag%_*}; mode=${tag#*_}
  case $lb in noLB) LB="-f 999";; sync) LB="-f 5 -b 5 $LBFLAGS";; async) LB="-f 5 -b 5 -a -l 3 $LBFLAGS +LBAsync";; *) echo "bad tag $tag"; exit 2;; esac
  [ $mode = pool ] && IPC="$POOL" || IPC="$NOPOOL"
  export RANKLOG_DIR=$RL/lbc_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  timeout ${TMO:-300} srun --jobid=$J --mpi=cray_shasta -N $NODES -n $NT --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --unbuffered $RL/moewrap.sh $P/moe $BASE $LB $IPC >/dev/null 2>&1; rc=$?; sleep 1
  tot=$(grep -ho -P 'Total time: \K[0-9.]+' $RANKLOG_DIR/rank_*.log | head -1)
  avg=$(grep -ho -P 'Average step time: \K[0-9.]+' $RANKLOG_DIR/rank_*.log | head -1)
  out2=$(grep -ho -P 'out2 \K[0-9.e+-]+' $RANKLOG_DIR/rank_0.log | tail -1)
  w2=$(grep -ho -P 'w2 \K[0-9.e+-]+' $RANKLOG_DIR/rank_0.log | tail -1)
  [ -z "$REF_OUT" ] && [ -n "$out2" ] && { REF_OUT=$out2; REF_W=$w2; }
  exact=0; [ -n "$out2" ] && [ "$out2" = "$REF_OUT" ] && [ "$w2" = "$REF_W" ] && exact=1
  emig=$(grep -aho -P 'emigrate n=\K[0-9]+' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)
  ok=1; [ "$rc" = 0 ] && [ -n "$tot" ] && [ "$exact" = 1 ] || ok=0
  echo "$tag rep$r ok=$ok rc=$rc total=${tot:-NONE} avg_ms=${avg:-NONE} out2=${out2:-NONE} w2=${w2:-NONE} exact=$exact emigrated=${emig:-0}" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag rep$r"; grep -ahnE 'Reason:|Fatal|Abort|outstanding for|WARNING|Segmentation|conserved|exceed|cuBLAS' $RANKLOG_DIR/rank_*.log | head -8 | cut -c1-220; for f in $RANKLOG_DIR/rank_*.log; do echo "== $(basename $f): $(tail -1 $f | cut -c1-140)"; done; exit 1; fi
done; done
echo "ALL PASSED"
