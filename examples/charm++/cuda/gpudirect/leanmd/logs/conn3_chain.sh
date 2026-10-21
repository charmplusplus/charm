#!/bin/bash
# conn3_chain.sh <jobid>: one A40 node, ~20 min. The no-LB channel gap (2.1-2.3%) and the zero-copy profile that hung.
# TARGET: flag writes per message on each path (16x8x8 no LB, 30 steps) and a zero-copy nsys report.
#  1. flag-count pair: channels then zero-copy with CHARM_ZC_STATS=1 ([hapi-stats] completion-flag writes per process,
#     [chan-stats]/[zc-stats]); binaries of 17:17 (counter added, protocol of 15:37).
#  2. zero-copy under nsys again, three ways, each with a stack of rank 0 taken 60 s after its last step if it has not
#     exited: (a) --duration=100 --kill=none; (b) no +nvtx; (c) +nvtx with the exit path skipped (CHARM_EXIT_QUICK=1 is
#     NOT a thing: (c) is simply the first form with --kill=sigkill). Reports: /scratch/mzu/bhosale/nsys/conn3_*/.
# KILL CRITERION: none that stops the chain; every arm is bounded by its timeout. Never sallocs, never scancels.
JID=$1; [ -n "$JID" ] || exit 1
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
LCI_LIB=/u/bhosale/lci/build-cuda/lib; export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
BASE="16 8 8 30 99999 99999 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 16384 +gpupoolalloc vmm"
echo "=== conn3 chain, job $JID, $(date +%T), nodes: $(squeue -j $JID -h -o %N), leanmd $(date -r $D/leanmd +%m-%d_%H:%M)"
echo "--- 1. flag writes per process, no LB, 30 steps"
for arm in ch zc; do
  TAG=c3_flags_${arm}_$(date +%H%M%S); RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL; LOG=$D/logs/$TAG.log
  C="$BASE"; [ $arm = ch ] && C="$C -channels"
  env CHARM_ZC_STATS=1 PES=8 GPP=1 RANKLOG_DIR=$RL timeout 300 $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd.sh $D/leanmd $C > $LOG 2>&1
  rc=$?; R=$RL/rank_0.log
  echo "$(date +%T) $TAG: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean11-30 $(mean_steps $R 11 30) ms"
  cat $RL/rank_*.log $LOG | grep -a 'hapi-stats\|chan-stats\|zc-stats' | sed -E 's/pid=[0-9]+ //' | cut -c1-200
done
echo "--- 2. zero-copy under nsys"
nsys_zc() {  # <name> <nsys opts> <extra app flags>
  local name=$1 opts="$2" extra="$3"
  local TAG=c3_nsys_${name}_$(date +%H%M%S) OUT=/scratch/mzu/bhosale/nsys/conn3_$name; rm -rf $OUT; mkdir -p $OUT
  local RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL; local LOG=$D/logs/$TAG.log
  echo "[nsys] $(date +%T) $TAG opts='$opts' app extra='$extra'" | tee $LOG
  env PES=8 GPP=1 RANKLOG_DIR=$RL LMD_NSYS_DIR=$OUT LMD_NSYS_OPTS="$opts" timeout 300 $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd_nsys.sh $D/leanmd $BASE $extra >> $LOG 2>&1 &
  local sp=$!
  # a stack of rank 0 once its last step is done and 60 s have passed without exit
  ( until grep -aq '^Step 30 ' $RL/rank_0.log 2>/dev/null; do sleep 5; done; sleep 60
    if kill -0 $sp 2>/dev/null; then
      srun --jobid=$JID --overlap -N 1 -n 1 --exact bash -c 'p=$(pgrep -f "leanmd 16 8 8" | head -1); echo "stack of rank-0 pid $p at $(date +%T)"; gstack $p 2>/dev/null || eu-stack -p $p 2>/dev/null || echo "no gstack/eu-stack"' > $D/logs/${TAG}_stack.txt 2>&1
      echo "$(date +%T) $TAG: still running 60 s after step 30; stack in logs/${TAG}_stack.txt ($(grep -c '^#' $D/logs/${TAG}_stack.txt) frames)"
      grep -m12 '^#' $D/logs/${TAG}_stack.txt | cut -c1-150
    fi ) &
  wait $sp; local rc=$?; wait
  echo "$(date +%T) $TAG: rc=$rc steps=$(grep -ac '^Step [0-9]* Benchmark Time' $RL/rank_0.log) | report: $(ls -la $OUT/rank_0.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
}
nsys_zc duration "--duration=100 --kill=none" "+nvtx"
nsys_zc nonvtx "" ""
nsys_zc sigkill "--duration=100 --kill=sigkill" "+nvtx"
echo "=== conn3 chain done $(date +%T), left $(squeue -j $JID -h -o %L)"
