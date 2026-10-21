#!/bin/bash
# conn4_chain.sh <jobid>: one A40 node, ~35 min. The batched channel put (one batched copy + one flag per producing
# stream per entry method), against the per-put path of 15:37/19:11 and zero-copy.
# TARGET: 16x8x8 no LB: channels 2584-2588 (one/two flags) vs zero-copy 2502 ms/step on gpub002; flags per step
# per process 17.9k (one-flag) -> ~7.7k expected (3.6k batches + 4.1k of the application's); zero-copy 11.1k.
# KILL CRITERION: a chantest arm fails, or a channel run hangs (rc=124) => stop and say so. Never sallocs/scancels.
#  1. chantest, all 8 arms.
#  2. 16x8x8 no LB, 30 steps, counters: channels (batched), zero-copy.
#  3. 16x8x8 async x1 and 8x4x4 async x1, channels (ties with zero-copy must hold).
#  4. nsys, rank 3 (the busiest), 10 steps, channels batched: copy-engine time against 918 (per put) / 713 (one flag)
#     / 512 (zero-copy) ms per 10 steps.
JID=$1; [ -n "$JID" ] || exit 1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
CT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
S=/tmp/claude-54324/-u-bhosale-charm-reconverse/52f63e60-0155-4055-9277-35b69c94a33f/scratchpad
until grep -q '=== build done' $S/build_batch.log 2>/dev/null; do sleep 10; done
echo "=== conn4 chain, job $JID, $(date +%T), nodes: $(squeue -j $JID -h -o %N), leanmd $(date -r $L/leanmd +%m-%d_%H:%M) batch=$(nm -C $L/leanmd | grep -c deviceChannelBatchFlush)"
[ "$(nm -C $L/leanmd | grep -c deviceChannelBatchFlush)" = 0 ] && { echo "leanmd lacks the batch code; stopped"; exit 1; }
echo "--- 1. chantest"
bash $CT/test_chantest_stable.sh $JID 24 3 2>&1 | grep -E 'PASS|FAIL|done' | cut -c1-110 | tee $L/logs/conn4_chantest.out
grep -q 'FAIL' $L/logs/conn4_chantest.out && { echo "KILL CRITERION: a chantest arm failed. Chain stopped."; exit 1; }
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd() {  # <tag> <channels> <grid> <steps> <lb>
  local tag=${1}_$(date +%H%M%S) ch=$2 grid="$3" steps=$4 lb=$5
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="$grid" STEPS=$steps LB=$lb CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=480 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | pre-LB 1-20 $(mean_steps $R 1 20) | 11-30 $(mean_steps $R 11 30) | mean42-100 $(mean_steps $R 42 100) ms | stalls=$(cat $L/ranklogs/$tag/rank_*.log | grep -ac CHAN-STALL)"
  cat $L/ranklogs/$tag/rank_*.log $L/logs/$tag.log 2>/dev/null | grep -a 'hapi-stats\|chan-stats\|zc-stats' | sed -E 's/pid=[0-9]+ //' | sort | uniq -c | sort -rn | head -5 | cut -c1-190
  if [ $rc -ne 0 ] && [ $ch = 1 ]; then echo "KILL CRITERION: channel run rc=$rc. Chain stopped."; exit 1; fi
}
echo "--- 2. 16x8x8 no LB, 30 steps"; lmd c4_nolb_ch 1 "16 8 8" 30 none; lmd c4_nolb_zc 0 "16 8 8" 30 none
echo "--- 3. async"; lmd c4_big_async_ch 1 "16 8 8" 100 async; lmd c4_async_ch 1 "8 4 4" 100 async
echo "--- 4. nsys rank 3, channels batched, 10 steps"
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
LCI_LIB=/u/bhosale/lci/build-cuda/lib; export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
TAG=c4_nsys_r3_batch_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/nsys/conn4_r3_batch; rm -rf $OUT; mkdir -p $OUT
RL=$L/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
env PES=8 GPP=1 RANKLOG_DIR=$RL LMD_NSYS_DIR=$OUT LMD_NSYS_RANK=3 timeout 600 $SRUN --chdir=$L stdbuf -oL -eL $L/numa_wrap_lmd_nsys.sh $L/leanmd 16 8 8 10 99999 99999 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 16384 +gpupoolalloc vmm -channels > $L/logs/$TAG.log 2>&1
echo "$(date +%T) $TAG: rc=$? steps=$(grep -ac '^Step [0-9]* Benchmark Time' $RL/rank_0.log) | report: $(ls -la $OUT/rank_3.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
echo "=== conn4 chain done $(date +%T), left $(squeue -j $JID -h -o %L)"
