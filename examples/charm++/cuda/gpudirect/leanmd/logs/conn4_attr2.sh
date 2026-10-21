#!/bin/bash
# conn4_attr.sh <jobid>: the batch flushed at CkCallstackPop (copies attributed to the element). Waits for the build.
# 8x4x4 async: ch, zc (the pairs before: ch 229.8/232.1 vs zc 198.4/209.0); then 16x8x8 async ch (1618.0 vs zc 1663.3).
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
S=/tmp/claude-54324/-u-bhosale-charm-reconverse/52f63e60-0155-4055-9277-35b69c94a33f/scratchpad
until grep -q '=== build done' $S/build_batch4.log 2>/dev/null && [ "$(nm -C $L/leanmd | grep -c CkRdmaDeviceChannelBatchFlush)" != 0 ] && ! ps -eo args | grep -q "[s]run --jobid=$JID"; do sleep 5; done
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
echo "=== conn4 attr2, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L), leanmd $(date -r $L/leanmd +%H:%M) pop-flush=$(nm -C $L/leanmd | grep -c CkRdmaDeviceChannelBatchFlush)"
lmd() {  # <tag> <channels> <grid> <lb>
  local tag=${1}_$(date +%H%M%S) ch=$2 grid="$3" lb=$4
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="$grid" STEPS=100 LB=$lb CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=150 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | pre-LB 1-20 $(mean_steps $R 1 20) | 25-40 $(mean_steps $R 25 40) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R | grep -oE '^[0-9]+' | head -2 | tr '\n' ' ')] stalls=$(cat $L/ranklogs/$tag/rank_*.log | grep -ac CHAN-STALL)"
  cat $L/ranklogs/$tag/rank_*.log | grep -a 'chan-stats\] put batches' | sed -E 's/pid=[0-9]+ //' | head -1 | cut -c1-110
}
lmd c4b_async_ch 1 "8 4 4" async; lmd c4b_async_zc 0 "8 4 4" async; lmd c4b_big_async_ch 1 "16 8 8" async
echo "=== conn4 attr2 done $(date +%T), left $(squeue -j $JID -h -o %L)"
