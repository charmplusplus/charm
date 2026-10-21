#!/bin/bash
# conn4_small.sh <jobid>: the 8x4x4 question after the batched channel put: one pair showed channels 229.8 vs zero-copy
# 198.4 (16% behind) where every earlier 8x4x4 pair today tied. ONE more async pair (ch, zc) and the no-LB pair
# (ch, zc, 60 steps) to say whether it repeats and whether it is in the moves or the steady state. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
echo "=== conn4 small, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
lmd() {  # <tag> <channels> <steps> <lb>
  local tag=${1}_$(date +%H%M%S) ch=$2 steps=$3 lb=$4
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="8 4 4" STEPS=$steps LB=$lb CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=300 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | pre-LB 1-20 $(mean_steps $R 1 20) | 21-60 $(mean_steps $R 21 60) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R | grep -oE '^[0-9]+' | head -2 | tr '\n' ' ')] packs=$(cat $L/ranklogs/$tag/rank_*.log | grep -a '\[mig-window\] pid=[0-9]* pack' | sed -E 's/.*n=([0-9]+).*/\1/' | paste -sd+ | bc)"
}
lmd c4s_async_ch 1 100 async; lmd c4s_async_zc 0 100 async
lmd c4s_nolb_ch 1 60 none; lmd c4s_nolb_zc 0 60 none
echo "=== conn4 small done $(date +%T), left $(squeue -j $JID -h -o %L)"
