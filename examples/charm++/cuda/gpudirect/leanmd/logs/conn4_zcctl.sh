#!/bin/bash
# conn4_zcctl.sh <jobid>: the zero-copy 16x8x8 async control for the batched channel run of 20:18 (1618.0 ms/step),
# same node and session. One run. Waits for conn4_chain. Never sallocs, never scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
until grep -q 'conn4 chain done\|Chain stopped' $L/logs/conn4_chain.out 2>/dev/null; do sleep 10; done
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
echo "=== conn4 zc control, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
tag=c4_big_async_zc_$(date +%H%M%S)
CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="16 8 8" STEPS=100 LB=async CHANNELS=0 BIN=leanmd \
  POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=480 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
rc=$?; R=$L/ranklogs/$tag/rank_0.log
echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | pre-LB 1-20 $(mean_steps $R 1 20) | 25-40 $(mean_steps $R 25 40) | mean42-100 $(mean_steps $R 42 100) ms"
echo "=== conn4 zc control done $(date +%T), left $(squeue -j $JID -h -o %L)"
