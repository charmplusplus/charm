#!/bin/bash
# conn3_oneflag.sh <jobid>: the one-flag channel put, on the node after conn3_chain. Waits for the chain and for the
# build (leanmd.oneflag). (1) chantest all arms on the new library; (2) leanmd.oneflag 16x8x8 no LB 30 steps with the
# counters (target: flags per put 2.26 -> ~1.3; mean11-30 against 2588.2 ch / 2502.4 zc on this node today);
# (3) leanmd.oneflag 16x8x8 async x1 and 8x4x4 async x1 (this node's zero-copy at 32 queues, 14:40-14:46: 8x4x4
# 203-219; 16x8x8 zc 1538/1549 from the 15:00 pairs). Never sallocs, never scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
CT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
S=/tmp/claude-54324/-u-bhosale-charm-reconverse/52f63e60-0155-4055-9277-35b69c94a33f/scratchpad
until grep -q 'conn3 chain done' $L/logs/conn3_chain.out 2>/dev/null && grep -q '=== build done' $S/build_oneflag.log 2>/dev/null; do sleep 10; done
echo "=== conn3 oneflag, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L); build: $(grep -c 'built leanmd.oneflag' $S/build_oneflag.log) $(grep -c 'relinked chantest' $S/build_oneflag.log)"
[ -x $L/leanmd.oneflag ] || { echo "no leanmd.oneflag; stopped"; exit 1; }
echo "--- 1. chantest on the one-flag library"
bash $CT/test_chantest_stable.sh $JID 24 3 2>&1 | grep -E 'PASS|FAIL|done' | cut -c1-120
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd() {  # <tag> <grid> <steps> <lb>
  local tag=${1}_$(date +%H%M%S) grid="$2" steps=$3 lb=$4
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="$grid" STEPS=$steps LB=$lb CHANNELS=1 BIN=leanmd.oneflag \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=480 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | pre-LB 1-20 $(mean_steps $R 1 20) | 11-30 $(mean_steps $R 11 30) | mean42-100 $(mean_steps $R 42 100) ms | stalls=$(cat $L/ranklogs/$tag/rank_*.log | grep -ac CHAN-STALL)"
  cat $L/ranklogs/$tag/rank_*.log $L/logs/$tag.log 2>/dev/null | grep -a 'hapi-stats\|chan-stats' | sed -E 's/pid=[0-9]+ //' | head -4 | cut -c1-170
}
echo "--- 2. 16x8x8 no LB, 30 steps"; lmd c3_oneflag_nolb "16 8 8" 30 none
echo "--- 3. async"; lmd c3_oneflag_big_async "16 8 8" 100 async; lmd c3_oneflag_async "8 4 4" 100 async
echo "=== conn3 oneflag done $(date +%T), left $(squeue -j $JID -h -o %L)"
