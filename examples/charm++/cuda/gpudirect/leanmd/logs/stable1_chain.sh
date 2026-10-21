#!/bin/bash
# stable1_chain.sh <jobid>: one node. (1) chantest: plain, migrateMe, balancer-path (+LBAsync, Stable snapshot) and
# the broken-promise abort, memcpy and ipc. (2) leanmd 8x4x4 async LB on the rebuilt runtime (zero-copy and channels)
# with the move-window tally, against the previous build; steady-state mean42-100 ms/step. Never scancels.
JID=$1; [ -n "$JID" ] || { echo "usage: stable1_chain.sh <jobid>"; exit 1; }
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
CT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
cd $L
echo "=== stable1 chain, job $JID, $(date +%T), nodes: $(squeue -j $JID -h -o %N)"
bash $CT/test_chantest_stable.sh $JID 24 3 2>&1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd() {  # <tag> <bin> <channels 0|1>
  local tag=${1}_$(date +%H%M%S) bin=$2 ch=$3
  CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="8 4 4" STEPS=100 LB=async CHANNELS=$ch BIN=$bin POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=300 \
    bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local log=$L/logs/$tag.log; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-100 $(mean_steps $R 21 100) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R 2>/dev/null | grep -oE '^[0-9]+' | head -6 | tr '\n' ' ')]"
  # the move-window tally, one line per stage per process: shown as the per-stage spread over the 4 processes
  cat $L/ranklogs/$tag/rank_*.log $log 2>/dev/null | grep -a '\[mig-window\]\|\[mig-stats\]' | sed -E 's/pid=[0-9]+ //' | sort | uniq -c | sort -k3 | head -40 | cut -c1-150
}
for r in 1 2; do
  lmd s1_new_zc_r$r leanmd 0
  lmd s1_new_ch_r$r leanmd 1
done
lmd s1_prev_zc leanmd.prev 0
lmd s1_prev_ch leanmd.prev 1
echo "=== stable1 chain done $(date +%T)"
