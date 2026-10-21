#!/bin/bash
# conn1_zcstats.sh <jobid>: leanmd 16x8x8 one node, NO LB, 40 steps, CHARM_ZC_STATS=1: per-put / per-message host time by
# tier, channels then zero-copy, to locate the 2.3% steady-state gap at 448 objects per PE (ch 2947 vs zc 2882 ms/step).
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
echo "=== conn1 zcstats, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
for ch in 1 0; do
  tag=c1_big_stats_$( [ $ch = 1 ] && echo ch || echo zc )_$(date +%H%M%S)
  CHARM_ZC_STATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="16 8 8" STEPS=40 LB=none CHANNELS=$ch BIN=leanmd POOL_ALLOC=vmm POOL=16384 \
    LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=300 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  rc=$?; R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-40 $(mean_steps $R 21 40) ms"
  cat $L/ranklogs/$tag/rank_*.log $L/logs/$tag.log 2>/dev/null | grep -a 'ZC-STATS\|zc-stats\|\[ZC\|put_us\|credit_us\|per put\|per message' | sort | uniq -c | sort -rn | head -24 | cut -c1-170
done
echo "=== conn1 zcstats done $(date +%T)"
