#!/bin/bash
# conn2_more.sh <jobid>: two more leanmd 16x8x8 async pairs (zc, ch, zc, ch) after conn2_chain, to widen the band
# from 2 to 4 runs per arm (first two pairs on gpub001: zc 1639.4/1579.5, ch 1548.2/1638.9: overlapping). Purpose:
# tie or win after the protocol change. Never sallocs, never scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
until grep -q 'conn2 chain done\|conn2 chain stopped' $L/logs/conn2_chain.out 2>/dev/null; do sleep 10; done
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
echo "=== conn2 more, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
for r in 3 4; do for ch in 0 1; do
  tag=c2_big_$( [ $ch = 1 ] && echo ch || echo zc )_r${r}_$(date +%H%M%S)
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="16 8 8" STEPS=100 LB=async CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=480 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  rc=$?; R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | pre-LB 1-20 $(mean_steps $R 1 20) | 25-40 $(mean_steps $R 25 40) | mean42-100 $(mean_steps $R 42 100) ms | stalls=$(cat $L/ranklogs/$tag/rank_*.log | grep -ac CHAN-STALL)"
  cat $L/ranklogs/$tag/rank_*.log 2>/dev/null | grep -a '\[mig-window\]' | sed -E 's/.*\[mig-window\] pid=[0-9]+ //' | awk '$1=="chan-hold"||$1=="stand-down"||$1=="pack"{n=substr($2,3)+0; m=$4+0; mx=$7+0; N[$1]+=n; S[$1]+=n*m; if(mx>MX[$1])MX[$1]=mx} END{for(s in N) printf "   %-11s n=%6d mean=%7.1f ms max=%8.1f ms\n", s, N[s], (N[s]?S[s]/N[s]:0), MX[s]}' | sort
done; done
echo "=== conn2 more done $(date +%T), left $(squeue -j $JID -h -o %L)"
