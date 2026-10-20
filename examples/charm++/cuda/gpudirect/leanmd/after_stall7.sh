#!/bin/bash
# after_stall7.sh <jobid>: the guard's own print only exists under CHARM_DEBUG_MIGRATE (the watchdog tally that
# carries the counter prints only on a quiet PE, so a healthy run shows nothing): 8x4x4 async on leanmd.fix2 with
# the trace on, at most 4 runs, until "[CHAN-STALE] ... ignored" appears. Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 CHARM_DEBUG_MIGRATE=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for r in 1 2 3 4; do
  TAG=s7_ch_async_8x4x4_r${r}_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=1 LB=async BIN=leanmd.fix2 GRID="8 4 4" STEPS=100 PPN=4 GPP=1 PES=8 TMO=90 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  ign=$(cat $R/rank_*.log | grep -a 'CHAN-STALE' | grep -ac 'ignored')
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | older resolutions ignored=$ign, double replies=$(cat $R/rank_*.log | grep -a 'CHAN-RESOLVE' | awk '{print $4, $NF, $(NF-5), $(NF-4)}' | sort | uniq -d | wc -l) stalls=$(cat $R/rank_*.log | grep -ac 'CHAN-STALL') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
  cat $R/rank_*.log | grep -a 'CHAN-STALE' | grep -a 'ignored' | head -3 | cut -c1-220 | sed 's/^/    /'
  [ "$ign" -gt 0 ] && break
done
echo "=== stall7 done $(date +%T); node KEPT"
