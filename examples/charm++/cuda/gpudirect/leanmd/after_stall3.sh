#!/bin/bash
# after_stall3.sh <jobid>: catch the protocol trace of the 8x4x4 stall (CHARM_DEBUG_MIGRATE prints every
# resolution/closing/ack per channel). Up to 3 tries (the stall is intermittent); stops at the first stall and
# prints the stalled channel's trace from every rank. Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 CHARM_DEBUG_MIGRATE=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for r in 1 2 3; do
  TAG=s3_ch_async_8x4x4_r${r}_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=1 LB=async BIN=leanmd.fix GRID="8 4 4" STEPS=100 PPN=4 GPP=1 PES=8 TMO=90 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  rc=$(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1)
  echo "$(date +%T) $TAG: $rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | stalls=$(cat $R/rank_*.log | grep -ac 'CHAN-STALL') | trace lines=$(cat $R/rank_*.log | grep -ac 'CHAN-RESOLVE\|CHAN-CLOSE\|CHAN-STALE')"
  if [ "$rc" = "rc=124" ]; then
    for id in $(cat $R/rank_*.log | grep -a 'CHAN-STALL' | grep -ao 'channel [0-9a-f]*' | awk '{print $2}' | sort -u | head -2); do
      echo "=== trace of channel $id"
      for f in $R/rank_*.log; do n=$(grep -ac "channel $id\b" $f); [ "$n" -gt 0 ] && { echo "--- $(basename $f) ($n lines)"; grep -a "channel $id\b" $f | cut -c1-260 | tail -40; }; done
    done
    break
  fi
done
echo "=== stall3 done $(date +%T); node KEPT"
