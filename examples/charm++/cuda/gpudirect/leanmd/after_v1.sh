#!/bin/bash
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
arm() { local lb=$1 ch=$2 grid=$3 tag=$4 tmo=$5; [ "$(left_s)" -gt $((tmo+15)) ] || { echo "skip $tag: $(left_s) s left"; return 1; }; local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=$tmo LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 BIN=leanmd.v1 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"; }
echo "=== V1 pairs (write protocol, 20:08 pup) $(date +%T), left $(left_s) s"
arm async 0 "16 8 8" v1_ctl_async_16x8x8 150
arm async 1 "8 4 4" v1_ch_async_8x4x4 100
arm async 0 "8 4 4" v1_ctl_async_8x4x4 100
echo "=== V1 done $(date +%T), left $(left_s) s"
