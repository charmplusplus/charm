#!/bin/bash
# after_ctl1.sh <jobid>: control band under async LB (CHANNELS=0, same build): 8x4x4 x3 (the 15 channel runs span
# 114-153 ms/step against one control of 125.8), 16x8x8 x1 (channels 838-854 against one control of 843.8). Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <grid> <tag> <tmo>
  local grid=$1 tag=$2 tmo=$3 TAG=${2}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env CHANNELS=0 LB=async BIN=leanmd GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
}
larm "8 4 4" c1_ctl_async_8x4x4_r1 90
larm "8 4 4" c1_ctl_async_8x4x4_r2 90
larm "8 4 4" c1_ctl_async_8x4x4_r3 90
larm "16 8 8" c1_ctl_async_16x8x8 150
echo "=== ctl1 done $(date +%T); node KEPT"
