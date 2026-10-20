#!/bin/bash
# after_stall2.sh <jobid>: the fix (signal slot kept until the acked last put is taken) on leanmd.fix / chantest.fix:
# chantest rdma, leanmd 8x4x4 async (the stall), leanmd 16x8x8 async (no regression). Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; cd $L; ulimit -c 0
CTBIN=chantest.fix bash $G/chantest/test_chantest_bin.sh $JID 24 3 2>&1 | grep -E 'rdma' | cut -c1-150
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <channels> <grid> <tag> <tmo> <lb> <bin>
  local ch=$1 grid=$2 tag=$3 tmo=$4 lb=$5 bin=$6 TAG=${3}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env CHANNELS=$ch LB=$lb BIN=$bin GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG ($bin): $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | imm=$(cat $R/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
  cat $R/rank_*.log | grep -a 'immediate notes dropped\|immediate headers\|signal slots released' | sort | uniq | head -8 | cut -c1-230 | sed 's/^/    /'
  cat $R/rank_*.log | grep -aiE 'Reason|CHAN-STALL' | head -2 | cut -c1-200 | sed 's/^/    /'
}
larm 1 "8 4 4" s2_ch_async_8x4x4 90 async leanmd.fix
larm 1 "16 8 8" s2_ch_async_16x8x8 150 async leanmd.fix
larm 1 "8 4 4" s2_ch_async_8x4x4b 90 async leanmd.fix
echo "=== stall2 done $(date +%T); node KEPT"
