#!/bin/bash
# after_imm6.sh <jobid> (18:39): immediate notes under LB at 8x4x4 (stalled after the migrations at 18:29, before the
# zero-byte fix), then the 2-repeat band of after_imm4.sh (jacobi3d message vs immediate notes, 512/1024).
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <channels> <grid> <tag> <tmo> <lb> <bin>
  local ch=$1 grid=$2 tag=$3 tmo=$4 lb=$5 bin=$6 TAG=${3}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env CHANNELS=$ch LB=$lb BIN=$bin GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | imm=$(cat $R/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
  cat $R/rank_*.log | grep -aiE 'Reason|CHAN-STALL|CHAN-WATCH' | head -2 | cut -c1-200 | sed 's/^/    /'
}
larm 1 "8 4 4" i6_ch_async_8x4x4 60 async leanmd.immon
bash $L/after_imm4.sh $JID
echo "=== imm6 done $(date +%T); node KEPT"
