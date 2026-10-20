#!/bin/bash
# after_imm5.sh <jobid> (18:36): immediate notes, LCI no longer allocating 0 bytes per note; leanmd.immon no-LB
# 16x8x8 (segfaulted at step 2 at 18:26, clean under gdb), jacobi3d.immon ch x3 (aborted 18:26, clean under gdb), then LB.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <channels> <grid> <tag> <tmo> <lb>
  local ch=$1 grid=$2 tag=$3 tmo=$4 lb=$5 TAG=${3}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env CHANNELS=$ch LB=$lb BIN=leanmd.immon GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | imm=$(cat $R/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') zeroalloc=$(cat $R/rank_*.log | grep -ac 'CmiAlloc: size') watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed') $(grep -a 'Segmentation\|Aborted' $L/logs/$TAG.log | head -1 | cut -c1-60)"
  cat $R/rank_*.log | grep -aiE 'Reason|CHAN-STALL' | head -2 | cut -c1-200 | sed 's/^/    /'
}
larm 1 "16 8 8" i5_ch_none_16x8x8 200 none
cd $J; export PPN=4 GPP=1 PES=4 ITERS=200 WARM=10 TMO=120
for g in 512 1024 2048; do
  TAG=j3di5_ch_g${g}_$(date +%H%M%S); env BIN=jacobi3d.immon MODE=ch GRID=$g $J/run_j3d_bin.sh $JID $TAG > /dev/null 2>&1
  R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | $(grep -a 'Checksum' $R | head -1) | imm=$(cat $J/ranklogs/$TAG/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed|CHAN-WATCH') | $(cat $J/ranklogs/$TAG/rank_*.log | grep -a 'Reason' | head -1 | cut -c1-80)"
done
cd $L; larm 1 "16 8 8" i5_ch_async_16x8x8 150 async
echo "=== imm5 done $(date +%T); node KEPT"
