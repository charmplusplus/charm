#!/bin/bash
# after_now2.sh <jobid> (18:05): results on the write protocol (shm same-host, message notes across hosts):
# chantest rdma, leanmd LB 16x8x8 ch/ctl + 8x4x4 pair, no-LB 16x8x8 ch/ctl, jacobi3d zc/ch x3. No time guards.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; cd $L; ulimit -c 0
bash $G/chantest/test_chantest.sh $JID 24 3 2>&1 | grep -E 'rdma' | cut -c1-100
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <channels> <grid> <tag> <tmo> <lb>
  local ch=$1 grid=$2 tag=$3 tmo=$4 lb=$5 TAG=${3}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env CHANNELS=$ch LB=$lb BIN=leanmd.imm GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
}
larm 1 "16 8 8" w_ch_async_16x8x8 150 async
larm 0 "16 8 8" w_ctl_async_16x8x8 150 async
larm 1 "8 4 4" w_ch_async_8x4x4 60 async
larm 0 "8 4 4" w_ctl_async_8x4x4 60 async
larm 1 "16 8 8" w_ch_none_16x8x8 200 none
larm 0 "16 8 8" w_ctl_none_16x8x8 200 none
cd $J; export PPN=4 GPP=1 PES=4 ITERS=200 WARM=10 TMO=120
for g in 512 1024 2048; do for mode in zc ch; do
  TAG=j3dw_${mode}_g${g}_$(date +%H%M%S); env MODE=$mode GRID=$g $J/run_j3d.sh $JID $TAG > /dev/null 2>&1
  R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | $(grep -a 'Checksum' $R | head -1) | bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed|CHAN-WATCH')"
done; done
echo "=== now2 done $(date +%T); node KEPT"
