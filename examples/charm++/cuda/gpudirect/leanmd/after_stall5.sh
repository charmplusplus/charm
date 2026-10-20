#!/bin/bash
# after_stall5.sh <jobid>: the final build (slot lifetime + older resolutions ignored) outside the LB stall:
# chantest.fix2 (all tiers), leanmd.fix2 16x8x8 no-LB, jacobi3d.fix2 channels 512/1024. Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; cd $L; ulimit -c 0
CTBIN=chantest.fix2 bash $G/chantest/test_chantest_bin.sh $JID 24 3 2>&1 | grep -E 'PASS|FAIL' | cut -c1-120
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
TAG=s5_ch_none_16x8x8_$(date +%H%M%S); R=$L/ranklogs/$TAG
env CHANNELS=1 LB=none BIN=leanmd.fix2 GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=200 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | imm=$(cat $R/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
cd $J; export PPN=4 GPP=1 PES=4 ITERS=200 WARM=10 TMO=120
for g in 512 1024; do TAG=j3df_ch_g${g}_$(date +%H%M%S); env BIN=jacobi3d.fix2 MODE=ch GRID=$g $J/run_j3d_bin.sh $JID $TAG > /dev/null 2>&1; R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | $(grep -a 'Checksum' $R | head -1) | bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed|CHAN-WATCH')"; done
echo "=== stall5 done $(date +%T); node KEPT"
