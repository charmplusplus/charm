#!/bin/bash
# after_ctl2.sh <jobid>: no-LB control band, 16x8x8 CHANNELS=0 x2 (channels measured 1518.1-1520.5 in four runs
# against a single control of 1509.1: is the 0.7% real). Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for r in 1 2; do TAG=c2_ctl_none_16x8x8_r${r}_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=0 LB=none BIN=leanmd GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=200 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
done
echo "=== ctl2 done $(date +%T); node KEPT"
