#!/bin/bash
# conn1_sync.sh <jobid>: leanmd 16x8x8 ONE node, SYNC balancer, zero-copy then channels, one run each. The async pairs at
# 32 queues gave zc 1538/1549 vs ch 1625/1568 (ms/step 42-100) with a pre-LB tie and a near tie in the quiet window
# after the Metis moves, and LOWER device time per interval in the channel arm: the loss is waiting during moves in
# flight. ONE VARIABLE: async -> sync (moves at a barrier, nothing overlapped). If the sync pair ties, the channel
# async move protocol (sender holds, stand-down) is the cause. Waits for the jacobi3d runs. Never sallocs/scancels.
JID=$1; [ -n "$JID" ] || exit 1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
until grep -q 'conn1 jacobi3d done' $L/logs/conn1_j3d.out 2>/dev/null; do sleep 10; done
echo "=== conn1 sync pair, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for ch in 0 1; do
  tag=c1_big_sync_$( [ $ch = 1 ] && echo ch || echo zc )_$(date +%H%M%S)
  CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="16 8 8" STEPS=100 LB=sync CHANNELS=$ch BIN=leanmd POOL_ALLOC=vmm POOL=16384 \
    LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=480 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  rc=$?; R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | pre-LB 1-20 $(mean_steps $R 1 20) | 25-40 $(mean_steps $R 25 40) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R 2>/dev/null | grep -oE '^[0-9]+' | head -6 | tr '\n' ' ')] | LB steps: $(grep -ac 'step [0-9]* finished' $R)"
done
echo "=== conn1 sync pair done $(date +%T)"
