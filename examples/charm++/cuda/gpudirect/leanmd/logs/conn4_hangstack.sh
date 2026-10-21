#!/bin/bash
# conn4_hangstack.sh <jobid>: the 8x4x4 async channel run that hangs before step 1 on the flush-at-CkCallstackPop build
# (20:31 leanmd), with the stacks of every leanmd thread on the node taken 100 s after launch. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
tag=c4h_async_ch_$(date +%H%M%S)
echo "=== hang stack run, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
( CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="8 4 4" STEPS=100 LB=async CHANNELS=1 BIN=leanmd POOL_ALLOC=vmm POOL=16384 \
    LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=200 bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1; echo "run rc=$? $(date +%T)" ) &
sleep 100
echo "--- steps so far: $(grep -ac '^Step [0-9]* Benchmark' $L/ranklogs/$tag/rank_0.log 2>/dev/null); stacks at $(date +%T)"
srun --jobid=$JID --overlap -N 1 -n 1 --exact bash -c 'for p in $(pgrep -x leanmd); do echo "== leanmd pid $p"; gstack $p; done' > $L/logs/conn4_hangstack.txt 2>&1
echo "stack file: $(grep -c '^Thread' $L/logs/conn4_hangstack.txt) threads, $(grep -c '^== leanmd' $L/logs/conn4_hangstack.txt) processes"
wait
echo "=== hang stack run done $(date +%T)"
