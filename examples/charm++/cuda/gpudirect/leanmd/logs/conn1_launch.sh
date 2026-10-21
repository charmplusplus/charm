#!/bin/bash
# conn1_launch.sh: one A40 node for an hour (same size as job 23047794: 64 CPUs, 4 GPUs), then conn1_chain.sh on it.
# No scancel here: the results are read first, then the node is released by hand.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
ulimit -c 0
echo "salloc submitted $(date +%T)"
salloc --no-shell --account=mzu-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --ntasks=4 --cpus-per-task=16 \
  --gpus-per-node=4 --mem=0 --time=01:00:00 --job-name=conn1 > $L/logs/conn1_salloc.log 2>&1
JID=$(grep -oE 'Granted job allocation [0-9]+' $L/logs/conn1_salloc.log | grep -oE '[0-9]+$')
[ -n "$JID" ] || { echo "no allocation $(date +%T): $(tr '\n' ' ' < $L/logs/conn1_salloc.log)"; exit 1; }
echo "job $JID granted $(date +%T) on $(squeue -j $JID -h -o %N)"
bash $L/logs/conn1_chain.sh $JID > $L/logs/conn1_chain.out 2>&1
echo "chain exited rc=$? $(date +%T)"
