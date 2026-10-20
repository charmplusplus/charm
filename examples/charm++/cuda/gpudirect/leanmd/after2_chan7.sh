#!/bin/bash
# after2_chan7.sh <jobid>: where LCI's post_am_x spends 10 us: a 2-second burst (-b 200000) from PE 0 to the other
# node, with stack snapshots of the sender every 0.3 s (gdb, innermost frames). Never sallocs/scancels.
JID=$1; SC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/sendcost; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
NODES=($(scontrol show hostnames $(squeue -j $JID -h -o %N))); N0=${NODES[0]}
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ENVL="LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so"
echo "=== after2_chan7 $(date +%T), left $(left_s) s, sender node $N0"
LOG=$SC/logs/sendcost_rdma_burst_$(date +%H%M%S).log
env $ENVL timeout 90 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$SC stdbuf -oL -eL ./sendcost +ppn 1 -b 300000 -p 10 > $LOG 2>&1 &
RP=$!
sleep 6
for i in 1 2 3 4 5 6; do
  srun --jobid=$JID --overlap -N 1 -n 1 -w $N0 --cpu-bind=none bash -c 'p=$(pgrep -x sendcost | head -1); [ -n "$p" ] && timeout 20 gdb -p $p -batch -ex "set pagination off" -ex "thread 1" -ex "bt 14" 2>/dev/null | grep -E "^#[0-9]+ " | cut -c1-170 | sed "s/^/    /"' 2>&1 | head -14
  echo "    --"
  sleep 0.4
done
wait $RP
echo "$(date +%T) burst: $(grep -E '^(converse|entry) burst' $LOG | tr '\n' '|' | cut -c1-200)"
grep -h 'am-send' $LOG | head -2
echo "=== after2_chan7 done $(date +%T), left $(left_s) s; node KEPT"
