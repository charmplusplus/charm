#!/bin/bash
# after3_chan7.sh <jobid>: the bare send with libfabric's registration cache monitor set to kdreg2 and to uffd
# (the memhooks monitor deadlocked with CUDA, which is why every run disables the cache; disabled = a map, copy and
# unmap of the user buffer on every send). Never sallocs/scancels.
JID=$1; SC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/sendcost; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ENVL="LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so"
echo "=== after3_chan7 $(date +%T), left $(left_s) s"
for mon in kdreg2 uffd; do
  [ "$(left_s)" -gt 40 ] || { echo "skip $mon"; break; }
  LOG=$SC/logs/sendcost_rdma_mon${mon}_$(date +%H%M%S).log
  env $ENVL FI_MR_CACHE_MONITOR=$mon timeout 40 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$SC stdbuf -oL -eL ./sendcost +ppn 1 -b 20000 -p 300 > $LOG 2>&1
  echo "$(date +%T) monitor=$mon rc=$? | $(grep -E '^(converse|entry) ' $LOG | tr '\n' '|' | cut -c1-300)"
  grep -h 'am-send' $LOG | head -1; grep -aiE 'error|abort|deadlock|warn' $LOG | head -2 | cut -c1-160
done
echo "=== after3_chan7 done $(date +%T), left $(left_s) s; node KEPT"
