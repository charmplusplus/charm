#!/bin/bash
# after5_chan6.sh <jobid>: which path loses the event in the two-process layout: channel modes alone (-m 10) and
# zero-copy modes alone (-m 5), up to 3 tries each, hang = no finish within 20 s. Never sallocs, never scancels.
JID=$1; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "=== after5 $(date +%T), left $(left_s) s"
for mask in 10 5 10 5 10 5; do
  [ "$(left_s)" -gt 45 ] || { echo "stop: $(left_s) s left"; break; }
  GL=$C/logs/channel_ipc_mask${mask}_$(date +%H%M%S).log
  ( export LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID; ulimit -c 0
    timeout 32 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 2 --ntasks-per-node=2 -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$C stdbuf -oL -eL \
      ./channel -m $mask -i 100 -l 20 -r 200 -R 20 +ppn 2 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm +gpuipceventpool 256 > $GL 2>&1 ) &
  RP=$!
  for s in $(seq 1 20); do sleep 1; grep -q 'Elapsed:' $GL 2>/dev/null && break; done
  if grep -q 'Elapsed:' $GL; then echo "  mask $mask ($([ $mask = 10 ] && echo channel || echo zero-copy) modes): completed, $(grep -c ' B | one-way' $GL) sizes"; else echo "  mask $mask ($([ $mask = 10 ] && echo channel || echo zero-copy) modes): HUNG with $(grep -c ' B | one-way' $GL) size lines"; fi
  wait $RP
done
echo "=== after5 done $(date +%T), left $(left_s) s; node KEPT"
