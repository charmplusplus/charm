#!/bin/bash
# after4_chan6.sh <jobid>: the two-process (ipc) benchmark hang is intermittent (12:34 hung after 2 sizes, 12:39 ran
# all 5): run the ipc layout up to 4 times with a 50 s timeout; when a run has not finished within 20 s, dump the
# stacks of both ranks with gdb before the timeout kills it. Never sallocs, never scancels.
JID=$1; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "=== after4 $(date +%T), left $(left_s) s, node $N0"
for t in 1 2 3 4; do
  [ "$(left_s)" -gt 80 ] || { echo "stop: $(left_s) s left"; break; }
  GL=$C/logs/channel_array_ipc_hunt${t}_$(date +%H%M%S).log
  ( export LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID CHARM_DEBUG_MIGRATE=1; ulimit -c 0
    timeout 50 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 2 --ntasks-per-node=2 -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$C stdbuf -oL -eL \
      ./channel -i 100 -l 20 -r 200 -R 20 +ppn 2 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm +gpuipceventpool 256 > $GL 2>&1 ) &
  RP=$!
  for s in $(seq 1 20); do sleep 1; grep -q 'Elapsed:' $GL 2>/dev/null && break; done
  if grep -q 'Elapsed:' $GL 2>/dev/null; then echo "  try $t: completed ($(grep -c ' B | one-way' $GL) sizes)"; wait $RP; continue; fi
  echo "  try $t: HUNG after 20 s with $(grep -c ' B | one-way' $GL) size lines; stacks:"
  srun --jobid=$JID --overlap -N 1 -n 1 -w $N0 --cpu-bind=none bash -c 'for p in $(pgrep -x channel); do echo "  == pid $p"; timeout 25 gdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt" 2>/dev/null | grep -E "^Thread|^#[0-9]+ " | grep -vE "libc\.so|libpthread|clock_gettime|sched_yield|__GI_|in poll|epoll" | head -44 | cut -c1-220 | sed "s/^/    /"; done' 2>&1
  wait $RP
  echo "  try $t: rc=$(grep -o 'rc=[0-9]*' $GL | tail -1) sizes $(grep -c ' B | one-way' $GL); log $GL"
  break
done
echo "=== after4 done $(date +%T), left $(left_s) s; node KEPT"
