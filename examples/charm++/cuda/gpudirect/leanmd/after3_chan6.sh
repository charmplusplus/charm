#!/bin/bash
# after3_chan6.sh <jobid>: the array benchmark hangs in the ipc layout (2 processes x 2 PEs, one node) where the
# group version with short counts completed. Run it for 25 s with the channel debug prints, then dump the stacks of
# both ranks with gdb, then let it die. Waits for after2 to finish. Never sallocs, never scancels.
JID=$1; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
while pgrep -f "^bash /u/bhosale/.*after2_chan6.sh" >/dev/null; do sleep 3; done
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "=== after3 $(date +%T), left $(left_s) s, node $N0"
[ "$(left_s)" -gt 100 ] || { echo "no time"; exit 0; }
GL=$C/logs/channel_array_ipc_dbg_$(date +%H%M%S).log
( export LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID CHARM_DEBUG_MIGRATE=1 CHARM_ZC_STATS=1; ulimit -c 0
  timeout 80 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 2 --ntasks-per-node=2 -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$C stdbuf -oL -eL \
    ./channel -i 100 -l 20 -r 200 -R 20 +ppn 2 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm +gpuipceventpool 256 > $GL 2>&1 ) &
sleep 30
echo "  array run after 30 s: $(grep -c ' B | one-way' $GL) size lines; last: $(grep -v '^HAPI>' $GL | tail -1 | cut -c1-140)"
grep -a 'CHAN-' $GL | sed -E 's/[0-9a-f]{6,}/X/g; s/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -8 | cut -c1-200 | sed 's/^/    /'
srun --jobid=$JID --overlap -N 1 -n 1 -w $N0 --cpu-bind=none bash -c 'for p in $(pgrep -x channel); do echo "  == pid $p"; timeout 40 gdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt" 2>/dev/null | grep -E "^Thread|^#[0-9]+ " | grep -vE "libc\.so|libpthread|clock_gettime|sched_yield|__GI_" | head -40 | cut -c1-200 | sed "s/^/    /"; done' 2>&1
wait
echo "  array run: rc=$(grep -o 'rc=[0-9]*' $GL | tail -1) size lines $(grep -c ' B | one-way' $GL)"
echo "=== after3 done $(date +%T), left $(left_s) s; node KEPT"
