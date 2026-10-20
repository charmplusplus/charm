#!/bin/bash
# run_channel.sh <jobid> <layout: memcpy|ipc|rdma> [pool: none|buddy|vmm] [channel args...]
# One run of the channel-vs-zero-copy microbenchmark on an allocation already held. Layouts: memcpy = one
# process with two PEs (same-process stream copies); ipc = two processes of two PEs on one node (PE 0 and PE 3 exchange: device-to-device copies
# through the import cache); rdma = one process on each of two nodes (one-sided puts and rgets). The pool
# flag picks the device allocator the buffers come from. CHARM_ZC_STATS=1 so the zero-copy path's own
# receive-side host cost ([zc-host] recv-issue) is printed per process at exit. Never sallocs, never scancels.
JID=$1; LAYOUT=$2; POOLM=${3:-vmm}; shift 3 2>/dev/null
[ -n "$JID" ] && [ -n "$LAYOUT" ] || { echo "usage: run_channel.sh <jobid> <memcpy|ipc|rdma> [none|buddy|vmm] [args]"; exit 1; }
D=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
LCI=/u/bhosale/lci/build-cuda/lib
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID CHARM_ZC_STATS=1
ulimit -c 0
mkdir -p $D/logs
case $LAYOUT in
  memcpy) N=1; NP=1; PPN=1; PPNARG="+ppn 2";;
  ipc)    N=1; NP=2; PPN=2; PPNARG="+ppn 2";;   # 4 PEs: PE 0 and PE 3 are the pair, in different processes
  rdma)   N=2; NP=2; PPN=1; PPNARG="+ppn 1";;
  *) echo "unknown layout $LAYOUT"; exit 1;;
esac
case $POOLM in
  none)  POOLARGS="+gpushm +gpuipceventpool 256";;
  buddy) POOLARGS="+gpushm +gpupool +gpupoolsize 256 +gpuipceventpool 256";;
  vmm)   POOLARGS="+gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm +gpuipceventpool 256";;
  *) echo "unknown pool mode $POOLM"; exit 1;;
esac
# The vmm-fixed LCI is preloaded for every multi-process run, as the applications do.
[ "$NP" -gt 1 ] && export LD_PRELOAD=$LCI/liblci.so LD_LIBRARY_PATH=$LCI:$LD_LIBRARY_PATH
LOG=$D/logs/channel_${LAYOUT}_${POOLM}_$(date +%H%M%S).log
echo "[run_channel] $(date +%T) layout=$LAYOUT pool=$POOLM nodes=$N np=$NP args: $* $PPNARG $POOLARGS" | tee $LOG
timeout 300 srun --jobid=$JID --overlap --mpi=cray_shasta -N $N -n $NP --ntasks-per-node=$PPN --gpus-per-node=4 \
    --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$D stdbuf -oL -eL \
    ./${CHBIN:-channel} "$@" $PPNARG $POOLARGS >> $LOG 2>&1
rc=$?
echo "[run_channel] rc=$rc $(date +%T)" >> $LOG
echo "$(date +%T) channel $LAYOUT/$POOLM: rc=$rc | $(grep -c ' B | one-way' $LOG) size lines | bad=$(grep -ciE 'abort|Fatal|Segmentation|error' $LOG) | $(grep -o 'Elapsed: [0-9.]* s' $LOG)"
grep -E '^# channel| B \| one-way' $LOG | cut -c1-230 | sed 's/^/    /'
grep -a '^\[zc-host\]' $LOG | cut -c1-200 | sed 's/^/    /'
[ "$rc" = 0 ] || grep -iE 'abort|Fatal|Segmentation|error|CkDeviceChannel|CkDevicePersistent' $LOG | head -4 | cut -c1-200 | sed 's/^/    /'
