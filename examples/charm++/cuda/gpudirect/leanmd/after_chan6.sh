#!/bin/bash
# after_chan6.sh <jobid>: the open items, in chan6's leftover minutes (launched by hand after the chain's runner is
# stopped at the start of its 8x4x4 loop). 1. pic2d at two nodes on HEAD: the 01:45 aborts ("index compress
# overflow", a 2D index holding two packed shorts) came from objects compiled on 09-18 and only relinked after the
# 10-17 header changes; rebuilt from clean 11:55. 2. The channel benchmark as a 1D chare array (the group version
# hangs across processes on one node; kept as channel_group for a stack capture). 3. The credit's cost split
# (alloc / converse send, same host / other host) printed by CHARM_ZC_STATS, and the two-node large-message gap
# with +lci_ndevices 1 against the default. Never sallocs, never scancels.
JID=$1; [ -n "$JID" ] || { echo "usage: after_chan6.sh <jobid>"; exit 1; }
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d
C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); [ -n "$END_EPOCH" ] || { echo "no end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
NODES=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | tr '\n' ' '); N0=${NODES%% *}
echo "=== after_chan6 $(date +%T), job $JID nodes [$NODES], left $(left_s) s; pic2d $(ls -la --time-style=+%m-%d_%H:%M $P/pic2d | awk '{print $6}') channel $(ls -la --time-style=+%m-%d_%H:%M $C/channel | awk '{print $6}')"
pic_arm() {  # <mode> <iters>: the lb2node recipe (8 processes x 8 PEs, 256 patches)
  local mode=$1 iters=$2 lb
  [ "$(left_s)" -gt 90 ] || { echo "skip pic2d $mode: $(left_s) s left"; return 1; }
  local TAG=pic2d_${mode}_$(date +%H%M%S); local LOG=$P/logs/$TAG.log; mkdir -p $P/logs
  local BAL="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBCommOn +LBDebug 1"
  case $mode in none) lb="-f 99999";; sync) lb="-f 50 -b 50 $BAL";; async) lb="-f 50 -b 50 -a -l 40 $BAL +LBAsync";; esac
  echo "[pic2d] $(date +%T) tag=$TAG LB=$mode iters=$iters args: $lb" > $LOG
  env LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout 150 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$P \
    $P/numa_wrap.sh stdbuf -oL -eL ./pic2d -W 2048 -H 2048 -w 128 -h 128 -p 4 -d 1 -F 0.9 -s 0.1 -c 20 -r 32 -i $iters -u 10 $lb +gpushm +gpupool +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[pic2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) pic2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LOG) ms/iter | iters $(grep -ac '^Iteration\|^iter' $LOG) | metis-partitions $(grep -ac 'Metis partitioning' $LOG) | migrating $(grep -aoE '#Objects migrating: [0-9]+' $LOG | awk '{s+=$3} END{print s+0}') | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|mismatch|Failed assert' $LOG)"
  grep -aiE 'fatal|abort|Segmentation|CORRUPTION|mismatch|exceed capacity|reason:|compress overflow|checksum' $LOG | head -3 | cut -c1-200 | sed 's/^/    /'
}
ch_arm() {  # <layout> <extra args...>: one channel-benchmark run; prints the size lines and the host-cost stats
  local lay=$1; shift
  [ "$(left_s)" -gt 90 ] || { echo "skip channel $lay: $(left_s) s left"; return 1; }
  echo "--- channel $lay $*"
  bash $C/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20 "$@" 2>&1 | grep -E 'channel .*rc=|one-way|zc-host|Elapsed' | cut -c1-230 | sed 's/^/  /'
}
echo "--- 1. pic2d, two nodes, HEAD binary rebuilt from clean"
pic_arm none 150
echo "--- 2. channel benchmark as a chare array: cross-process one node (ipc), then two nodes (rdma) with the credit cost split"
ch_arm ipc
ch_arm rdma
echo "--- 3. two-node large-message gap: one LCI device"
ch_arm rdma +lci_ndevices 1
pic_arm async 150
echo "--- 4. the group version across processes on one node: 20 s, then stacks"
if [ "$(left_s)" -gt 120 ]; then
  GL=$C/logs/channel_group_ipc_$(date +%H%M%S).log
  ( export LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID CHARM_DEBUG_MIGRATE=1; ulimit -c 0
    timeout 75 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 2 --ntasks-per-node=2 -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$C stdbuf -oL -eL \
      ./channel_group -i 20 -l 5 -r 20 -R 5 +ppn 2 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm +gpuipceventpool 256 > $GL 2>&1 ) &
  sleep 25
  echo "  group run after 25 s: $(grep -c ' B | one-way' $GL) size lines; CHAN lines: $(grep -c 'CHAN-' $GL); last: $(grep -v '^HAPI>' $GL | tail -1 | cut -c1-120)"
  grep -a 'CHAN-' $GL | sed -E 's/[0-9a-f]{6,}/X/g; s/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -6 | cut -c1-200 | sed 's/^/    /'
  srun --jobid=$JID --overlap -N 1 -n 1 -w $N0 --cpu-bind=none bash -c 'for p in $(pgrep -x channel_group); do echo "  == pid $p"; timeout 40 gdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt" 2>/dev/null | grep -E "^Thread|^#[0-9]+ " | grep -vE "libc\.so|libpthread|clock_gettime|sched_yield|__GI_" | head -36 | cut -c1-200 | sed "s/^/    /"; done' 2>&1
  wait
  echo "  group run: rc=$(grep -o 'rc=[0-9]*' $GL | tail -1) size lines $(grep -c ' B | one-way' $GL)"
fi
pic_arm sync 150
ch_arm memcpy
echo "=== after_chan6 done $(date +%T), left $(left_s) s; node KEPT"
