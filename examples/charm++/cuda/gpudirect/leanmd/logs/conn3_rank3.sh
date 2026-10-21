#!/bin/bash
# conn3_rank3.sh <jobid>: the BUSIEST rank under nsys, 10 steps, channels (two-flag leanmd) then zero-copy. Rank 3 holds
# the top of the density gradient (-computemap local, -density gradient), so its GPU sets the step; rank 0 (profiled so
# far) is the lightest. Reports: /scratch/mzu/bhosale/nsys/conn3_r3_{ch,zc}/rank_3.nsys-rep. Never sallocs/scancels.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
LCI_LIB=/u/bhosale/lci/build-cuda/lib; export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
BASE="16 8 8 10 99999 99999 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 16384 +gpupoolalloc vmm"
echo "=== conn3 rank3, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
for arm in ch zc; do
  TAG=c3_nsys_r3_${arm}_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/nsys/conn3_r3_$arm; rm -rf $OUT; mkdir -p $OUT
  RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL; LOG=$D/logs/$TAG.log
  C="$BASE"; [ $arm = ch ] && C="$C -channels"
  env PES=8 GPP=1 RANKLOG_DIR=$RL LMD_NSYS_DIR=$OUT LMD_NSYS_RANK=3 timeout 600 $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd_nsys.sh $D/leanmd $C > $LOG 2>&1
  echo "$(date +%T) $TAG: rc=$? steps=$(grep -ac '^Step [0-9]* Benchmark Time' $RL/rank_0.log) | report: $(ls -la $OUT/rank_3.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
done
echo "=== conn3 rank3 done $(date +%T), left $(squeue -j $JID -h -o %L)"
