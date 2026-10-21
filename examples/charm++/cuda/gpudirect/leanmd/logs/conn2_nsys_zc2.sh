#!/bin/bash
# conn2_nsys_zc2.sh <jobid>: the zero-copy arm of conn2_nsys again, with nsys capped at 115 s (--duration): the first run
# reached Exit on every PE and hung before nsys could write the report (timeout at 420 s). Same args otherwise.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
LCI_LIB=/u/bhosale/lci/build-cuda/lib; export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
TAG=c2_nsys_zc2_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/nsys/conn2_zc; rm -rf $OUT; mkdir -p $OUT
RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL; LOG=$D/logs/$TAG.log
C="16 8 8 30 99999 99999 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 16384 +gpupoolalloc vmm +nvtx"
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
echo "[nsys] $(date +%T) tag=$TAG duration 115 s, args: $C" | tee $LOG
env PES=8 GPP=1 RANKLOG_DIR=$RL LMD_NSYS_DIR=$OUT LMD_NSYS_OPTS="--duration=115" timeout 300 $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd_nsys.sh $D/leanmd $C >> $LOG 2>&1
rc=$?; R=$RL/rank_0.log
echo "$(date +%T) $TAG: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean11-30 $(mean_steps $R 11 30) ms | report: $(ls -la $OUT/rank_0.nsys-rep 2>/dev/null | awk '{print $5}') bytes"
echo "=== conn2 nsys zc2 done $(date +%T), left $(squeue -j $JID -h -o %L)"
