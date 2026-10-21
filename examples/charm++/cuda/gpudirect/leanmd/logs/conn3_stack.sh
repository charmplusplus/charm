#!/bin/bash
# conn3_stack.sh <jobid>: one more zero-copy run under nsys (no duration, no kill), only to take the APPLICATION's
# stack in its hung window after "Exit called" (c3_appstack.sh matches the process by name; the chain's capture
# matched nsys). Waits for conn3_oneflag. Never sallocs, never scancels.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
until grep -q 'conn3 oneflag done\|no leanmd.oneflag' $D/logs/conn3_oneflag.out 2>/dev/null; do sleep 10; done
echo "=== conn3 stack run, job $JID, $(date +%T), left $(squeue -j $JID -h -o %L)"
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
LCI_LIB=/u/bhosale/lci/build-cuda/lib; export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
TAG=c3_nsys_stack_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/nsys/conn3_stack; rm -rf $OUT; mkdir -p $OUT
RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL; LOG=$D/logs/$TAG.log
setsid -f nohup bash $D/logs/c3_appstack.sh $JID c3_nsys_stack < /dev/null > $D/logs/c3_nsys_stack_appstack.txt 2>&1
C="16 8 8 30 99999 99999 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 16384 +gpupoolalloc vmm"
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
env PES=8 GPP=1 RANKLOG_DIR=$RL LMD_NSYS_DIR=$OUT timeout 240 $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd_nsys.sh $D/leanmd $C > $LOG 2>&1
echo "$(date +%T) $TAG: rc=$? steps=$(grep -ac '^Step [0-9]* Benchmark Time' $RL/rank_0.log) | stack file: $(grep -c '^Thread' $D/logs/c3_nsys_stack_appstack.txt) threads"
echo "=== conn3 stack run done $(date +%T), left $(squeue -j $JID -h -o %L)"
