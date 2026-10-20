#!/bin/bash
# run_j3d.sh <jobid> <tag>: one jacobi3d run on an allocation, leanmd's process/GPU/PE split and pinning wrapper.
# Env: NODES (2), PPN (4), GPP (1), PES (per process, 4), MODE=zc|ch|reg|persistent, GRID (512), ITERS (200), WARM (10),
# POOL (MB per arena, 8192), TMO (s, 300). One Block per PE. Rank logs: ranklogs/<tag>/.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d
W=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/numa_wrap_lmd.sh
JID=$1; TAG=$2; shift 2
NODES=${NODES:-2}; PPN=${PPN:-4}; NP=$((PPN*NODES)); GPUS=${GPUS:-4}
GPP=${GPP:-$((GPUS/PPN))}; [ "$GPP" -lt 1 ] && GPP=1
PES=${PES:-4}
GRID=${GRID:-512}; ITERS=${ITERS:-200}; WARM=${WARM:-10}; POOL=${POOL:-8192}
case ${MODE:-zc} in zc) MF="-d";; ch) MF="-k";; persistent) MF="-s";; *) MF="";; esac
ulimit -c 0
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${LCI_LIB:-}" ] && export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL $D/logs; LOG=$D/logs/$TAG.log
C="-x $GRID -c $((PES*NP)) -i $ITERS -w $WARM $MF +pe $((PES*NP)) +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize $POOL +gpupoolalloc vmm"
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N $NODES -n $NP --ntasks-per-node=$PPN --gpus-per-node=$GPUS --cpus-per-task=$((64/PPN)) --cpu-bind=none --exact --kill-on-bad-exit=1"
echo "[run_j3d] $(date +%T) tag=$TAG mode=${MODE:-zc} grid=$GRID blocks=$((PES*NP)) args: $C" | tee $LOG
env PES=$PES GPP=$GPP RANKLOG_DIR=$RL timeout ${TMO:-300} $SRUN --chdir=$D $W $GDBW $D/${BIN:-jacobi3d} $C "$@" >> $LOG 2>&1
rc=$?; echo "[run_j3d] rc=$rc $(date +%T)" >> $LOG
exit $rc
