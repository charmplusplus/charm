#!/bin/bash
# run_lmd.sh <jobid> <tag> [extra leanmd args]: one leanmd run on an allocation with the process/GPU/PE split knobs of
# sph3d's run3d.sh. Env: NODES (2), PPN (processes per node, 4), GPP (GPUs per process, GPUS/PPN), PES (PEs per process,
# 8*GPP), LCIDEV=1 (+lci_ndevices GPP), LB=none|sync|async, GRID ("16 8 8"), STEPS (100), PERIOD (20), LAG (16), POOL (MB
# per arena, 16384), POOL_ALLOC (vmm|buddy), TMO (s, 600), LBDEBUG (1), LMD_EXTRA (extra LB flags). PPN=1 GPP=4 PES=32 is
# one process driving all four GPUs with 8 PEs each; PPN=4 GPP=1 PES=8 is the control. Rank logs: ranklogs/<tag>/.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
JID=$1; TAG=$2; shift 2
NODES=${NODES:-2}; PPN=${PPN:-4}; NP=$((PPN*NODES)); GPUS=${GPUS:-4}
GPP=${GPP:-$((GPUS/PPN))}; [ "$GPP" -lt 1 ] && GPP=1
PES=${PES:-$((8*GPP))}
GPUMAP=""; [ "$GPP" -gt 1 ] && GPUMAP="+gpudevs $GPP +gpumap block${LCIDEV:+ +lci_ndevices $GPP}"
GRID=${GRID:-16 8 8}; STEPS=${STEPS:-100}; PERIOD=${PERIOD:-20}; LAG=${LAG:-16}; POOL=${POOL:-16384}
LBARGS=""
case ${LB:-none} in
  sync)  LBARGS="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug ${LBDEBUG:-1} ${LMD_EXTRA:-}";;
  async) LBARGS="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim -lbasync -lblag $LAG +LBAsync +LBDebug ${LBDEBUG:-1} ${LMD_EXTRA:-}";;
esac
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
# As sph3d/run3d.sh: cray PMI's KVS defaults to 30 entries (a 2-node launch aborts in _pmi2_add_kvs), the libfabric
# memhooks monitor deadlocks with CUDA, and LCI_LIB is PRELOADED (the vmm device-lookup fix: the stock LCI pushes a NULL
# owning context for vmm buffers -- cuCtxPushCurrent fails in accelerator_cuda.cpp:154, job 22953717 first arms).
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
[ -n "${LCI_LIB:-}" ] && export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH
RL=$D/ranklogs/$TAG; rm -rf $RL; mkdir -p $RL $D/logs; LOG=$D/logs/$TAG.log
C="$GRID $STEPS $PERIOD $PERIOD -computemap local -density gradient +pe $((PES*NP)) +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize $POOL"
[ "${POOL_ALLOC:-buddy}" = vmm ] && C="$C +gpupoolalloc vmm"
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N $NODES -n $NP --ntasks-per-node=$PPN --gpus-per-node=$GPUS --cpus-per-task=$((64/PPN)) --cpu-bind=none --exact --kill-on-bad-exit=1"
echo "[run_lmd] $(date +%T) tag=$TAG nodes=$NODES np=$NP split=P${PPN}xG${GPP}xK$((PES/GPP)) pes/proc=$PES pool=$POOL alloc=${POOL_ALLOC:-buddy} LB=${LB:-none} args: $C $LBARGS $* $GPUMAP" | tee $LOG
env PES=$PES GPP=$GPP RANKLOG_DIR=$RL timeout ${TMO:-600} $SRUN --chdir=$D stdbuf -oL -eL $D/numa_wrap_lmd.sh $D/leanmd $C $LBARGS "$@" $GPUMAP >> $LOG 2>&1
rc=$?; echo "[run_lmd] rc=$rc $(date +%T)" >> $LOG
exit $rc
