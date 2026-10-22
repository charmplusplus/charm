#!/bin/bash
# run3d.sh <jobid> <tag> [sph3d args...]   -- default: one node, 4 processes x 4 PEs, one GPU each.
# Split knobs: PPN (processes per node, P, default 4), GPP (GPUs per process, G, default GPUS/PPN), PES (PEs per process = G x K, default 4 x GPP).
#   PPN=2 GPP=2 PES=8 is two processes of two GPUs with 4 PEs per GPU; PPN=1 GPP=4 PES=16 is one process driving all four.
# Env: BIN (binary in this dir, default ./sph3d), NODES (default 1), NP (processes, default 4 per node), GPUS (devices per node the step sees, default 4; NODES=1 NP=1 GPUS=1 is a true single-GPU run),
#      NODELIST (-w), POOL_ALLOC=vmm (+gpupoolalloc vmm, required off-node), LCI_LIB (dir of a liblci.so to preload: the vmm device-lookup fix), POOL (MB, power of two, default 8192), POOLX (extra pool flags), TMO (s), LB=sync|async|none
# (default none; sync/async add Metis+Diffusion with sph2d's calibrated cost table; LBF/LBB/LBLAG).
JID=$1; TAG=$2; shift 2
DIR=${SPH_DIR:-/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d}
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ulimit -c 0
POOL=${POOL:-8192}
MD="+LBDebug ${LBDEBUG:-1} ${XFLAGS:-}"   # 2026-10-18: the runtime picks the strategy, cost table and floors; XFLAGS only for experiments
case ${LB:-none} in
  none)  lbargs="-f 99999";;
  sync)  lbargs="-f ${LBF:-25} -b ${LBB:-25} -A $MD";;
  async) lbargs="-f ${LBF:-25} -b ${LBB:-25} -a -l ${LBLAG:-10} $MD";;   # period 25, lag 10 = the measured best (2026-10-09)
esac
NODES=${NODES:-1}
PPN=${PPN:-4}
NP=${NP:-$((PPN*NODES))}
PPN=$((NP/NODES))
GPP=${GPP:-$(( ${GPUS:-4} / PPN ))}; [ "$GPP" -lt 1 ] && GPP=1
PES=${PES:-$((4*GPP))}
GPUMAP=""; [ "$GPP" -gt 1 ] && GPUMAP="+gpudevs $GPP +gpumap block${LCIDEV:+ +lci_ndevices $GPP}"   # several GPUs per process: contiguous PE blocks per device; LCIDEV=1 adds one LCI device per GPU group (needs CMK_NOCOPY_DIRECT_BYTES >= 32 x GPP: the RDMA descriptor carries one rmr per LCI device)
POOLX="${POOLX:-}"; [ "${POOL_ALLOC:-buddy}" = vmm ] && POOLX="$POOLX +gpupoolalloc vmm"
if [ -n "$LCI_LIB" ]; then export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH; fi
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N $NODES -n $NP --ntasks-per-node=$PPN ${NODELIST:+-w $NODELIST} --gpus-per-node=${GPUS:-4} --cpus-per-task=$((64/PPN)) --cpu-bind=none --exact --kill-on-bad-exit=1"
LOG=$DIR/logs/$TAG.log
echo "[run3d] $(date +%T) tag=$TAG nodes=$NODES np=$NP split=P${PPN}xG${GPP}xK$((PES/GPP)) ppn=$PES pool=$POOL alloc=${POOL_ALLOC:-buddy} lci=${LCI_LIB:-install} LB=${LB:-none} args: $* $lbargs" | tee $LOG
env X=1 PES=$PES GPP=$GPP timeout ${TMO:-900} $SRUN --chdir=$DIR stdbuf -oL -eL ${SPH_WRAP:-$DIR/numa_wrap_sph.sh} ${BIN:-./sph3d} "$@" $lbargs \
  +gpushm +gpupool +gpupoolsize $POOL ${POOLX:-} +gpuipceventpool ${IPCEV:-256} +ppn $PES $GPUMAP >> $LOG 2>&1
RC=$?
# srun's status is lost behind stdbuf/numa_wrap in some failures: a step that was cancelled for a task
# failure, or never printed its final average, is a failure whatever the pipeline said (job 22768193).
if grep -q "CANCELLED AT .* DUE TO TASK FAILURE\|Segmentation fault\|Fatal error\|CmiAbort" $LOG || ! grep -q "Average iteration time" $LOG; then [ "$RC" = 0 ] && RC=99; fi
echo "[run3d] $(date +%T) rc=$RC tag=$TAG" | tee -a $LOG
grep -E "SPH 3D|column|lattice|Init:|step +[0-9]+:|CORRUPTION|Abort|abort|WARN|LB at step|Average iteration|rc=|Fatal|error" $LOG | tail -30
