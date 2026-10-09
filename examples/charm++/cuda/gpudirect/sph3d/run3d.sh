#!/bin/bash
# run3d.sh <jobid> <tag> [sph3d args...]   -- one node, 4 processes x 4 PEs, one GPU each.
# Env: NODES (default 1), NP (processes, default 4 per node), GPUS (devices per node the step sees, default 4; NODES=1 NP=1 GPUS=1 is a true single-GPU run),
#      NODELIST (-w), POOL_ALLOC=vmm (+gpupoolalloc vmm, required off-node), LCI_LIB (dir of a liblci.so to preload: the vmm device-lookup fix), POOL (MB, power of two, default 8192), POOLX (extra pool flags), TMO (s), LB=sync|async|none
# (default none; sync/async add Metis+Diffusion with sph2d's calibrated cost table; LBF/LBB/LBLAG).
JID=$1; TAG=$2; shift 2
DIR=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ulimit -c 0
POOL=${POOL:-8192}
COST=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph2d/lbcost.delta-a40.conf
MD="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBCostConfig $COST +LBDebug ${LBDEBUG:-1}"
case ${LB:-none} in
  none)  lbargs="-f 99999";;
  sync)  lbargs="-f ${LBF:-100} -b ${LBB:-100} $MD";;
  async) lbargs="-f ${LBF:-100} -b ${LBB:-100} -a -l ${LBLAG:-50} $MD +LBAsync";;
esac
NODES=${NODES:-1}
NP=${NP:-$((4*NODES))}
PPN=$((NP/NODES))
POOLX="${POOLX:-}"; [ "${POOL_ALLOC:-buddy}" = vmm ] && POOLX="$POOLX +gpupoolalloc vmm"
if [ -n "$LCI_LIB" ]; then export LD_PRELOAD=$LCI_LIB/liblci.so LD_LIBRARY_PATH=$LCI_LIB:$LD_LIBRARY_PATH; fi
SRUN="srun --jobid=$JID --overlap --mpi=cray_shasta -N $NODES -n $NP --ntasks-per-node=$PPN ${NODELIST:+-w $NODELIST} --gpus-per-node=${GPUS:-4} --cpus-per-task=$((64/PPN)) --cpu-bind=none --exact --kill-on-bad-exit=1"
LOG=$DIR/logs/$TAG.log
echo "[run3d] $(date +%T) tag=$TAG nodes=$NODES np=$NP pool=$POOL alloc=${POOL_ALLOC:-buddy} lci=${LCI_LIB:-install} LB=${LB:-none} args: $* $lbargs" | tee $LOG
env X=1 PES=4 timeout ${TMO:-900} $SRUN --chdir=$DIR stdbuf -oL -eL $DIR/numa_wrap_sph.sh ./sph3d "$@" $lbargs \
  +gpushm +gpupool +gpupoolsize $POOL ${POOLX:-} +gpuipceventpool 256 +ppn 4 >> $LOG 2>&1
echo "[run3d] $(date +%T) rc=$? tag=$TAG" | tee -a $LOG
grep -E "SPH 3D|column|lattice|Init:|step +[0-9]+:|CORRUPTION|Abort|abort|WARN|LB at step|Average iteration|rc=|Fatal|error" $LOG | tail -30
