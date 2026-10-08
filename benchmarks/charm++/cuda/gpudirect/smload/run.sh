#!/bin/bash
# smload arms on an existing A40 allocation (one node, 4 processes, one GPU
# each). NEVER cancels the allocation.
#   run.sh <jobid> iso     <sms>           1 block per GPU (+ppn 1): isolated kernels
#   run.sh <jobid> conc    <sms>           8 blocks per GPU (+ppn 8): concurrent kernels
#   run.sh <jobid> mixed   <lo> <hi>       8 per GPU, checkerboard of lo/hi, pinned
#   run.sh <jobid> balance <lo> <hi> [+gpuloadbusy]   same, blocks free to move
# Env: ITERS (60) FIRST (10) PERIOD (10) WORK (16) TOL (0.10) EXTRA (more flags)
#      OBJDUMP=1 adds the runtime's per-object line (CHARM_GPU_LOAD_OBJDUMP).
set -u
D=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/smload
WRAP=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d/numa_wrap.sh
JOB=${1:?jobid}; ARM=${2:?arm}; shift 2
mkdir -p $D/logs
export LD_LIBRARY_PATH=/u/bhosale/lci_install/lib64:/u/bhosale/charm-reconverse/lib:${LD_LIBRARY_PATH:-}
export CUDA_DEVICE_ORDER=PCI_BUS_ID FI_MR_CACHE_MONITOR=disabled MODE=match
[ "${OBJDUMP:-0}" = "1" ] && export CHARM_GPU_LOAD_OBJDUMP=1
ulimit -c 0

ITERS=${ITERS:-60}; FIRST=${FIRST:-10}; PERIOD=${PERIOD:-10}; WORK=${WORK:-16}; TOL=${TOL:-0.10}
COMMON="-i $ITERS -u 5 -f $FIRST -b $PERIOD -m $WORK -t $TOL"
LB="+balancer DiffusionLB +LBDiffusionGpuDim"
POOL="+gpushm +gpupool +gpupoolsize 1024 +gpuipceventpool 256"

case "$ARM" in
  iso)     SMS=${1:?sms}; PPN=1; GEOM="-W 4096 -H 1024 -w 1024 -h 1024"; APP="-s $SMS -F";;
  conc)    SMS=${1:?sms}; PPN=8; GEOM="-W 8192 -H 4096 -w 1024 -h 1024"; APP="-s $SMS -F";;
  mixed)   LO=${1:?lo}; HI=${2:?hi}; PPN=8; GEOM="-W 8192 -H 4096 -w 1024 -h 1024"; APP="-s $LO:$HI -P checker -F";;
  balance) LO=${1:?lo}; HI=${2:?hi}; shift 2; PPN=8; GEOM="-W 8192 -H 4096 -w 1024 -h 1024"; APP="-s $LO:$HI -P checker $*";;
  *) echo "unknown arm $ARM"; exit 2;;
esac
export PES_PER_PROC=$PPN
TAG="${ARM}_$(echo "$APP" | tr -d ' :+-')_$(date +%m%d_%H%M%S)"
LOG=$D/logs/$TAG.log
echo "=== $TAG"
timeout ${TMO:-600} srun --jobid="$JOB" --unbuffered -N 1 -n 4 --ntasks-per-node=4 \
     --cpus-per-task=8 --cpu-bind=none --exact \
     "$WRAP" "$D/smload" $GEOM $COMMON $APP ${EXTRA:-} +ppn $PPN $LB $POOL > "$LOG" 2>&1
rc=$?
echo "rc=$rc  log=$LOG"
grep -E "^\[smload\] (SMs|device|worst|LB step|moves|checksum|total)|^SMLOAD" "$LOG" | sed 's/^/    /'
[ $rc -ne 0 ] && { echo "!!! FAILED, tail:"; tail -15 "$LOG"; }
exit $rc
