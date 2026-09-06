#!/bin/bash
# Run one phase-02 step inside an existing interactive allocation.
#   usage: JOBID=<id> ./run_step.sh <step>
#   steps: 0=device gate  1=no-LB reference x3  2=sync  3=async lags  4=metis+diffusion
set -u
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance

export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
export IPATH_NO_BACKTRACE=1
# Deliberately NOT set: CHARM_METALB_GPU_FAKE_DEVICES.

RUN="srun --jobid=${JOBID} --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact"
PEFLAGS="+pe 32 +setcpuaffinity"
GPUFLAGS="+gpushm +gpucommbuffer 256 +gpulbbuffer 256 +gpuipceventpool 256"
GEOM="-W 8192 -H 8192 -w 1024 -h 1024 -i 200 -u 10 -m 8"

case "${1}" in
0)
  echo "### STEP 0: device identity gate (short run)"
  $RUN ./jacobi2d -W 8192 -H 8192 -w 1024 -h 1024 -i 40 -u 5 -m 8 -M -s 5 \
    $PEFLAGS $GPUFLAGS +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
  echo "### STEP 0 exit: $?"
  ;;
1)
  for rep in 1 2 3; do
    echo "### STEP 1 reference rep $rep"
    $RUN ./jacobi2d $GEOM -b 100000 $PEFLAGS $GPUFLAGS +balancer MetisLB
    echo "### rep $rep exit: $?"
  done
  ;;
2)
  echo "### STEP 2: GPU-triggered, synchronous"
  $RUN ./jacobi2d $GEOM -M -s 5 $PEFLAGS $GPUFLAGS \
    +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
  echo "### STEP 2 exit: $?"
  ;;
3)
  for lag in 0 3 9; do
    echo "### STEP 3: async, lag $lag"
    $RUN ./jacobi2d $GEOM -A -s 5 -l $lag $PEFLAGS $GPUFLAGS \
      +LBAsync +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
    echo "### lag $lag exit: $?"
  done
  ;;
4)
  echo "### STEP 4: Metis first, Diffusion thereafter"
  $RUN ./jacobi2d $GEOM -A -s 5 -l 3 $PEFLAGS $GPUFLAGS +LBAsync \
    +balancer MetisLB +balancer DiffusionLB +LBDiffusionGpuDim \
    +MetaLB +MetaLBGpuTrigger +LBDebug 1
  echo "### STEP 4 exit: $?"
  ;;
*) echo "unknown step ${1}"; exit 2;;
esac
