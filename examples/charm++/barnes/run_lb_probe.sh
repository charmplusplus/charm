#!/bin/bash
cd /u/bhosale/charm-reconverse/examples/charm++/barnes
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=${JOB:?}
R="srun --unbuffered --jobid=$J -n 4 --cpus-per-task=8 --exact ./numa_wrap.sh ./barnes"
B="-in=big.bin -killat=30 -b=128 -p=700 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1 -lbperiod=5"

# What load does the balancer actually see, and how much does it move?
CHARM_LB_LOADDUMP=1 timeout 130 $R $B +p 4 +balancer DiffusionLB +LBDiffusionCommOn \
  +LBDiffusionGpuDim +LBDebug 1 > check/LDgpu.log 2>&1
echo "gpuDim   exit=$? avg=$(grep -oP 'avg time \K[0-9.]+' check/LDgpu.log|tail -1)"

# Same balancer on the host wallTime dimension, to see whether the flag changes anything
CHARM_LB_LOADDUMP=1 timeout 130 $R $B +p 4 +balancer DiffusionLB +LBDiffusionCommOn \
  +LBDebug 1 > check/LDwall.log 2>&1
echo "wallDim  exit=$? avg=$(grep -oP 'avg time \K[0-9.]+' check/LDwall.log|tail -1)"
