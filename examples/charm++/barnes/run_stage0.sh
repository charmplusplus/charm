#!/bin/bash
# Stage 0 of GPU_RESIDENT_PLAN.md: find out where the iteration time goes.
# Run from inside an salloc. Every run is no-LB: the question here is the
# baseline pipeline, not the balancer.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes
ulimit -c 0

export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1
export FI_MR_CACHE_MONITOR=disabled
export BARNES_PHASE_REPORT=1
export BARNES_WALK_REPORT=1

RUN="srun --jobid=${JOB:?} --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact"
PEFLAGS="+pe 4 +setcpuaffinity"
GPUFLAGS="+gpushm +gpucommbuffer 256 +gpulbbuffer 256 +gpuipceventpool 256"

echo "########## S0-A: 50K particles, p=128 (the configuration all prior numbers used) ##########"
timeout 600 $RUN ./barnes -in=particles.bin -p=128 -killat=30 -b=128 $PEFLAGS $GPUFLAGS
echo "=== S0-A exit: $? ==="

echo "########## S0-B: 500K particles, p=128 (10x, to see what scales) ##########"
timeout 900 $RUN ./barnes -in=big.bin -p=128 -killat=30 -b=128 $PEFLAGS $GPUFLAGS
echo "=== S0-B exit: $? ==="

echo "ALL DONE"
