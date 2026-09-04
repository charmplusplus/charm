#!/bin/bash
#SBATCH --job-name=barnes-asynclb2
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -x
ulimit -c 0
cd /u/bhosale/charm-reconverse/examples/charm++/barnes

export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1

RUN="srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact"
PEFLAGS="+pe 4 +setcpuaffinity"
GPUFLAGS="+gpushm +gpucommbuffer 256 +gpulbbuffer 256 +gpuipceventpool 256"
APP="-in=particles.bin -p=128 -killat=30 -b=128"

echo "########## R0: no LB (oracle) ##########"
timeout 600 $RUN ./barnes $APP $PEFLAGS $GPUFLAGS
echo "=== R0 exit: $? ==="

echo "########## R1: sync LB (AtSync) ##########"
timeout 600 $RUN ./barnes $APP -lbperiod=5 -firstlb=2 $PEFLAGS $GPUFLAGS \
  +balancer GreedyRefineCentralGPULB
echo "=== R1 exit: $? ==="

echo "########## R2: async LB (+LBAsync, split barrier) ##########"
timeout 600 $RUN ./barnes $APP -lbperiod=5 -firstlb=2 -lbasync=1 $PEFLAGS $GPUFLAGS \
  +balancer GreedyRefineCentralGPULB +LBAsync
echo "=== R2 exit: $? ==="

echo "########## R3: async LB again (repeat, nondeterminism check) ##########"
timeout 600 $RUN ./barnes $APP -lbperiod=5 -firstlb=2 -lbasync=1 $PEFLAGS $GPUFLAGS \
  +balancer GreedyRefineCentralGPULB +LBAsync
echo "=== R3 exit: $? ==="

echo "ALL DONE"
