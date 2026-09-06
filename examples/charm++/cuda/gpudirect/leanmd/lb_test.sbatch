#!/bin/bash
#SBATCH --job-name=leanmd-lbtest
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -x
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd

# libreconverse.so (built at multicore-linux-x86_64-cuda/lib/) is not rpath'd
# into the leanmd binary, so it must be on LD_LIBRARY_PATH explicitly -- do not
# rely on the submitting shell's inherited env having this set.
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"

echo "=== BASELINE: no load balancer (for comparison) ==="
echo "reference from job 21342954: total application time 21.929942 s, drift 5.651E-17"
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 6 6 6 100 20 20 +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpucommbuffer 256
echo "=== BASELINE exit code: $? ==="

echo "=== LB TEST: MetisLB first, DiffusionLB after (same problem size) ==="
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 6 6 6 100 20 20 +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpucommbuffer 256 \
  +balancer MetisLB +balancer DiffusionLB +LBDebug 1
echo "=== LB TEST exit code: $? ==="

echo "=== ALL DONE ==="
