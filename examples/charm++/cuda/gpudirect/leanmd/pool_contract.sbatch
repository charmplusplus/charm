#!/bin/bash
#SBATCH --job-name=leanmd-poolcontract
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# The Pool Contract on four A40s: the multi-device checks of the memory-aware
# balancing work (MetisLB, DiffusionLB, ShedLB under +gpupool) that a one-GPU
# machine cannot run. MIGRATION_MEMORY_FIX.md's reproduce, at the slot pool
# that used to deadlock.
#
# Needs the whole Charm tree rebuilt at or after cdb754540 (ProcStats and the
# CkLocMgr interface changed) and leanmd relinked from clean objects.
#
# What to look for in the .out:
#   CharmLB> memory contract: executing this step's N migrations in K batches
#   [memcontract] device D (...): T ..., sigma_max ..., N object(s) holding ...
#   memory contract: a migration landing waits for device memory   (gate)
#   [node N] memory: T ... advertises ... stages up to ...           (DiffusionLB)
#   "no free CUDA IPC event slot" or "out of memory"                 (must not appear)

set -x
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd

# libreconverse.so (built at multicore-linux-x86_64-cuda/lib/) is not rpath'd
# into the leanmd binary, so it must be on LD_LIBRARY_PATH explicitly -- do not
# rely on the submitting shell's inherited env having this set.
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"

echo "=== TEST 1: MetisLB then DiffusionLB, gradient, +gpuipceventpool 256 (previously the slot deadlock) ==="
CHARM_DEBUG_MEMCONTRACT=1 timeout 480 \
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 8 8 8 100 20 20 -computemap local -density gradient \
  +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool \
  +balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 1
echo "=== TEST 1 exit code: $? ==="

echo "=== TEST 2: DiffusionLB alone, clump (advertise and plan across four devices) ==="
timeout 480 \
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 8 8 8 100 20 20 -computemap local -density clump \
  +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool \
  +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 2
echo "=== TEST 2 exit code: $? ==="

echo "=== TEST 3: ShedLB, gradient (the ledger in its placement search) ==="
CHARM_DEBUG_MEMCONTRACT=1 timeout 480 \
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 8 8 8 100 20 20 -computemap local -density gradient \
  +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool \
  +balancer ShedLB +LBDebug 1
echo "=== TEST 3 exit code: $? ==="

echo "=== ALL DONE ==="
