#!/bin/bash
#SBATCH --job-name=leanmd-verify
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

echo "=== TEST 1: single-process correctness oracle ==="
echo "expect: Energy: initial 5.9198343288E-15  final 5.8662386214E-15  drift 5.360E-17"
# Must go through srun -n 1: the batch step inherits SLURM_NPROCS/SLURM_NTASKS=4
# from the job's --ntasks=4 request, and the reconverse network layer reads
# those directly, so a bare (non-srun) launch here makes it think it's rank 0
# of 4 and hang forever waiting for 3 peers that never launch. srun -n 1 gets
# a step with its own correct SLURM_NPROCS=1/SLURM_PROCID=0.
srun -n 1 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd +p4 3 3 3 60 1000 1000
echo "=== TEST 1 exit code: $? ==="

echo "=== TEST 2: small bisect case (previously segfaulted, dev_idx=-1 ev_idx=-1) ==="
srun -n 4 --cpus-per-task=2 --cpu-bind=cores --exact \
  ./leanmd 3 3 3 5 1000 1000 +pe 8 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpucommbuffer 256
echo "=== TEST 2 exit code: $? ==="

echo "=== TEST 3: full original repro (previously hit CUDA error 400 at ckrdmadevice.C:463) ==="
srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact \
  ./leanmd 6 6 6 100 20 20 +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpucommbuffer 256
echo "=== TEST 3 exit code: $? ==="

echo "=== ALL TESTS DONE ==="
