#!/bin/bash
#SBATCH --job-name=poolbisect2
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Round 1 (job 22079951, 2 nodes, leanmd noLB): 256 MB ok (6 arenas), 1024 MB ok
# (2 arenas), 4096 MB aborts in poll_comp_impl Err 5 right after the arena is
# registered for RDMA. The cliff is between 1 and 4 GB.
#
# Two questions left:
#   1. does 2048 register?  (the smallest single arena that covers the 1.75 GB
#      peak measured WITH load balancing at N=1)
#   2. does the winning size hold through the LB steps?  noLB never migrates,
#      and migration payloads are what took the 7th arena mid-run at N=1 --
#      so the sync arm, not noLB, is the real test of "no growth in the window".
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
export DEADLINE_MIN=22
echo "### 2048 MB, noLB -- does it register?"
LMD_POOL_WEAK=2048 bash $ROOT/scale.sh "$SLURM_JOB_ID" 2 leanmd weak nolb
echo "### 2048 MB, sync -- does it hold through the migrations?"
LMD_POOL_WEAK=2048 bash $ROOT/scale.sh "$SLURM_JOB_ID" 2 leanmd weak sync
echo "### 1024 MB, sync -- the fallback if 2048 does not register"
LMD_POOL_WEAK=1024 bash $ROOT/scale.sh "$SLURM_JOB_ID" 2 leanmd weak sync
echo "### done $(date +%T)"
