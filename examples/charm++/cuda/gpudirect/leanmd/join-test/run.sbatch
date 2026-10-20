#!/bin/bash
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=leanmd-join
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --output=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/join-test/job-%j.out
#SBATCH --error=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/join-test/job-%j.err
ulimit -c 0
export LEANMD_JOIN_TRACE=1 DEADLINE_MIN=10
exec bash /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/join-test/scale.sh "$SLURM_JOB_ID" 2 leanmd weak "nolb sync"
