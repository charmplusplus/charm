#!/bin/bash
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=leanmd-grow
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:07:00
#SBATCH --output=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/grow-test/job-%j.out
#SBATCH --error=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/grow-test/job-%j.err
set -eu
ulimit -c 0
export LEANMD_JOIN_TRACE=1 DEADLINE_MIN=7 LBDEBUG=2
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/grow-test
unset CHARM_DIFFUSION_GROW_ANY
export GROW_CASE=baseline
echo '=== matched baseline: frontier restriction enabled ==='
bash "$ROOT/scale.sh" "$SLURM_JOB_ID" 2 leanmd weak sync
export CHARM_DIFFUSION_GROW_ANY=1 GROW_CASE=grow-any
echo '=== grow-any: frontier restriction disabled, costs unchanged ==='
bash "$ROOT/scale.sh" "$SLURM_JOB_ID" 2 leanmd weak sync
