#!/bin/bash
#SBATCH --job-name=scale
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
bash /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/diffvar.sh "$SLURM_JOB_ID" 2 5
