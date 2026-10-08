#!/bin/bash
#SBATCH --job-name=sph3d
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d/logs/hold-%j.out
# Holds the node for interactive arms launched from the login node with
# srun --jobid (run3d.sh / run_slab.sh). Nothing here releases the node early.
echo "node $(hostname) held from $(date +%T)"
sleep 3540
