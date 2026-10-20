#!/bin/bash
#SBATCH --job-name=poolbisect
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
# Does the device pool's ARENA SIZE break inter-node transport? At 2 nodes every
# arm of leanmd and sph2d aborted in
#   poll_comp_impl:65 <lci:Assert failed: false> Err 5: Input/output error
# immediately after "device pool: arena (N MB) registered once for RDMA" -- the
# registration that only happens when there is an off-node peer. 1 node was
# clean at the same sizes. The one thing this sweep changed from every previous
# multi-node run is the arena: 256 MB before, 4096 (leanmd) / 32768 (sph2d) now.
#
# 256 MB is the CONTROL: if it also aborts, arena size is innocent and this is a
# multi-node regression to chase elsewhere. leanmd noLB only, ~3 min per size.
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
export DEADLINE_MIN=22
for mb in 256 1024 4096; do
  echo "### arena ${mb} MB"
  LMD_POOL_WEAK=$mb bash $ROOT/scale.sh "$SLURM_JOB_ID" 2 leanmd weak nolb
done
echo "### bisect done $(date +%T)"
