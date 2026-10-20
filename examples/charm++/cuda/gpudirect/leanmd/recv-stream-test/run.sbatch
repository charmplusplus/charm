#!/bin/bash
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=leanmd-recv
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --output=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test/job-%j.out
#SBATCH --error=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test/job-%j.err
set -eu
ulimit -c 0
ROOT=/u/bhosale/charm-reconverse
TEST=$ROOT/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test
export LD_LIBRARY_PATH=$ROOT/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192
unset LEANMD_SHARED_RECV_STREAM
export CHARM_ZC_VALIDATE=1
for arm in nolb sync async; do
  export RANKLOG_DIR=$TEST/$SLURM_JOB_ID-$arm
  mkdir -p "$RANKLOG_DIR"
  args=()
  if [ "$arm" != nolb ]; then
    args=(+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim
      +LBCostConfig "$ROOT/examples/charm++/cuda/gpudirect/sph2d/lbcost.delta-a40.2node.conf"
      +LBMetisMaxCutRise 1.0 +LBDebug 2)
  fi
  if [ "$arm" = async ]; then args+=(-lbasync -lblag 4 +LBAsync); fi
  echo "=== $arm ==="
  timeout 85 srun --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 \
    "$TEST/rankwrap.sh" "$TEST/leanmd" 8 4 4 30 10 10 -computemap local -density gradient \
    +pe 64 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 2048 "${args[@]}"
done
