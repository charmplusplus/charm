#!/bin/bash
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=leanmd-recv-gdb
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test/job-gdb-%j.out
#SBATCH --error=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test/job-gdb-%j.err
# NOT set -e: a failing srun must not abort the script before stacks are collected.
set -u
ulimit -c 0
ROOT=/u/bhosale/charm-reconverse
TEST=$ROOT/examples/charm++/cuda/gpudirect/leanmd/recv-stream-test
export LD_LIBRARY_PATH=$ROOT/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192
unset LEANMD_SHARED_RECV_STREAM
export CHARM_ZC_VALIDATE=1
for rep in 1 2 3; do
  export RANKLOG_DIR=$TEST/gdb-$SLURM_JOB_ID-sync-$rep
  mkdir -p "$RANKLOG_DIR"
  echo "=== sync attempt $rep  $(date +%H:%M:%S)"
  timeout 240 srun --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 \
    --cpus-per-task=16 --cpu-bind=none --exact \
    "$TEST/rankwrap_gdb.sh" "$TEST/leanmd" 8 4 4 30 10 10 -computemap local -density gradient \
    +pe 64 +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 2048 \
    +balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim \
    +LBCostConfig "$ROOT/examples/charm++/cuda/gpudirect/sph2d/lbcost.delta-a40.2node.conf" \
    +LBMetisMaxCutRise 1.0 +LBDebug 2
  rc=$?
  echo "=== attempt $rep rc=$rc"
  if grep -lq "===== FAULT" "$RANKLOG_DIR"/rank_*.log 2>/dev/null; then
    echo "=== FAULT CAPTURED in attempt $rep"
    grep -l "===== FAULT" "$RANKLOG_DIR"/rank_*.log
    break
  fi
  if grep -hq "SIMULATION SUCCESSFULL" "$RANKLOG_DIR"/rank_0.log 2>/dev/null; then
    echo "=== attempt $rep PASSED (no fault)"
  fi
done
echo "=== DONE $(date +%H:%M:%S)"
