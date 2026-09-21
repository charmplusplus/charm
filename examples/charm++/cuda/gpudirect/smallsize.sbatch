#!/bin/bash
#SBATCH --job-name=smallsize
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Are the SMALLER problem sizes GPU-bound at CUDA_DEVICE_MAX_CONNECTIONS=32?
# One node, nolb + sync per size; the verdict is the balancer's own line
# "measured loads explain host H%, device D%, launch L% of the interval"
# (busiest PE / GPU / process over the LB window) plus nolb vs sync ms/step.
# conn=32 first (the question), conn=8 after it (the control, same sizes).
ROOT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
T0=$(date +%s); LIMIT_MIN=42
left() { echo $(( LIMIT_MIN - ( $(date +%s) - T0 ) / 60 )); }
export LCI_LIB=/u/bhosale/lci/build-cuda/lib POOL_ALLOC=vmm LMD_EXTRA='+LBMetisMaxCutRise 1.0' CHARM_LB_ASYNC_NO_PEER_DECISION=1
echo "=== smallsize job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
for conn in 32 8; do
  export CONNECTIONS=$conn
  for grid in "4 4 4" "8 4 4" "8 8 4"; do
    TAG=small_c${conn}_g${grid// /x} LMD_GRID="$grid" STEPS=60 PERIOD=20 DEADLINE_MIN=$(left) \
      bash $ROOT/scale.sh "$SLURM_JOB_ID" 1 leanmd strong "nolb sync"
  done
  for s in 0.0012 0.00084 0.0006 0.00042; do
    TAG=small_c${conn}_s$s SPH_WEAK_S=$s SPH_WEAK_I=3000 SPH_WEAK_F=500 SPH_WEAK_B=1000 SPH_WEAK_REPORT=500 DEADLINE_MIN=$(left) \
      bash $ROOT/scale.sh "$SLURM_JOB_ID" 1 sph2d weak "nolb sync"
  done
done
echo "=== VERDICT LINES (sync arms)"
for d in $ROOT/leanmd/ranklogs/small_c*_strong_N1_sync; do echo "-- $(basename $d)"; grep -ah "measured loads explain\|load dimension:" $d/rank_0.log | cut -c1-230; done
for f in $ROOT/sph2d/small_c*_WEAK_N1_sync.log; do echo "-- $(basename $f)"; grep -ah "measured loads explain\|load dimension:" $f | cut -c1-230; done
