#!/bin/bash
#SBATCH --job-name=pic2d-lb2n
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# pic2d on 2 nodes: noLB vs sync LB vs async LB, balancing with MetisLB first
# and DiffusionLB for every step after it.
#
# Metis-then-Diffusion is what "+balancer MetisLB +balancer DiffusionLB" means
# without +LBLoop: LBManager::nextLoadbalancer advances the sequence once per LB
# step and then clamps to the last entry ("keep using the last one"). So Metis
# runs at the first LB step only -- one global partition off the initial cyclic
# map -- and Diffusion maintains it from there. Verified in the log by
# "Metis partitioning ..." appearing exactly once.
#
# Requires the comm-graph fix (91686ca7d). Before it, every comm-aware balancer
# received an edgeless graph and these numbers meant nothing.

set -u
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d

# Cores on Delta's /u quota are precious and a GPU crash dumps GBs.
ulimit -c 0

# libreconverse.so is not rpath'd into the binary, and the link/runtime both
# need LCI's lib64. Do not rely on the submitting shell's environment.
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
# libfabric's memory-registration monitor deadlocks against CUDA's allocator
# hooks during migration; disabling the monitor took the async-LB hang 4/18 -> 0/20.
export FI_MR_CACHE_MONITOR=disabled
export PMI_MAX_KVS_ENTRIES=1024

# Device memory pool: runtime allocation policy, direct CUDA IPC only.
POOL="+gpushm +gpupool +gpuipceventpool 256"
PPN=8                       # PEs per process -> 8 procs x 8 = 64 PEs

# 2048^2 grid in 128^2 patches = 256 objects over 64 PEs, 4 per PE, so the
# balancer has somewhere to move things. -d 1 -F 0.9 -s 0.1 puts 90% of the
# particles in a narrow bunch, which is the imbalanced case -- and that bunch
# lands ~10.7x the average in one patch, so it needs headroom well above the
# default -r 12 (which gives 4*128*128*12 = 786432 and overflows on the first
# step: "initial particles exceed capacity").
GEOM="-W 2048 -H 2048 -w 128 -h 128 -p 4"
PHYS="-d 1 -F 0.9 -s 0.1 -c 20 -r 32"
ITERS="-i 300 -u 10"

# LB schedule. -f is the FIRST LB iteration and -b the LB PERIOD (the flag
# letters suggest the opposite; the log header prints which is which).
FIRST=50
PERIOD=50
LAG=40                      # async wait lag: 80% of the period, must be < PERIOD

BAL="+balancer MetisLB +balancer DiffusionLB"
LBFLAGS="+LBDiffusionCommOn +LBDiffusionGpuDim +LBCommOn"
# Set DEBUG_LB=1 to get MetisLB's cross-group cut before/after each partition.
[ "${DEBUG_LB:-0}" = "1" ] && LBFLAGS="$LBFLAGS +LBDebug 2"

REPS=${REPS:-3}             # the run-to-run spread here is ~25%; one run proves nothing

# 300 iterations at ~20 ms is ~6 s of compute, so a healthy run finishes well
# inside a minute. The timeout exists to catch a hang, not to wait out a dead
# job: --kill-on-bad-exit tears the step down the moment any rank aborts,
# instead of leaving srun parked until the timeout fires.
run () {                    # run <name> <pic2d-args...>
  local name=$1; shift
  local log="lb2n-${SLURM_JOB_ID}-${name}.log"
  env X=1 timeout 120 srun --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 \
      --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 \
      ./numa_wrap.sh stdbuf -oL -eL ./pic2d $GEOM $PHYS $ITERS "$@" $POOL +ppn $PPN \
      > "$log" 2>&1
  local rc=$?
  echo "### $name rc=$rc avg=$(grep -aoP 'Average iteration time: \K[0-9.]+' "$log")"
  # Read this run's own log now, not a wrapper or squeue.
  grep -aiE "reason:|fatal cuda|double free|corrupt|mismatch|abort" "$log" | head -3
  echo "    metis partitions: $(grep -ac 'Metis partitioning' "$log")  log: $log"
  grep -a "MetisLB locality" "$log" | head -1
}

echo "=== pic2d 2-node LB comparison, job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ==="
echo "=== 8 processes x $PPN PEs = 64 PEs, $GEOM $PHYS $ITERS"
echo "=== first LB $FIRST, period $PERIOD, async lag $LAG, $REPS repetition(s)"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -4

# Cheap viability check first: if the geometry cannot even start, fail here in
# seconds rather than discovering it three aborted runs into the matrix.
echo "----- smoke test (30 iterations, noLB) -----"
env X=1 timeout 120 srun --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 \
    --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 \
    ./numa_wrap.sh stdbuf -oL -eL ./pic2d $GEOM $PHYS -i 30 -u 10 -f 99999 \
    $POOL +ppn $PPN > "lb2n-${SLURM_JOB_ID}-smoke.log" 2>&1
if ! grep -aq "Average iteration time:" "lb2n-${SLURM_JOB_ID}-smoke.log"; then
  echo "SMOKE TEST FAILED -- aborting before the matrix:"
  grep -aiE "reason:|exceed capacity|fatal cuda|srun: error" "lb2n-${SLURM_JOB_ID}-smoke.log" | head -5
  exit 1
fi
echo "smoke ok: avg=$(grep -aoP 'Average iteration time: \K[0-9.]+' lb2n-${SLURM_JOB_ID}-smoke.log)"

for r in $(seq 1 "$REPS"); do
  echo "----- repetition $r/$REPS -----"

  # 1. No load balancing: first LB never reached.
  run "nolb_r$r" -f 99999

  # 2. Synchronous LB: AtSync, the step stops for the migration.
  run "sync_r$r" -f $FIRST -b $PERIOD $BAL $LBFLAGS

  # 3. Asynchronous LB: split barrier, the step continues while objects move.
  #    Needs BOTH +LBAsync (runtime) and -a (application); -a alone is ignored
  #    with a warning, and +LBAsync without -a hangs every migration.
  run "async_r$r" -f $FIRST -b $PERIOD -a -l $LAG $BAL $LBFLAGS +LBAsync
done

echo "=== SUMMARY (ms/iter) ==="
for cfg in nolb sync async; do
  vals=$(grep -h -aoP 'Average iteration time: \K[0-9.]+' lb2n-${SLURM_JOB_ID}-${cfg}_r*.log 2>/dev/null | tr '\n' ' ')
  echo "$(printf '%-6s' $cfg) $vals" | awk '{
    s=0; n=0; mn=1e18; mx=0;
    for (i=2; i<=NF; i++) { s+=$i; n++; if($i<mn)mn=$i; if($i>mx)mx=$i }
    if (n>0) printf "%-6s n=%d  mean=%.3f  min=%.3f  max=%.3f\n", $1, n, s/n, mn, mx;
    else printf "%-6s no completed runs\n", $1
  }'
done
echo "=== DONE ==="
