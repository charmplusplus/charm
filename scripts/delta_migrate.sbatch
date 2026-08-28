#!/bin/bash
#SBATCH --job-name=migrate-validate
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=migrate-validate-%j.out

# Cross-node validation of the staged migration rework (bounded-migration
# workstream): the non-blocking sender + finishGPUSend ack, the aligned
# device layout, the per-buffer manifest, and the arena receive -- all on the
# path this machine can run and the dev laptop cannot: two nodes, so every
# RotateLB migration is inter-node staged with a real network transfer.
#
# Build first, then submit from the repository root:
#   cd multicore-linux-x86_64-cuda && make && cd ..
#   cd examples/charm++/cuda/gpudirect/jacobi2d && touch jacobi2d.C && make OPTS="-L/u/bhosale/lci_install/lib64 -llct" && cd ../../../../..
#   cd tests/charm++/load_balancing/migrate_device_manifest && make OPTS="-O2 -L/u/bhosale/lci_install/lib64 -llct" && cd ../../../..
#   sbatch scripts/delta_migrate.sbatch
#
# Reading the result: the summary block at the end prints PASS/FAIL per
# section. Sections 4 and 5 pass by ABORTING with the right message.

set -u

REPO=$SLURM_SUBMIT_DIR
JAC=$REPO/examples/charm++/cuda/gpudirect/jacobi2d/jacobi2d
MIG=$REPO/tests/charm++/load_balancing/migrate_device_manifest/migmanifest

export LD_LIBRARY_PATH="$REPO/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
# libinfinipath installs a SIGSEGV handler that swallows real crash reports.
export IPATH_NO_BACKTRACE=1
export CHARM_ZC_VALIDATE=1

# Required for any device migration: the staged pack block comes from the
# +gpulbbuffer pool. Event pool sized per the 2026-08-17 working-state notes.
FLAGS="+ppn 1 +gpushm +gpucommbuffer 512 +gpulbbuffer 128 +gpuipceventpool 256"

JAC_ARGS="-W 4096 -H 4096 -w 1024 -h 1024 -i 30 -u 5 +balancer RotateLB $FLAGS"
MIG_ARGS="+balancer RotateLB $FLAGS"

OUT=migrate-validate-$SLURM_JOB_ID

echo "=== 1. jacobi2d cross-node staged, ARENA REBIND ==="
CHARM_MIGRATE_ARENA=1 srun -N2 -n2 $JAC $JAC_ARGS > $OUT.1.log 2>&1
echo "exit=$?"

echo "=== 2. jacobi2d cross-node staged, compat (no arena) ==="
srun -N2 -n2 $JAC $JAC_ARGS > $OUT.2.log 2>&1
echo "exit=$?"

echo "=== 3. migmanifest clean, arena ==="
CHARM_MIGRATE_ARENA=1 srun -N2 -n2 $MIG $MIG_ARGS > $OUT.3.log 2>&1
echo "exit=$?"

echo "=== 4. migmanifest wrong-size injection (must abort) ==="
CHARM_MIGRATE_ARENA=1 MIGTEST_INJECT_MISMATCH=1 srun -N2 -n2 $MIG $MIG_ARGS > $OUT.4.log 2>&1
echo "exit=$?"

echo "=== 5. migmanifest skipped-buffer injection (must abort) ==="
CHARM_MIGRATE_ARENA=1 MIGTEST_INJECT_MISSING=1 srun -N2 -n2 $MIG $MIG_ARGS > $OUT.5.log 2>&1
echo "exit=$?"

echo ""
echo "================ SUMMARY ================"
s1_done=$(grep -c "Average iteration time" $OUT.1.log || true)
s1_bad=$(grep -c "VALIDATE FAIL\|direction mismatch" $OUT.1.log || true)
[ "$s1_done" -ge 1 ] && [ "$s1_bad" -eq 0 ] && echo "1 arena rebind:      PASS" || echo "1 arena rebind:      FAIL (done=$s1_done bad=$s1_bad)"

s2_done=$(grep -c "Average iteration time" $OUT.2.log || true)
s2_bad=$(grep -c "VALIDATE FAIL\|direction mismatch" $OUT.2.log || true)
[ "$s2_done" -ge 1 ] && [ "$s2_bad" -eq 0 ] && echo "2 staged compat:     PASS" || echo "2 staged compat:     FAIL (done=$s2_done bad=$s2_bad)"

s3_pass=$(grep -c "TEST PASSED" $OUT.3.log || true)
[ "$s3_pass" -ge 1 ] && echo "3 manifest clean:    PASS" || echo "3 manifest clean:    FAIL"

s4_abort=$(grep -c "bytes on unpack but the sender packed" $OUT.4.log || true)
[ "$s4_abort" -ge 1 ] && echo "4 wrong-size abort:  PASS" || echo "4 wrong-size abort:  FAIL (did not abort with the mismatch message)"

s5_abort=$(grep -c "unpack consumed only" $OUT.5.log || true)
[ "$s5_abort" -ge 1 ] && echo "5 skip-buffer abort: PASS" || echo "5 skip-buffer abort: FAIL (did not abort with the consumed-count message)"
echo "========================================="
echo "Per-section logs: $OUT.[1-5].log"
