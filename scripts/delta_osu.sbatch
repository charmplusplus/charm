#!/bin/bash
#SBATCH --job-name=osu-gpu-ipc
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=osu-gpu-ipc-%j.out

# Sweep the GPU OSU latency and bandwidth benchmarks across both transports, in
# both GPU layouts, and write two CSVs. Two ranks on ONE node: the benchmarks
# refuse to run in any other layout, because a same-process pair resolves to a
# plain device-to-device copy and a two-node pair resolves to RDMA -- neither of
# which is the path under test.
#
# Submit from the repository root:
#   sbatch scripts/delta_osu.sbatch

set -u

REPO=$SLURM_SUBMIT_DIR
export BENCH_DIR=$REPO/examples/charm++/cuda/gpudirect/osu

# libreconverse.so is not rpath'd into the benchmark binaries, so it has to be
# named here rather than inherited from whatever the submitting shell had.
export LD_LIBRARY_PATH="$REPO/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"

# libinfinipath installs a SIGSEGV handler that swallows real crash reports.
export IPATH_NO_BACKTRACE=1

# Which GPU layouts to sweep, in order (see scripts/delta_pin.sh):
#   distinct  one GPU per rank -- peer copies, the multi-GPU job's layout
#   same      both ranks on GPU 0 -- neither transport crosses a link, which is
#             the cleanest read on what staging's second copy costs
LAYOUTS=${GPU_LAYOUTS:-"distinct same"}

export CHARMRUN=srun
export LAUNCH_ARGS="--cpu-bind=none --exact -n 2 $REPO/scripts/delta_pin.sh"

# +gpushm is what enables the cross-process path at all.
# +gpucommbuffer 512 (MB) so the staged arm is not throttled by its own buffer:
#   osu_bw_gpu keeps a 32-deep window of up to 4 MB messages in flight, which is
#   128 MB of staged blocks. Undersize it and staging serializes on slot reuse,
#   which would read as a transport cost rather than a configuration one.
# +gpuipceventpool 256 per PE: both transports claim an event pair per message,
#   so the window needs at least 32 and the default of 16 is not enough.
# +gpulbbuffer 0: these benchmarks never migrate, so the load-balancing region
#   is dead weight. Passed explicitly because the runtime allocates
#   comm_buffer + lb_buffer in one block and the flag is how you say "none".
export CHARM_ARGS="+ppn 1 +gpushm +gpucommbuffer 512 +gpulbbuffer 0 +gpuipceventpool 256"

export MIN_BYTES=8
export MAX_BYTES=4194304

nvidia-smi topo -m
echo

# Correctness before numbers. This is the first cross-process execution of the
# direct transport, so run both benchmarks with -v (which checks an offset-keyed
# byte pattern, so a right-length-wrong-base transfer fails rather than passing)
# under both transports, over a small size range. CHARM_ZC_STATS prints the
# staged/direct counts at exit -- read them: if the "direct" run reports
# direct=0, the switch never fired and the whole sweep below would be measuring
# staging twice.
export GPU_FOR_RANK=${LAYOUTS%% *}
export CHARM_ZC_STATS=1

for bench in osu_latency_gpu osu_bw_gpu; do
  for threshold in unset 0; do
    echo "---- validate: $bench, threshold=$threshold ----"
    if [ "$threshold" = unset ]; then unset CHARM_GPU_IPC_THRESHOLD;
    else export CHARM_GPU_IPC_THRESHOLD=0; fi
    ( cd "$BENCH_DIR" && $CHARMRUN $LAUNCH_ARGS "./$bench" \
        -s 1024 -e 262144 -i 20 -w 2 -v $CHARM_ARGS ) || {
      echo "VALIDATION FAILED ($bench, threshold=$threshold) -- stopping." >&2
      exit 1
    }
    echo
  done
done

unset CHARM_GPU_IPC_THRESHOLD
unset CHARM_ZC_STATS

for layout in $LAYOUTS; do
  export GPU_FOR_RANK=$layout
  echo "================ GPU layout: $layout ================"
  "$REPO/scripts/ipc_crossover.sh" "$REPO/ipc_crossover_${layout}_${SLURM_JOB_ID}.csv"
  echo
done

echo "CSVs:"
ls -la "$REPO"/ipc_crossover_*_"${SLURM_JOB_ID}".csv
