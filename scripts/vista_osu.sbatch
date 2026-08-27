#!/bin/bash
#SBATCH --job-name=osu-gpu-ipc
#SBATCH --account=MCB24004
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
#SBATCH --output=osu-gpu-ipc-%j.out

# Vista (TACC GH200) counterpart of delta_osu.sbatch: sweep the GPU OSU latency
# and bandwidth benchmarks across the IPC transports and write one CSV.
#
# Two ranks on ONE node, which is all a gh node can offer anyway. The benchmarks
# refuse any other layout: a same-process pair resolves to a plain
# device-to-device copy and a two-node pair resolves to RDMA, neither of which is
# the path under test.
#
# Unlike Delta there is no GPU-layout loop. A gh node is a single Grace-Hopper
# superchip with exactly one H100, so both ranks necessarily drive the same
# device -- Delta's "same" arm, and the cleaner read: no device-to-device link is
# crossed either way, so what separates the transports is staging's second copy
# and the handle work.
#
# Submit from the repository root:
#   sbatch scripts/vista_osu.sbatch

set -u

REPO=$SLURM_SUBMIT_DIR
BUILD=$REPO/multicore-linux-arm8-cuda
export BENCH_DIR=$REPO/examples/charm++/cuda/gpudirect/osu

# Neither libreconverse.so nor the LCI libraries it pulls in are rpath'd into a
# charmc-built binary, so both directories have to be named here rather than
# inherited from whatever the submitting shell had. The same is true at *link*
# time: `make` in $BENCH_DIR needs this exported too, or ld cannot resolve
# liblci.so's own DT_NEEDED entries (liblct.so, liblci-ucx.so).
export LD_LIBRARY_PATH="$BUILD/lib:/work/08450/tg877491/vista/lci_install/lib64:${LD_LIBRARY_PATH:-}"

export CHARMRUN=srun
export LAUNCH_ARGS="--cpu-bind=none --exact -n 2 $REPO/scripts/vista_pin.sh"

# +gpushm is what enables the cross-process path at all.
# +gpucommbuffer 2048 (MB) so the staged arm is not throttled by its own buffer:
#   osu_bw_gpu keeps a 32-deep window in flight, which at the 32 MB top of the
#   sweep is 1 GB of staged blocks. Undersize it and staging serializes on slot
#   reuse, which would read as a transport cost rather than a configuration one.
# +gpuipceventpool 256 per PE: both transports claim an event pair per message,
#   so the window needs at least 32 and the default of 16 is not enough.
# +gpulbbuffer 0: these benchmarks never migrate, so the load-balancing region is
#   dead weight. Passed explicitly because the runtime allocates
#   comm_buffer + lb_buffer in one block and the flag is how you say "none".
export CHARM_ARGS="+ppn 1 +gpushm +gpucommbuffer 2048 +gpulbbuffer 0 +gpuipceventpool 256"

# To 32 MB rather than Delta's 4 MB: on this machine neither bandwidth curve has
# flattened by 4 MB, and the transports are still separating there.
export MIN_BYTES=8
export MAX_BYTES=33554432
export MACHINE=vista-gh200

nvidia-smi topo -m
echo

# MPS is not optional here, it is the difference between measuring the transports
# and measuring the CUDA driver. Both ranks share the node's single H100, and
# without MPS each one gets its own context, so every ping-pong crosses a GPU
# context switch: round-trip latency reads ~154 us instead of ~18 us, and the
# switch cost scales with the number of device copies a transport makes, which
# hands staging a penalty proportional to nothing but the missing MPS server.
# Measured on c642-032, 2026-08-27: no MPS makes direct look a uniform 2.1x
# faster in bandwidth at every size; with MPS the two are level below 512 KB.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-$SLURM_JOB_ID/pipe
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-$SLURM_JOB_ID/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { echo "could not start MPS -- refusing to report numbers" >&2; exit 1; }
trap 'echo quit | nvidia-cuda-mps-control' EXIT
sleep 2
echo

# Correctness before numbers. -v checks an offset-keyed byte pattern, so a
# right-length-wrong-base transfer fails rather than passing. CHARM_ZC_STATS
# prints the staged/direct counts at exit -- read them: if the "direct" run
# reports direct=0, the switch never fired and the sweep below would be measuring
# staging twice.
export CHARM_ZC_STATS=1
#
# Validate from 8 B, not from 1 KB. The comm buffer used to be sized one byte
# short of a power of two, which made the buddy allocator hand out overlapping
# blocks and corrupt the last byte of every power-of-two payload with more than
# one send in flight. It went unnoticed for as long as it did because this loop
# used to start at 1 KB, above the sizes where the damage is visible.
for bench in osu_latency_gpu osu_bw_gpu; do
  for arm in staged direct; do
    echo "---- validate: $bench, arm=$arm ----"
    unset CHARM_GPU_IPC_THRESHOLD
    [ "$arm" = direct ] && export CHARM_GPU_IPC_THRESHOLD=0
    ( cd "$BENCH_DIR" && $CHARMRUN $LAUNCH_ARGS "./$bench" \
        -s 8 -e 262144 -i 20 -w 2 -v $CHARM_ARGS ) || {
      echo "VALIDATION FAILED ($bench, arm=$arm) -- stopping." >&2
      exit 1
    }
    echo
  done
done
unset CHARM_GPU_IPC_THRESHOLD
unset CHARM_ZC_STATS

"$REPO/scripts/ipc_crossover.sh" "$REPO/ipc_crossover_vista_${SLURM_JOB_ID}.csv"

echo "CSV:"
ls -la "$REPO"/ipc_crossover_vista_"${SLURM_JOB_ID}".csv
