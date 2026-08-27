#!/bin/bash
# Run the intra-node, cross-process GPU messaging benchmarks under each
# transport and emit one CSV, so the crossover between them can be read off per
# machine. Each run sweeps message size (8 B to 4 MB, doubling); the size is
# what is being swept, and it is the answer being looked for, since the
# crossover IS a size.
#
# There are two transports:
#
#   staged        CHARM_GPU_IPC_THRESHOLD unset -- today's behaviour, every
#                 cross-process send copied through the device communication
#                 buffer.
#   direct        CHARM_GPU_IPC_THRESHOLD=0 -- every send exports its source
#                 allocation. Peer mappings are cached, which is the steady
#                 state a real run sees and the arm that sets the default.
#
# and one diagnostic that is not a transport:
#
#   direct-nocache
#                 direct with CHARM_GPU_IPC_CACHE=0, reopening and closing the
#                 mapping per transfer. This prices the handle operations
#                 themselves; it also synchronizes each receive to make the
#                 close safe, so read it as an upper bound on the handle cost,
#                 not as something anyone would ship. Drop this arm if you only
#                 want the crossover.
#
# The crossover is where the staged and direct curves meet. Round down to a
# power of two for the default threshold on that machine.
#
# Usage: scripts/ipc_crossover.sh [output.csv]
#
# Environment:
#   BENCH_DIR   where the benchmarks were built
#               (default examples/charm++/cuda/gpudirect/osu)
#   CHARMRUN    launcher (default ./charmrun)
#   LAUNCH_ARGS process layout -- MUST put the two endpoints in different
#               processes on ONE host (default "+p2 ++ppn 1")
#   CHARM_ARGS  extra runtime arguments (default "+gpushm +gpucommbuffer 256")
#   MIN_BYTES / MAX_BYTES   size sweep bounds (default 8 .. 4194304)
#   MACHINE     name recorded in the CSV's machine column (default `hostname -s`,
#               which is wrong when srun sends the work to another node)
#   ARMS        which arms to run, space separated, from the three named above
#               (default all three). Set to "staged direct" for the crossover
#               alone, without the handle-cost diagnostic.

set -u

BENCH_DIR=${BENCH_DIR:-examples/charm++/cuda/gpudirect/osu}
CHARMRUN=${CHARMRUN:-./charmrun}
LAUNCH_ARGS=${LAUNCH_ARGS:-"+p2 ++ppn 1"}
CHARM_ARGS=${CHARM_ARGS:-"+gpushm +gpucommbuffer 256"}
MIN_BYTES=${MIN_BYTES:-8}
MAX_BYTES=${MAX_BYTES:-4194304}
ARMS=${ARMS:-"staged direct direct-nocache"}
OUT=${1:-ipc_crossover.csv}

# The machine the numbers describe, which is not always the one running this
# script: driven from a login node with CHARMRUN=srun, `hostname -s` names the
# login node and the CSV would attribute the results to it.
MACHINE=${MACHINE:-$(hostname -s)}

if [ ! -x "$BENCH_DIR/osu_latency_gpu" ] || [ ! -x "$BENCH_DIR/osu_bw_gpu" ]; then
  echo "error: benchmarks not built. Run 'make' in $BENCH_DIR first." >&2
  exit 1
fi

echo "machine,benchmark,config,cache,size_bytes,value" > "$OUT"

# $1 = benchmark binary, $2 = config label, $3 = cache label. Reads the
# benchmark's "<size> <value>" table -- the value is latency in us or bandwidth
# in MB/s depending on which benchmark ran -- and appends it to the CSV.
run_one() {
  local bench=$1 config=$2 cache=$3
  echo "== $bench / $config (cache $cache)" >&2
  ( cd "$BENCH_DIR" && \
    $CHARMRUN $LAUNCH_ARGS "./$bench" -s "$MIN_BYTES" -e "$MAX_BYTES" $CHARM_ARGS ) \
  | awk -v m="$MACHINE" -v b="$bench" -v c="$config" -v k="$cache" \
        '/^#/ {next}
         NF==2 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+\.[0-9]+$/ {
           printf "%s,%s,%s,%s,%s,%s\n", m, b, c, k, $1, $2
         }' \
  >> "$OUT"
}

for bench in osu_latency_gpu osu_bw_gpu; do
  for arm in $ARMS; do
    case "$arm" in
      staged)
        unset CHARM_GPU_IPC_THRESHOLD CHARM_GPU_IPC_CACHE
        run_one "$bench" staged on
        ;;
      direct)
        export CHARM_GPU_IPC_THRESHOLD=0
        unset CHARM_GPU_IPC_CACHE
        run_one "$bench" direct on
        ;;
      direct-nocache)
        export CHARM_GPU_IPC_THRESHOLD=0
        export CHARM_GPU_IPC_CACHE=0
        run_one "$bench" direct-nocache off
        ;;
      *)
        echo "error: unknown arm '$arm' in ARMS" >&2
        exit 1
        ;;
    esac
  done
done

unset CHARM_GPU_IPC_THRESHOLD CHARM_GPU_IPC_CACHE

echo >&2
echo "wrote $OUT" >&2
echo "crossover: the smallest size where 'direct' beats 'staged' in both" >&2
echo "benchmarks. Round down to a power of two and set that as" >&2
echo "CHARM_GPU_IPC_THRESHOLD (or +gpuipcthreshold) on $MACHINE." >&2
