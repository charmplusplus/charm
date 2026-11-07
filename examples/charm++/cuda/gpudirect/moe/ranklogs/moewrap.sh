#!/bin/bash
# Per-rank launcher for moe under srun: one process per GPU, pinned to the
# cores of the GPU's NUMA node (read from the hardware by gpunuma.sh: the A40
# nodes number GPUs in the reverse order of their NUMA nodes, the A100 nodes
# are read the same way), with this rank's output in its own log.
#
# MOE_PPN=N runs N PEs per process on N consecutive cores of that NUMA node
# (default 1). The process still has one dispatcher and its -t tokens, on its
# first PE; the experts spread over all N PEs.
#
# MOE_ONE_DEVICE=1 shows each process only its own GPU (CUDA_VISIBLE_DEVICES).
# The CUDA driver keeps one event-handler thread per GPU a process can see
# (four with the whole node visible, one with this). Measured 12 Sep 2026:
# it does NOT change the eight-PE output-exchange stall (52 ms either way),
# so that stall is not the driver threads' doing; kept as an experiment knob.
R=${SLURM_PROCID:-0}; L=${SLURM_LOCALID:-$R}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# The GPU's NUMA node and core range from the hardware (gpunuma.sh): the A40 nodes put GPU L on node 3-L,
# other nodes need not.
read -r NUMA FIRST LAST < <("$(dirname "$0")/gpunuma.sh" "$L")
PPN=${MOE_PPN:-1}
# Extra environment for a run whose job script is already queued: ranklogs/env.extra, if present, is sourced here.
[ -f "$(dirname "$0")/env.extra" ] && . "$(dirname "$0")/env.extra"
echo "# rank $R local $L: GPU $L on NUMA node $NUMA, cores $FIRST-$LAST, host $(hostname)" > "$RANKLOG_DIR/rank_${R}.log"
[ "${MOE_ONE_DEVICE:-0}" = 1 ] && export CUDA_VISIBLE_DEVICES=$L
if [ -n "${MOE_GPUS_PER_PROC:-}" ]; then
  # One process per node driving MOE_GPUS_PER_PROC GPUs (the example puts a dispatcher on each): PE i on the first core of GPU i's NUMA
  # node, all GPUs visible, the runtime's block map putting PE i on device i.
  G=$MOE_GPUS_PER_PROC; MAP=""
  for ((i = 0; i < G; i++)); do read -r n f l < <("$(dirname "$0")/gpunuma.sh" "$i"); MAP="$MAP${MAP:+,}$f"; done
  echo "# rank $R: $G GPUs in one process, PEs on cores $MAP" >> "$RANKLOG_DIR/rank_${R}.log"
  exec "$@" +ppn $G +pemap $MAP >> "$RANKLOG_DIR/rank_${R}.log" 2>&1
elif [ "$PPN" -gt 1 ]; then
  exec "$@" +ppn $PPN +pemap $FIRST-$((FIRST + PPN - 1)) >> "$RANKLOG_DIR/rank_${R}.log" 2>&1
else
  exec "$@" +pemap $FIRST >> "$RANKLOG_DIR/rank_${R}.log" 2>&1
fi
