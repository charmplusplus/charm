#!/bin/bash
# Per-rank launcher for moe under srun: one process per GPU, pinned to the
# cores of the GPU's NUMA node (the A40 nodes number GPUs in the reverse
# order of their NUMA nodes), with this rank's output in its own log.
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
R=${SLURM_PROCID:-0}; L=${SLURM_LOCALID:-$R}; NUMA=$((3 - L)); FIRST=$((NUMA * 16))
PPN=${MOE_PPN:-1}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
[ "${MOE_ONE_DEVICE:-0}" = 1 ] && export CUDA_VISIBLE_DEVICES=$L
if [ "$PPN" -gt 1 ]; then
  exec "$@" +ppn $PPN +pemap $FIRST-$((FIRST + PPN - 1)) > "$RANKLOG_DIR/rank_${R}.log" 2>&1
else
  exec "$@" +pemap $FIRST > "$RANKLOG_DIR/rank_${R}.log" 2>&1
fi
