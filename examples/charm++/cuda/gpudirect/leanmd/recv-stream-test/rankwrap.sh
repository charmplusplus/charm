#!/bin/bash
# NUMA and GPU follow the node-local rank so this works on any node count;
# the log is named by the global rank.
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
# PCI order, so device k is the GPU on NUMA (3-k); the default FASTEST_FIRST
# order is not stable between steps (see numa_wrap_sph.sh).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
