#!/bin/bash
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
