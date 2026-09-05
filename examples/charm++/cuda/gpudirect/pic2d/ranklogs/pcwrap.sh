#!/bin/bash
# rank r on the cores of NUMA (3-r), which hosts GPU r on gpuA40x4; one PE.
R=${SLURM_PROCID:-0}; NUMA=$((3 - R)); FIRST=$((NUMA * 16))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap $FIRST > "$RANKLOG_DIR/rank_${R}.log" 2>&1
