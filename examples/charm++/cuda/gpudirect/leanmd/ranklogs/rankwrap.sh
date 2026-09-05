#!/bin/bash
# NUMA and GPU follow the node-local rank so this works on any node count;
# the log is named by the global rank.
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
exec "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
