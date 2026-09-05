#!/bin/bash
NUMA=$((3 - SLURM_PROCID)); LO=$((NUMA * 16)); HI=$((LO + 7))
exec "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${SLURM_PROCID}.log" 2>&1
