#!/bin/bash
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
R=${SLURM_PROCID:-0}
if [ "$R" = 4 ]; then
  exec gdb -q -batch -ex "handle SIGUSR1 SIGUSR2 SIG34 nostop noprint pass" -ex run \
       -ex "bt 30" -ex "info registers rip rsp" --args "$@" +pemap "${LO}-${HI}" \
       > "$RANKLOG_DIR/rank_${R}.log" 2>&1
fi
exec "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${R}.log" 2>&1
