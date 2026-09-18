#!/bin/bash
# Same pinning as rankwrap.sh; every rank runs under gdb batch so whichever one
# faults prints its own stack (no core file -- /u quota).
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
R=${SLURM_PROCID:-0}
exec gdb -q -batch \
  -ex "set confirm off" \
  -ex "set pagination off" \
  -ex "handle SIGUSR1 SIGUSR2 SIG34 SIG35 SIG36 nostop noprint pass" \
  -ex run \
  -ex "printf \"\\n===== FAULT rank $R =====\\n\"" \
  -ex "bt 40" \
  -ex "info registers rip rsp" \
  -ex "thread apply all bt 12" \
  --args "$@" +pemap "${LO}-${HI}" \
  > "$RANKLOG_DIR/rank_${R}.log" 2>&1
