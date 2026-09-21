#!/bin/bash
# rankwrap.sh under gdb: same pinning and log, the process runs inside a batch
# gdb that prints a backtrace of every thread when it aborts or faults.
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}; NUMA=$((3 - L)); LO=$((NUMA * 16)); HI=$((LO + 7))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec gdb -q -batch -ex "set pagination off" -ex "handle SIGUSR1 SIGUSR2 SIG34 nostop noprint" \
  -ex "run" -ex "echo \n=== GDB: signal caught, backtrace of the faulting thread ===\n" -ex "bt 30" \
  -ex "echo \n=== all threads ===\n" -ex "thread apply all bt 12" \
  --args "$@" +pemap "${LO}-${HI}" > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
