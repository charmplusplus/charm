#!/bin/bash
# numa_wrap_sph.sh under a batch gdb: same pinning, the process runs inside gdb and
# prints every thread's backtrace when it faults. Rank logs go to $SPH_GDB_DIR.
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-8}
NUMA=$((3 - R))
FIRST=$((NUMA * 16))
LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec gdb -q -batch -ex "set pagination off" -ex "handle SIGUSR1 SIGUSR2 SIG34 nostop noprint" \
  -ex "run" -ex "echo \n=== GDB: signal caught, faulting thread ===\n" -ex "bt 30" \
  -ex "echo \n=== all threads ===\n" -ex "thread apply all bt 12" \
  --args "$@" +pemap ${FIRST}-${LAST} > "${SPH_GDB_DIR:-.}/gdb_rank_${SLURM_PROCID:-0}.log" 2>&1
