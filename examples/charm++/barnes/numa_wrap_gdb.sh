#!/bin/bash
# numa_wrap.sh, but rank 7 runs under gdb and prints a backtrace on a fault.
R=${SLURM_PROCID:-0}; L=${SLURM_LOCALID:-$R}; PES=${PES:-4}; NUMA=$((3 - L)); FIRST=$((NUMA * 16)); LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$L
if [ "$R" = 7 ]; then
  exec gdb -q -batch -ex run -ex 'bt 14' -ex 'info threads' --args "$@" +pemap ${FIRST}-${LAST}
else
  exec "$@" +pemap ${FIRST}-${LAST}
fi
