#!/bin/bash
# NUMA-aware placement for a gpuA40x4 node.
#
# The node is one socket, 64 cores, four NUMA domains of 16, and the GPU order
# is REVERSED against them: nvidia-smi topo -m reports GPU0 on NUMA 3, GPU1 on
# NUMA 2, GPU2 on NUMA 1, GPU3 on NUMA 0. Taking GPU k from the cores Slurm
# hands rank k is therefore the worst pairing available, and with
# --cpus-per-task=8 all four ranks land inside NUMA 0 and 1 while 32-63 sit
# idle.
#
# CUDA_DEVICE_ORDER defaults to FASTEST_FIRST, which is not stable between
# steps; PCI_BUS_ID makes the enumeration deterministic so a rank gets the same
# device every run.
R=${SLURM_PROCID:-0}
PES=${PES:-4}
NUMA=$((3 - R))
FIRST=$((NUMA * 16))
LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$R
exec "$@" +pemap ${FIRST}-${LAST}
