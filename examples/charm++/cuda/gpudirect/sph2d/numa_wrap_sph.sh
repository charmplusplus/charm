#!/bin/bash
# NUMA-aware placement for sph2d on a gpuA40x4 node: like barnes/numa_wrap.sh
# (rank k on the cores of NUMA 3-k, PCI bus order) but WITHOUT narrowing
# CUDA_VISIBLE_DEVICES to one GPU. sph2d's direct-IPC device path needs every
# peer GPU visible: with one device per process the runtime reported "1
# device(s) per host", MetisLB saw a single GPU group and could balance
# nothing, and the 12.5M-particle noLB run went from 150 s to over 400 s.
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-8}
NUMA=$((3 - R))
FIRST=$((NUMA * 16))
LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap ${FIRST}-${LAST}
