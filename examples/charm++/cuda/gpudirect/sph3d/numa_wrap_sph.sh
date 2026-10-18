#!/bin/bash
# NUMA-aware placement for sph3d on a gpuA40x4 node, for any process/GPU split: process R (local
# rank) drives GPUs R*G .. R*G+G-1 (G = GPP, GPUs per process, default 1; the runtime's contiguous
# device assignment), A40 d sits on NUMA domain 3-d (PCI bus order, reversed vs NUMA), and the K =
# PES/G PEs of each device go on that domain's cores, in PE-rank order (the runtime's block PE map).
# G=1 gives the old map: rank k on the cores of NUMA 3-k.
# CUDA_VISIBLE_DEVICES is NOT narrowed: the direct-IPC device path needs every peer GPU visible
# (with one device per process the runtime reported "1 device(s) per host", MetisLB saw a single
# GPU group and could balance nothing, and the 12.5M-particle noLB run went from 150 s to over 400 s).
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-8}
G=${GPP:-1}
K=$((PES / G))
MAP=""
for i in $(seq 0 $((G - 1))); do
  d=$((R * G + i)); n=$((3 - d)); f=$((n * 16))
  MAP="$MAP${MAP:+,}$f-$((f + K - 1))"
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap $MAP
