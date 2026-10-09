#!/bin/bash
# numa_wrap_sph.sh under Nsight Systems: same pinning; each rank writes $SPH_NSYS_DIR/rank_<procid>.nsys-rep with its
# CUDA kernels/memcpys and NVTX ranges (pass +nvtx to the app for the runtime's entry-method and idle ranges).
# SPH_NSYS_OPTS adds nsys options. nsys keeps the caller's LD_PRELOAD (liblci.so stays first) and appends its own.
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-4}
NUMA=$((3 - R))
FIRST=$((NUMA * 16))
LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
OUT=${SPH_NSYS_DIR:-.}/rank_${SLURM_PROCID:-0}
exec nsys profile -t cuda,nvtx -s none --cpuctxsw=none ${SPH_NSYS_OPTS:-} -f true -o "$OUT" "$@" +pemap ${FIRST}-${LAST}
