#!/bin/bash
# numa_wrap_lmd.sh: pins one leanmd process's PEs. PES PEs per process over GPP GPUs (K = PES/GPP per GPU) on the first K
# cores of each GPU's NUMA domain (A40 node: GPU d sits on domain 3-d, cores (3-d)*16.., as sph3d/numa_wrap_sph.sh);
# the process's rank log goes to $RANKLOG_DIR/rank_<procid>.log (leanmd prints per rank).
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-8}; G=${GPP:-1}; K=$((PES / G)); MAP=""
for i in $(seq 0 $((G - 1))); do d=$((R * G + i)); n=$((3 - d)); f=$((n * 16)); MAP="$MAP${MAP:+,}$f-$((f + K - 1))"; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec "$@" +pemap $MAP > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
