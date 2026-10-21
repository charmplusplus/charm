#!/bin/bash
# numa_wrap_lmd.sh under Nsight Systems for rank 0 only (the other ranks run plain): same pinning and rank log;
# rank 0 writes $LMD_NSYS_DIR/rank_0.nsys-rep with its CUDA kernels, memcpys and NVTX ranges (+nvtx on the app).
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-8}; G=${GPP:-1}; K=$((PES / G)); MAP=""
for i in $(seq 0 $((G - 1))); do d=$((R * G + i)); n=$((3 - d)); f=$((n * 16)); MAP="$MAP${MAP:+,}$f-$((f + K - 1))"; done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ "${SLURM_PROCID:-0}" = "${LMD_NSYS_RANK:-0}" ]; then
  exec nsys profile -t cuda,nvtx -s none --cpuctxsw=none ${LMD_NSYS_OPTS:-} -f true -o "${LMD_NSYS_DIR:-.}/rank_${SLURM_PROCID:-0}" "$@" +pemap $MAP > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
else
  exec "$@" +pemap $MAP > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
fi
