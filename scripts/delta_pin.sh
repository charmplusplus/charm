#!/bin/bash
# Place one PE per rank on the NUMA node that hosts the GPU that rank will drive,
# and select which GPU that is. Used as the srun target:
#
#   srun --cpu-bind=none -n 2 scripts/delta_pin.sh ./osu_latency_gpu ...
#
# On Delta's gpuA40x4, `nvidia-smi topo -m` reports GPU g on NUMA node 3-g,
# while Slurm's default block binding would put rank r on NUMA r -- so every
# rank would reach its GPU across the interconnect. Hence --cpu-bind=none plus
# an explicit +pemap from here. Same reasoning as leanmd/run_numa.sh, which
# does this for 8 PEs per rank; this is the one-PE-per-rank version.
#
# GPU_FOR_RANK=distinct (default)  rank r drives GPU r -- peer copies between
#                                  two GPUs, the layout a multi-GPU job has.
# GPU_FOR_RANK=same                every rank drives GPU 0 -- copies within one
#                                  device, which is the cleanest read on
#                                  staging's second copy since neither transport
#                                  crosses a link.
#
# HAPI assigns devices itself (round robin over what the process can see), so
# "distinct" needs nothing but the default, and "same" is arranged by hiding the
# other GPUs.
MODE=${GPU_FOR_RANK:-distinct}

case "$MODE" in
  distinct)
    GPU=$SLURM_PROCID
    ;;
  same)
    GPU=0
    export CUDA_VISIBLE_DEVICES=0
    ;;
  *)
    echo "delta_pin.sh: GPU_FOR_RANK must be 'distinct' or 'same', got '$MODE'" >&2
    exit 1
    ;;
esac

NUMA=$((3 - GPU))
# Offset by rank so two ranks sharing a NUMA node do not land on one core.
CORE=$((NUMA * 16 + SLURM_PROCID))

exec "$@" +pemap "$CORE"
