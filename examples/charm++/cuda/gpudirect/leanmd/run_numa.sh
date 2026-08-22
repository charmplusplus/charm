#!/bin/bash
# Launch leanmd with each rank pinned to the NUMA node that actually hosts its GPU.
#
# On Delta's gpuA40x4 the GPU<->NUMA mapping is reversed relative to the rank
# order Slurm hands out: nvidia-smi topo -m reports GPU0 on NUMA3, GPU1 on NUMA2,
# GPU2 on NUMA1, GPU3 on NUMA0, while HAPI gives process rank r the device with
# global index r (hapi_impl.cpp: device_managers.emplace_back(i, device_count *
# CmiMyNodeRankLocal() + i)). Slurm's default block binding puts rank r on NUMA r,
# so every rank ends up talking to a GPU across the interconnect.
#
# Run with --cpu-bind=none so the whole node is reachable, and place the PEs here.
#   MODE=match     rank r -> NUMA 3-r  (the GPU's own NUMA node)
#   MODE=mismatch  rank r -> NUMA r    (what Slurm does by default)
#   MODE=none      no pemap at all
MODE=${MODE:-match}
PES_PER_PROC=${PES_PER_PROC:-8}
case "$MODE" in
  match)    NUMA=$((3 - SLURM_PROCID)) ;;
  mismatch) NUMA=$SLURM_PROCID ;;
  none)     NUMA=-1 ;;
esac
if [ "$NUMA" -ge 0 ]; then
  LO=$((NUMA * 16)); HI=$((LO + PES_PER_PROC - 1))
  exec "$@" +pemap "${LO}-${HI}"
else
  exec "$@"
fi
