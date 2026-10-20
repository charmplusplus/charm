#!/bin/bash
# Pin each rank's PEs to the NUMA node that actually hosts its GPU.
#
# On Delta's gpuA40x4 the GPU<->NUMA mapping is reversed relative to the rank
# order Slurm hands out: nvidia-smi topo -m reports GPU0 on NUMA3, GPU1 on
# NUMA2, GPU2 on NUMA1, GPU3 on NUMA0, while HAPI gives the rank-r process the
# device with global index r. Slurm's default block binding puts rank r on
# NUMA r, so without this every rank talks to its GPU across the interconnect.
#
# Keyed on SLURM_LOCALID, the rank's index WITHIN its node -- not SLURM_PROCID,
# which is global and on two nodes would send ranks 4..7 to negative NUMA
# nodes. (leanmd/run_numa.sh uses PROCID and is single-node only.)
#
#   MODE=match     local rank r -> NUMA 3-r   (the GPU's own NUMA node)
#   MODE=mismatch  local rank r -> NUMA r     (Slurm's default)
#   MODE=none      no pemap
MODE=${MODE:-match}
PES_PER_PROC=${PES_PER_PROC:-8}
LOCAL=${SLURM_LOCALID:-0}
case "$MODE" in
  match)    NUMA=$((3 - LOCAL)) ;;
  mismatch) NUMA=$LOCAL ;;
  none)     NUMA=-1 ;;
  *)        echo "numa_wrap.sh: unknown MODE=$MODE" >&2; exit 2 ;;
esac
if [ "$NUMA" -ge 0 ]; then
  LO=$((NUMA * 16)); HI=$((LO + PES_PER_PROC - 1))
  exec "$@" +pemap "${LO}-${HI}"
else
  exec "$@"
fi
