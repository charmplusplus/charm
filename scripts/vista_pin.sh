#!/bin/bash
# Place one PE per rank on Vista's GH200 nodes, used as the srun target:
#
#   srun --cpu-bind=none -n 2 scripts/vista_pin.sh ./osu_latency_gpu ...
#
# A gh/gh-dev node is one Grace-Hopper superchip: 72 CPU cores in a single NUMA
# domain and exactly one H100. So unlike Delta's gpuA40x4 there is no GPU-to-NUMA
# pairing to get right (`nvidia-smi topo -m` reports CPU affinity 0-71 for the
# one GPU), and no "distinct GPU per rank" layout to run -- both ranks
# necessarily drive the same device. That is Delta's "same" arm, and it is the
# cleaner read anyway: neither transport crosses a device-to-device link, so what
# separates them is staging's second copy and the handle work, nothing else.
#
# Slurm's default binding would hand each rank a single core and the two ranks
# adjacent ones; --cpu-bind=none plus an explicit +pemap here keeps the placement
# in one visible place and puts the ranks half a die apart so the sender and
# receiver are not sharing an L2.
CORE=$(( ${SLURM_PROCID:-0} * 36 ))
exec "$@" +pemap "$CORE"
