#!/bin/bash
# Per-rank launcher for lbcalib: eight PEs on the cores of the GPU's NUMA node, read from the hardware by the
# MoE example's ranklogs/gpunuma.sh (node 3-L on the A40 nodes; the A100 nodes as wired).
L=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
read -r NUMA LO LAST < <(/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/moe/ranklogs/gpunuma.sh "$L"); HI=$((LO + 7))
echo "# rank ${SLURM_PROCID:-0} local $L: GPU $L on NUMA node $NUMA, cores $LO-$HI, host $(hostname)" > "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log"
exec "$@" +pemap "${LO}-${HI}" >> "$RANKLOG_DIR/rank_${SLURM_PROCID:-0}.log" 2>&1
