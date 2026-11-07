#!/bin/bash
# gpunuma.sh <local rank>: prints "NUMA FIRST LAST" for the GPU this rank drives: the L-th device in PCI bus
# order (CUDA_DEVICE_ORDER=PCI_BUS_ID), its NUMA node from sysfs, and that node's core range. The A40 nodes
# number GPUs in the reverse order of their NUMA nodes (GPU L on node 3-L); the A100 nodes need not, so the
# mapping is read from the hardware, and the A40 formula is the fallback when sysfs or nvidia-smi is unavailable.
L=${1:-0}
bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | sort | sed -n "$((L + 1))p" | tr 'A-F' 'a-f' | sed 's/^0000//')
node=""
[ -n "$bus" ] && [ -r "/sys/bus/pci/devices/$bus/numa_node" ] && node=$(cat "/sys/bus/pci/devices/$bus/numa_node")
if [ -n "$node" ] && [ "$node" -ge 0 ] 2>/dev/null && [ -r "/sys/devices/system/node/node$node/cpulist" ]; then
  range=$(cut -d, -f1 "/sys/devices/system/node/node$node/cpulist")
  echo "$node ${range%-*} ${range#*-}"
else
  n=$((3 - L)); echo "$n $((n * 16)) $((n * 16 + 15))"
fi
