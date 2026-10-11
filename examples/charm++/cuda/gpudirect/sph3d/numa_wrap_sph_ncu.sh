#!/bin/bash
# numa_wrap_sph.sh under Nsight Compute: the first NCU_COUNT launches of the forces kernels of every process are
# profiled (SpeedOfLight + memory workload), the rest run normally. Reports to $NCU_DIR/rank_<id>.ncu-rep.
R=${SLURM_LOCALID:-${SLURM_PROCID:-0}}
PES=${PES:-4}
NUMA=$((3 - R))
FIRST=$((NUMA * 16))
LAST=$((FIRST + PES - 1))
export CUDA_DEVICE_ORDER=PCI_BUS_ID
exec /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu --target-processes all -k "regex:${NCU_REGEX:-forces}" -s ${NCU_SKIP:-0} -c ${NCU_COUNT:-4} --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy -f -o ${NCU_DIR}/rank_${SLURM_PROCID:-0} "$@" +pemap ${FIRST}-${LAST}
