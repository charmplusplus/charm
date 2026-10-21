#!/bin/bash
# conn3_import.sh <jobid>: convert the raw nsys streams the zero-copy runs left in the node's /tmp into reports
# (the conversion, not the app, is what outlived every timeout). caf5 = no +nvtx, full 30 steps; 6abd = +nvtx,
# 100 s window. Output: /scratch/mzu/bhosale/nsys/conn3_zc_{nonvtx,nvtx}/rank_0.nsys-rep.
JID=$1; I=/opt/nvidia/nsight-systems/2026.1.3/host-linux-x64/QdstrmImporter
mkdir -p /scratch/mzu/bhosale/nsys/conn3_zc_nonvtx /scratch/mzu/bhosale/nsys/conn3_zc_nvtx
echo "=== import start $(date +%T), left $(squeue -j $JID -h -o %L)"
srun --jobid=$JID --overlap -N 1 -n 1 --cpus-per-task=16 --exact bash -c "
  ( $I -i /tmp/nsys-report-caf5.qdstrm -o /scratch/mzu/bhosale/nsys/conn3_zc_nonvtx/rank_0.nsys-rep > /scratch/mzu/bhosale/nsys/conn3_zc_nonvtx/import.log 2>&1; echo \"nonvtx import rc=\$? \$(date +%T)\" ) &
  ( $I -i /tmp/nsys-report-6abd.qdstrm -o /scratch/mzu/bhosale/nsys/conn3_zc_nvtx/rank_0.nsys-rep > /scratch/mzu/bhosale/nsys/conn3_zc_nvtx/import.log 2>&1; echo \"nvtx import rc=\$? \$(date +%T)\" ) &
  wait"
echo "=== import done $(date +%T): $(ls -la /scratch/mzu/bhosale/nsys/conn3_zc_*/rank_0.nsys-rep 2>/dev/null | awk '{print $5, $9}' | tr '\n' ' ')"
