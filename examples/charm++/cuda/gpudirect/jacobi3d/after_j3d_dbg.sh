#!/bin/bash
# after_j3d_dbg.sh <jobid> (11:02): on the leanmd allocation after its arms (the QOS allows one job at a time). TARGET: where the zero-copy and channel modes stall
# after init (both hung on job 23011983 at 512^3 while the host-staged mode ran). VARIABLE: the mode. Each arm: 16
# blocks (4 processes x 4 PEs), 256^3, 3 iterations after 1 warm-up, -v (block 0 prints its progress), 60 s timeout.
J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; cd $J; ulimit -c 0
JID=$1
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "on job $JID, $(left_s) s left"
export NODES=2 PPN=4 GPP=1 PES=4 LCI_LIB=/u/bhosale/lci/build-cuda/lib ITERS=3 WARM=1 TMO=60 POOL=4096
for mode in reg zc ch; do
  [ "$(left_s)" -gt 90 ] || { echo "skip $mode: $(left_s) s left"; continue; }
  TAG=j3ddbg_${mode}_$(date +%H%M%S)
  env MODE=$mode GRID=256 $J/run_j3d.sh $JID $TAG -v > /dev/null 2>&1
  R=$J/ranklogs/$TAG
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time\|Checksum' $R/rank_*.log | tr '\n' ' ' | cut -c1-120)"
  echo "  trace (block 0):"; cat $R/rank_*.log | grep -a '^\[j3d' | head -40 | sed 's/^/    /'
  echo "  diagnostics:"; cat $R/rank_*.log | grep -a 'CHAN-WATCH\|CHAN-STALL\|rror\|abort\|Fatal' | head -6 | cut -c1-200 | sed 's/^/    /'
done
echo "=== j3d dbg done $(date +%T); node KEPT"
