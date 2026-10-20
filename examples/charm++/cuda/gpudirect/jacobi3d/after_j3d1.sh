#!/bin/bash
# after_j3d1.sh <jobid> (09:40): jacobi3d WITHOUT load balancing, device channels (-k) vs zero-copy (-d), on the node
# n2_pup1.sh keeps. 32 blocks (4 per GPU) on 8 A40s, 200 iterations after 10 warm-up. Three grids: 512^3 (128^3 blocks,
# 128 KB faces), 1024^3 (512 KB faces), 2048^3 (2 MB faces). Each pair twice. Judge by "Average iteration time".
JID=$1; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; cd $J; ulimit -c 0
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export NODES=2 PPN=4 GPP=1 PES=4 LCI_LIB=/u/bhosale/lci/build-cuda/lib ITERS=200 WARM=10 TMO=240
arm() {  # <mode> <grid>
  local mode=$1 grid=$2; [ "$(left_s)" -gt 260 ] || { echo "skip $mode $grid: $(left_s) s left"; return 1; }
  local TAG=j3d_${mode}_g${grid}_$(date +%H%M%S)
  env MODE=$mode GRID=$grid $J/run_j3d.sh $JID $TAG > /dev/null 2>&1
  local R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | $(grep -a 'Checksum' $R | head -1) | bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|exceed|CHAN-WATCH') | last: $(tail -1 $R | cut -c1-80)"
}
arm reg 512   # host-staged reference for the checksum
for g in 512 1024 2048; do arm zc $g; arm ch $g; done
echo "=== j3d1 done $(date +%T), left $(left_s) s; node KEPT"
