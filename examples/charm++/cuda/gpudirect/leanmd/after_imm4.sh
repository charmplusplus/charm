#!/bin/bash
# after_imm4.sh <jobid> (18:29): run-to-run band, 2 repeats each, jacobi3d channels 512^3 and 1024^3: message
# notes (jacobi3d, imm off) vs immediate notes (jacobi3d.immon). Purpose: is the imm3-vs-now2 difference real.
JID=$1; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; cd $J; ulimit -c 0
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib LCIDEV=1 PPN=4 GPP=1 PES=4 ITERS=200 WARM=10 TMO=120
for r in 1 2; do for g in 512 1024; do for bin in jacobi3d jacobi3d.immon; do
  TAG=j3db_${bin}_g${g}_r${r}_$(date +%H%M%S); env BIN=$bin MODE=ch GRID=$g $J/run_j3d_bin.sh $JID $TAG > /dev/null 2>&1
  R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | imm=$(cat $J/ranklogs/$TAG/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'received as immediate') bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed|CHAN-WATCH')"
done; done; done
echo "=== imm4 done $(date +%T); node KEPT"
