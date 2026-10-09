#!/bin/bash
# n2_rest.sh <jobid>: the rest of allocation 3 -- LeanMD with a vmm pool, the 3% floor pair, the periodic sort, 25M.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
echo "rest start $(date +%T) left $(squeue -j $JID -h -o %L)"
$D/n2_leanmd_pool.sh $JID
export POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="+LBMetisMinGain 0.03" TAGPFX=g3 $D/n2_final2.sh $JID "8 4 8" 2000 syncT asyncT
echo "floor pair done $(date +%T) left $(squeue -j $JID -h -o %L)"
SPH_SORT_EVERY=100 DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="+LBMetisMinGain 0.03" TAGPFX=s100g3 $D/n2_final2.sh $JID "8 4 8" 2000 syncT
echo "sort arm done $(date +%T) left $(squeue -j $JID -h -o %L)"
export NODES=2 NP=8
for ARM in none syncT asyncT; do
  LBM=${ARM%T}; EX=""; [ "$ARM" != "$LBM" ] && EX="+LBMetisHostByMsgs +LBMetisMinGain 0.03"
  TAG=f25M_p128_${ARM}
  DIFF_FLAGS= POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0035 -V 10 -x 4 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 $EX > /dev/null 2>&1
  L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms"
done
echo "rest done $(date +%T) left $(squeue -j $JID -h -o %L)"
