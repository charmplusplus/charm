#!/bin/bash
# n2_chain5.sh <jobid>: confirmation set for the final configuration (256 patches, sort/100, MetisLB alone, floor 3%, cut ceiling 10%).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
echo "chain5 start $(date +%T) on $(squeue -j $JID -h -o %N)"
DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="$XF" TAGPFX=c5 $D/n2_final2.sh $JID "8 4 8" 2000 none syncT asyncT syncT asyncT
DIFF_FLAGS= LBF=100 LBB=100 LBLAG=90 XFLAGS="$XF" TAGPFX=c5lag90 $D/n2_final2.sh $JID "8 4 8" 2000 asyncT asyncT
echo "98M set done $(date +%T) node time $(squeue -j $JID -h -o %M)"
SPH_HOST_PROFILE=1 DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="$XF" TAGPFX=c5prof $D/n2_final2.sh $JID "8 4 8" 1000 syncT asyncT
echo "profile pair done $(date +%T)"
export NODES=2 NP=8
for ARM in none syncT asyncT syncT asyncT; do
  LBM=${ARM%T}; EX=""; [ "$ARM" != "$LBM" ] && EX="+LBMetisHostByMsgs $XF"
  TAG=c5_25M_p128_${ARM}_$(date +%H%M%S)
  DIFF_FLAGS= POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0035 -V 10 -x 4 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 $EX > /dev/null 2>&1
  L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,m); if(st>1000){s+=m[1];n++}} END{if(n) printf "%.1f", s/n}') ms"
done
echo "chain5 done $(date +%T) node time $(squeue -j $JID -h -o %M)"
