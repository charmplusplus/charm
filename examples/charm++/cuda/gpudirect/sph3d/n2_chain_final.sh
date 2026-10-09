#!/bin/bash
# n2_chain_final.sh <jobid>: the final comparison on 2 nodes, in one go.
JID=$1
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
echo "chain start $(date +%T) node time $(squeue -j $JID -h -o %M)"
LBF=25 LBB=25 LBLAG=10 $D/n2_final.sh $JID "8 4 8" 2000 none syncT asyncT syncT asyncT
LBF=100 LBB=100 LBLAG=50 $D/n2_final.sh $JID "8 4 8" 2000 syncT asyncT
echo "sph3d finals done $(date +%T) node time $(squeue -j $JID -h -o %M)"
/u/bhosale/tnl-sph/examples/WCSPH-DBC/tank3D_d8/run_variant.sh $JID tnl98_lb100_2000 lb100_2000
/u/bhosale/tnl-sph/examples/WCSPH-DBC/tank3D_d8/run_variant.sh $JID tnl98_nolb_2000 nolb_2000
echo "tnl done $(date +%T) node time $(squeue -j $JID -h -o %M)"
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect && STEPS=100 PERIOD=20 LAG=16 bash scale.sh $JID 2 leanmd weak "nolb sync async" 2>&1 | grep -E "leanmd|ms/step|mean|rc=|FAIL|Fatal|abort" | cut -c1-200
echo "chain done $(date +%T) node time $(squeue -j $JID -h -o %M)"
