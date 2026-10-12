#!/bin/bash
# round 5 on the held node: sync -c 2000 with the identity audit (names the duplicated and lost ids and their holders).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
until grep -q "round 4 done" $D/logs/hold_round4_$JID.out 2>/dev/null; do sleep 10; done
n=0; until [ "$(stat -c %Y $D/sph3d)" -gt "$(stat -c %Y $D/n2_round5.sh)" ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== round 5 start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M:%S $D/sph3d | awk '{print $6}')"
env SPH_ID_AUDIT=1 TMO=600 DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="$XF -c" TAGPFX=ScA_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 syncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
L=$(ls -t $D/logs/ScA_256_2000_p100_syncT_*.log | head -1)
grep -E "IDAUDIT|CORRUPTION|Reason:" $L | grep -v "identical to the reference" | head -24 | cut -c1-300
echo "=== round 5 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
