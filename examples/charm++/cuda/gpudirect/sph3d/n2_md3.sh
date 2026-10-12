#!/bin/bash
# n2_md3.sh <jobid>: after the V2 draw: (1) a second device-dimension Metis-then-Diffusion draw (G2; was the 45.3 its first
# partition's bad cut 2.29, or the device dimension?), (2) a second Metis-alone async@25 control (eqp2) for the pair's spread.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"; ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
until grep -qE "md done|skip V2" $D/logs/mdV2_chain.out 2>/dev/null; do sleep 5; done
echo "=== md3 start $(date +%T), left $(left_s) s"
[ "$(left_s)" -gt 200 ] && DIM="+LBDiffusionGpuDim" TAGX=G2 bash $D/n2_md.sh $JID 0.02 2>&1 | grep -E "whole-run|acted|faults|Fatal|md done" | cut -c1-200
if [ "$(left_s)" -gt 200 ]; then env $ENVC TMO=300 DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=eqp2_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200; L=$(ls -t $D/logs/eqp2_256_2000_p25_asyncT_*.log | head -1); echo "  eqp2 Metis-alone async@25 whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); first cut: $(grep -m1 -oE 'cross-group cut [0-9.]+ -> [0-9.]+' $L | awk '{print $5}')"; else echo "skip eqp2: $(left_s) s left"; fi
echo "=== md3 done $(date +%T), left $(left_s) s; node KEPT"
