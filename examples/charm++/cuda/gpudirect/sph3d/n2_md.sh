#!/bin/bash
# n2_md.sh <jobid> <floor> (2026-10-11): Metis THEN Diffusion (the run script's own default pair) with the Diffusion imbalance
# floor at <floor>, async period 25 lag 10, against the Metis-alone async@25 control of this pair (eqp: 42.01). Keeps the node.
JID=$1; FLOOR=${2:-0.02}; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="+balancer DiffusionLB +LBDiffusionCommOn ${DIM:-+LBDiffusionGpuDim} +LBDiffusionMinImbalance $FLOOR"
echo "=== md start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), flags: $DF"
env $ENVC TMO=300 DIFF_FLAGS="$DF" LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=md${TAGX:-}${FLOOR}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 ${MODE:-async}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
L=$(ls -t $D/logs/md${TAGX:-}${FLOOR}_256_2000_p25_${MODE:-async}T_*.log | head -1)
grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation" $L | head -2 | cut -c1-200
echo "  whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; $(grep -c 'SPH FAULT' $L) faults; series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%10==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}')"
echo "  events by balancer:"; grep -hE "#Objects migrating" $L | awk '{print $2, $7}' | tr -d ',' | awk '{n[$1]++; if($2>0){a[$1]++; m[$1]+=$2}} END{for(b in n) print "    "b": "a[b]+0" of "n[b]" acted, "m[b]+0" moves"}'
echo "=== md done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
