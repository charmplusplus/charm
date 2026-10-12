#!/bin/bash
# n2_diff2.sh <jobid>: after n2_diff.sh's D10, the D25 arm that died in a pool collision (two runs on one GPU) and its
# kernels-only trace. DiffusionLB alone under +LBAsync, period 25 lag 10. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="${DIFFX:-+LBDiffusionCommOn +LBDiffusionGpuDim}"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
until grep -q "chain done" $D/logs/diff_chain.out 2>/dev/null; do sleep 5; done
echo "=== diff2 start $(date +%T), left $(squeue -j $JID -h -o %L), runners $(ps -u bhosale -o cmd | grep -E 'run3d\.sh' | grep -v grep | wc -l)"
[ "$(left_s)" -gt 150 ] && env $ENVC TMO=300 LB_BAL=DiffusionLB DIFF_FLAGS="$DF" LBF=25 LBB=25 LBLAG=10 XFLAGS="" TAGPFX=D25b_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 async 2>&1 | grep -v "^HAPI\[" | cut -c1-200
L=$(ls -t $D/logs/D25b_256_2000_p25_async_*.log 2>/dev/null | head -1)
[ -n "$L" ] && echo "  physics @2000: $(grep -E '^  step +2000:' $L | sed -E 's/.*(rho [0-9.]+ +KE [0-9.e+]+ +\|v\|max [0-9.]+).*/\1/'); series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%8==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}'); max ratio seen: $(grep -oE 'max ratio [0-9.]+' $L | awk 'NR%8==1{printf "%s ", $3}')"
[ "$(left_s)" -gt 100 ] && LB_BAL=DiffusionLB DIFF_FLAGS="$DF" GT_NOCOPY=1 TMO=100 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* 2>/dev/null | head -1); [ -n "$T" ] && { echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "largest gap" | cut -c1-160; python3 $D/trace_compare.py $(basename $T)=$T/trace_last1s.json 2>&1 | grep -E "busy|idle %" | cut -c1-200; }
echo "=== diff2 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
