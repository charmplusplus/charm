#!/bin/bash
# n2_nocopy.sh <jobid>: the kernels-only (+gputracenocopy) look at the async LB bubble (2026-10-10). Same nodes, same binary:
# untraced async@25 200 steps (control for the trace tax), then +gputrace +gputracenocopy async and asyncOS (ONE variable: the
# helpers), merged traces + gap reports (gaps > 5 ms) on the login node. Releases the node when every arm returned rc=0;
# otherwise keeps it and says so.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
echo "=== nocopy chain start $(date +%T), left $(squeue -j $JID -h -o %L)"
env DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=u_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 200 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
L=$(ls -t $D/logs/u_256_200_p25_asyncT_*.log | head -1); grep -q "rc=0" $L; ok1=$?
GT_NOCOPY=1 bash $D/n2_gputrace.sh $JID 200 async asyncOS 2>&1 | grep -v "^HAPI\[" | cut -c1-200
ok2=$(ls -t $D/logs/gk200_async_*.log | head -1 | xargs grep -c "rc=0"); ok3=$(ls -t $D/logs/gk200_asyncOS_*.log | head -1 | xargs grep -c "rc=0")
for t in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_async*_* | head -2); do echo "  $(basename $t): $(grep -c '^K' $t/*.proc | awk -F: '{s+=$2} END{print s}') K, $(grep -c '^M' $t/*.proc | awk -F: '{s+=$2} END{print s}') M records"; done
echo "=== nocopy chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: untraced=$ok1 async=$ok2 asyncOS=$ok3"
if [ "$ok1" = 0 ] && [ "$ok2" = 1 ] && [ "$ok3" = 1 ]; then echo "all arms clean: releasing $JID"; scancel $JID; else echo "an arm failed: node KEPT for a look"; fi
