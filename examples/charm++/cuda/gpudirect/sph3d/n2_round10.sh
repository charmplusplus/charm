#!/bin/bash
# round 10 on the held node: control+helper with the runtime's slot-wait print, to split the send stalls.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
until grep -q "round 9 done" $D/logs/hold_round9_$JID.out 2>/dev/null; do sleep 10; done
n=0; until [ "$(stat -c %Y $D/sph3d)" -gt "$(stat -c %Y $D/n2_round10.sh)" ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== round 10 start $(date +%T), left $(squeue -j $JID -h -o %L)"
env CHARM_DEBUG_SENDSLOT=3 CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 TMO=600 DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=cHs_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
L=$(ls -t $D/logs/cHs_256_2000_p25_asyncT_*.log | head -1)
echo "  send stalls over 5 ms: $(grep -c SENDSTALL $L); slot waits over 3 ms: $(grep -c SENDSLOT $L)"
grep SENDSLOT $L | sed -E 's/.*waited ([0-9.]+) ms.*/\1/' | sort -n | awk '{a[NR]=$1} END{if(NR) printf "  slot waits: n=%d p50 %.1f p90 %.1f max %.1f ms\n", NR, a[int(NR/2)+1], a[int(NR*0.9)+1], a[NR]}'
echo "=== round 10 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
