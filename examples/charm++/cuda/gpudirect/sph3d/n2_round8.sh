#!/bin/bash
# round 8 on the held node: the IPC event pool at 512/PE (1024 and 2048 were out of memory creating the IPC events).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" TMO=600 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "CORRUPTION|Reason:|Fatal|Segmentation" $L | head -2 | cut -c1-200; grep -c SENDSTALL $L | sed 's/^/  send stalls over 5 ms: /'; grep -q "rc=0" $L; }
until grep -q "round 7 done" $D/logs/hold_round7_$JID.out 2>/dev/null; do sleep 10; done
echo "=== round 8 start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm cE512 2000 async "" IPCEV=512 CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5; ok=$?
echo "=== round 8 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: 512=$ok; node KEPT"
