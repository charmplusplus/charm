#!/bin/bash
# round 4 on the held node (job 22799507): the plain control on THIS node for the step-time comparison, then
# control+helper with a bigger IPC event pool (the send stalls cluster after LB triggers: slot exhaustion suspected).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" TMO=600 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "CORRUPTION|Reason:|Fatal|Segmentation" $L | head -2 | cut -c1-200; grep -c SENDSTALL $L | sed 's/^/  send stalls over 5 ms: /'; grep -q "rc=0" $L; }
until grep -q "round 3 done" $D/logs/hold_round3_$JID.out 2>/dev/null; do sleep 10; done
echo "=== round 4 start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm c4 2000 async "" SPH_SEND_WARN_MS=5; okc=$?
run_arm cHe 2000 async "" IPCEV=2048 CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5; okE=$?
echo "=== round 4 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: c4=$okc cHe=$okE; node KEPT"
