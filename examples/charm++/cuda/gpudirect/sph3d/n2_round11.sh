#!/bin/bash
# round 11 on the held node: -c with idempotent consumption: sync 2000 with the audit, then async+helper 2000 with the audit.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" TMO=600 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "IDAUDIT\] (step|id)|CORRUPTION|Reason:|Fatal|Segmentation|expected" $L | grep -v "identical" | head -12 | cut -c1-300; grep -q "rc=0" $L; }
until grep -q "round 10 done" $D/logs/hold_round10_$JID.out 2>/dev/null; do sleep 10; done
n=0; until [ "$(stat -c %Y $D/sph3d)" -gt "$(stat -c %Y $D/n2_round11.sh)" ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== round 11 start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M:%S $D/sph3d | awk '{print $6}')"
run_arm ScA4 2000 sync "-c" SPH_ID_AUDIT=1; okS=$?
run_arm CH3 2000 async "-c" SPH_ID_AUDIT=1 CHARM_LB_HELPER=near SPH_STUCK_SECONDS=30; okC=$?
echo "=== round 11 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: sync=$okS async=$okC; node KEPT"
