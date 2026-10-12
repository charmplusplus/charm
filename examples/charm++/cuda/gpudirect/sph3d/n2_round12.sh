#!/bin/bash
# round 12 on the held node: the restructured concurrent exchange (plain receive entries, arrival signals, the LB park
# inside the step after the receives): sync 2000 and async+helper 2000, with the identity audit, the pup check and the
# stuck watchdog.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" TMO=400 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "IDAUDIT\] (step|id)|PUP CORRUPTION|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | grep -v "identical" | head -10 | cut -c1-300; grep -c "STUCK\] patch" $L | sed 's/^/  stuck patch lines: /'; grep "STUCK\] patch" $L | head -3 | cut -c1-250; grep -q "rc=0" $L; }
until grep -q "round 11 done" $D/logs/hold_round11_$JID.out 2>/dev/null; do sleep 10; done
n=0; until [ "$(stat -c %Y $D/sph3d)" -gt "$(stat -c %Y $D/n2_round12.sh)" ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== round 12 start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M:%S $D/sph3d | awk '{print $6}')"
run_arm Sr 2000 sync "-c" SPH_ID_AUDIT=1 SPH_PUP_CHECK=1 SPH_STUCK_SECONDS=30; okS=$?
run_arm Cr 2000 async "-c" SPH_ID_AUDIT=1 SPH_PUP_CHECK=1 SPH_STUCK_SECONDS=30 CHARM_LB_HELPER=near; okC=$?
echo "=== round 12 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: sync=$okS async=$okC; node KEPT"
