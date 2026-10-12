#!/bin/bash
# round 3 on the held node (job 22799507): the concurrent exchange with idempotent receive hooks.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <mode> <extra app args> <env...>
  local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4
  local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" TMO=600 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "PUP CORRUPTION|CORRUPTION|Reason:|Fatal|Segmentation|did not consume|overflowed|expected|ERROR" $L | head -3 | cut -c1-420
  grep -q "rc=0" $L
}
until grep -q "round 2 done" $D/logs/hold_round2_$JID.out 2>/dev/null; do sleep 10; done
echo "=== round 3 start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm ScP2 2000 sync "-c" SPH_PUP_CHECK=1; okS=$?
run_arm CH2 2000 async "-c" CHARM_LB_HELPER=near SPH_PUP_CHECK=1 SPH_STUCK_SECONDS=30; okC=$?
if [ $okS = 0 ] && [ $okC = 0 ]; then
  GT_NOCOPY=1 TMO=150 XTRA_APP="-c" bash $D/n2_gputrace.sh $JID 200 asyncH sync 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  python3 $D/trace_compare.py $(for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_* | head -1) $(ls -td /scratch/mzu/bhosale/gputrace/gk200_sync_* | head -1); do echo "$(basename $T)=$T/trace_last1s.json"; done) 2>&1 | grep -E "=====|busy|by the kernel|idle %" | cut -c1-200
fi
echo "=== round 3 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: ScP2=$okS CH2=$okC; node KEPT"
