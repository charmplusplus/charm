#!/bin/bash
# n2_hold.sh <jobid> (2026-10-10 night): a long hold for the fix-rebuild-retry loop. Runs the first round of arms, then
# KEEPS the node (the user decides the release). Round 1: kernels-only async+helper trace with the CUPTI-free timestamps;
# sync -c 2000 with SPH_PUP_CHECK (per-patch identity across the migration pup, compared at the first step after landing);
# control+helper 2000 untraced with the send-stall timer.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <mode> <extra app args> <env...>
  local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4
  local tmo=$(( ns > 500 ? 600 : 150 ))
  local P=25 LAG=10; [ $mode = sync ] && { P=100; LAG=50; }
  env "$@" TMO=$tmo DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "PUP CORRUPTION|CORRUPTION|Reason:|Fatal|Segmentation|did not consume|overflowed|expected|ERROR" $L | head -3 | cut -c1-420
  grep -c "SENDSTALL" $L | sed 's/^/  send stalls over 5 ms: /'
  grep -q "rc=0" $L
}
n=0; until [ -f $D/.build_done_hold ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== hold chain start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_* | head -1); python3 $D/trace_compare.py $(basename $T)=$T/trace_last1s.json 2>&1 | grep -E "busy|by the kernel|idle %" | cut -c1-200
run_arm ScP 2000 sync "-c" SPH_PUP_CHECK=1; okScP=$?
run_arm cH 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5; okcH=$?
echo "=== hold chain round 1 done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: ScP=$okScP cH=$okcH"
echo "node KEPT: the user decides the release"
