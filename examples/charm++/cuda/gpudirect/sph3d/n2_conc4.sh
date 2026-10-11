#!/bin/bash
# n2_conc4.sh <jobid> (2026-10-10): first the user's question -- kernels-only traces of the CONTROL modes on the same nodes
# (async, sync, none) -- then the concurrent exchange with the stuck-state dump: -c async smoke and -c sync smoke (TMO 150 s,
# SPH_STUCK_SECONDS=20), the -c 2000-step arms if the smokes pass, and the -c traces. Releases the node at the end.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <mode sync|async> <extra app args> <env...>
  local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4
  local need=$(( ns > 500 ? 150 : 60 )); local tmo=$(( ns > 500 ? 600 : 150 ))
  [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  local P=25 LAG=10; [ $mode = sync ] && { P=100; LAG=50; }
  env "$@" TMO=$tmo DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "CORRUPTION|Reason:|Fatal|Segmentation|did not consume|overflowed|expected|ERROR" $L | head -3 | cut -c1-220
  grep -c "^\[STUCK\] patch" $L | sed 's/^/  stuck patch lines: /'
  grep "^\[STUCK\] patch" $L | head -6 | cut -c1-230
  grep -A12 "^Message latency" $L | grep -E "halo |migration " | head -6 | cut -c1-200
  grep -q "rc=0" $L
}
trace() {  # <mode> <extra app args>
  [ "$(left_s)" -gt 70 ] || { echo "skip trace $1 $2: $(left_s) s left"; return 1; }
  GT_NOCOPY=1 XTRA_APP="$2" TMO=150 bash $D/n2_gputrace.sh $JID 200 $1 2>&1 | grep -v "^HAPI\[" | cut -c1-200
}
n=0; until [ -f $D/.build_done_conc5 ] || [ $n -ge 60 ]; do sleep 5; n=$((n+1)); done
echo "=== conc4 chain start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
# The helpers (CUPTI load build and the strategy off the PE threads) are part of the async configuration from here on.
H="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5"
run_arm Cs 200 async "-c" $H SPH_STUCK_SECONDS=20; okCs=$?
run_arm Ss 200 sync "-c" SPH_STUCK_SECONDS=20; okSs=$?
run_arm c 2000 async ""; okc=$?
run_arm cH 2000 async "" $H; okcH=$?
[ $okCs = 0 ] && { run_arm CH 2000 async "-c" $H; okCH=$?; }
[ $okSs = 0 ] && { run_arm Sc 2000 sync "-c"; okSc=$?; }
trace asyncH ""
[ "${okCH:-1}" = 0 ] && trace asyncH "-c"
wait
python3 $D/trace_compare.py $(for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_* 2>/dev/null | head -2); do echo "$(basename $T)=$T/trace_last1s.json"; done) 2>&1 | grep -E "=====|busy|by the kernel|idle %" | cut -c1-200
echo "=== conc4 chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: Cs=$okCs Ss=$okSs c=$okc cH=$okcH CH=${okCH:-skipped} Sc=${okSc:-skipped}"
echo "chain done; the node is KEPT for fix-and-retry (release by hand when finished)"
