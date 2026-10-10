#!/bin/bash
# n2_idle2.sh <jobid>: async idle-gap arms, round 2 (2026-10-10). Untraced 2000-step async@25 arms on the SAME nodes and binary:
# control (with CHARM_LB_CUPTI_TIME=1, prints the window-close build cost), S = strategy on the helper thread, O = CUPTI load build
# on the helper thread (now also in CentralLB), OS = both; then traced 200-step asyncOS (gap report); LeanMD weak N=2 if >= 5 min
# remain; scancel at the end (the user asked for the release once done).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <env...>
  local pfx=$1; shift
  [ "$(left_s)" -gt 240 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_2000_p25_asyncT_*.log | head -1); grep -q "rc=0" $L
}
echo "=== idle chain 2 start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm c CHARM_LB_CUPTI_TIME=1; okc=$?
L=$(ls -t $D/logs/c_256_2000_p25_asyncT_*.log | head -1); grep -h "LBCUPTI" $L | sed -E 's/.*flush=([0-9.]+)s process=([0-9.]+)s normalize=([0-9.]+)s total=([0-9.]+)s/\1 \2 \3 \4/' | awk '{f+=$1;p+=$2;z+=$3;t+=$4;n++; if($4>mx)mx=$4} END{if(n) printf "  window-close build (untraced, %d events): flush %.1f parse %.1f normalize %.1f total %.1f ms avg, max %.1f ms\n", n, 1000*f/n, 1000*p/n, 1000*z/n, 1000*t/n, 1000*mx}'
run_arm S CHARM_LB_STRATEGY_THREAD=near; okS=$?
run_arm O CHARM_LB_CUPTI_OFFTHREAD=near; okO=$?
if [ $okS = 0 ] && [ $okO = 0 ]; then
  run_arm OS CHARM_LB_CUPTI_OFFTHREAD=near CHARM_LB_STRATEGY_THREAD=near
  [ "$(left_s)" -gt 150 ] && bash $D/n2_gputrace.sh $JID 200 asyncOS 2>&1 | grep -v "^HAPI\[" | cut -c1-200
else
  echo "S rc ok=$okS O rc ok=$okO: skipping OS arms"
fi
if [ "$(left_s)" -gt 300 ]; then bash $D/n2_leanmd_pool.sh $JID 2>&1 | tail -5 | cut -c1-200; else echo "skip leanmd: $(left_s) s left"; fi
echo "=== idle chain 2 done $(date +%T), left $(squeue -j $JID -h -o %L); releasing"
scancel $JID
