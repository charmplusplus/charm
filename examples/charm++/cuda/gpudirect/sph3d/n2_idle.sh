#!/bin/bash
# n2_idle.sh <jobid>: the async LB idle-gap arms of 2026-10-10 -- traced 200-step asyncO / asyncOS (gap vs the 46 ms baseline
# gt200_async_165046), untraced 2000-step asyncOS@25 and asyncO@25 (vs 36.9 un-helped), LeanMD weak N=2 if >= 5 min remain
# (both knobs are env-gated and off there), then scancel (the user asked for the release once done).
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
echo "=== idle chain start $(date +%T), left $(squeue -j $JID -h -o %L)"
bash $D/n2_gputrace.sh $JID 200 asyncO asyncOS 2>&1 | grep -v "^HAPI\[" | cut -c1-220
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
for arm in "OS CHARM_LB_CUPTI_OFFTHREAD=near CHARM_LB_STRATEGY_THREAD=near" "O CHARM_LB_CUPTI_OFFTHREAD=near"; do
  set -- $arm; sfx=$1; shift
  [ "$(left_s)" -gt 270 ] || { echo "skip untraced $sfx: $(left_s) s left"; continue; }
  env "$@" DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=h${sfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-220
done
if [ "$(left_s)" -gt 300 ]; then bash $D/n2_leanmd_pool.sh $JID 2>&1 | tail -5 | cut -c1-200; else echo "skip leanmd: $(left_s) s left"; fi
echo "=== idle chain done $(date +%T), left $(squeue -j $JID -h -o %L); releasing"
scancel $JID
