#!/bin/bash
# n2_gate.sh <jobid> (2026-10-11): the CUPTI flush GATE (launches shared, the flush exclusive; hapi_impl.cpp) on the
# pair where the quarantine build faulted twice in two 128-patch arms (gpub[067,074]): 128-patch controls and 256-patch
# controls alternating until the hold runs out, plus the 1-cell band at 128 (lost to the fault earlier). Every arm is a
# crash test of the gate and a step-time check against this pair's controls (256: 39.99 / 40.6). Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1 CHARM_LB_CUPTI_TIME=1"
ROWS="SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <sync|async> <extra app args> <env...>
  local pfx=$1 g="$2" ns=$3 mode=$4 extra=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local P=25 LAG=10; [ $mode = sync ] && { P=100; LAG=50; }
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no|hapiCuptiFlushGated" $L | head -3 | cut -c1-300
  grep -A24 "SPH FAULT" $L | head -26 | cut -c1-200
  echo "  physics @2000: $(grep -E '^  step +2000:' $L | sed -E 's/.*(rho [0-9.]+ +KE [0-9.e+]+ +\|v\|max [0-9.]+).*/\1/'); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); send stalls: $(grep -c SENDSTALL $L); flush ms (mean, PE 0): $(grep '^\[LBCUPTI pe=0\]' $L | sed -E 's/.*flush=([0-9.]+)s.*/\1/' | awk '{f+=$1; n++} END{if(n) printf "%.1f over %d", 1000*f/n, n}')"
  grep -q "rc=0" $L
}
echo "=== gate chain start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm g128c "4 4 8" 2000 async "" $ENVC
run_arm g256c "8 4 8" 2000 async "" $ENVC
run_arm g128b1 "4 4 8" 2000 async "" $ENVC $ROWS SPH_INTERIOR_LAYERS=1
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 16 ]; do i=$((i+1)); if [ $((i % 2)) = 1 ]; then run_arm g128c$i "4 4 8" 2000 async "" $ENVC; else run_arm g256c$i "8 4 8" 2000 async "" $ENVC; fi; done
echo "=== gate chain done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
