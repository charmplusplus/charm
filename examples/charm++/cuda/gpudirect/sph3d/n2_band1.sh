#!/bin/bash
# n2_band1.sh <jobid> (2026-10-11): interior-first with a 1-CELL band (SPH_INTERIOR_LAYERS=1 + the migrant scatter
# kernel) against the 2-cell band and the control, 256 patches then 128, async@25 lag 10, helper on, Metis alone,
# 2000 steps, in-session control; a kernels-only trace of the 1-cell form with the target counts; then a sync arm.
# The 2-cell band is a third of the targets and its pass costs 1.6x its share; the 1-cell band is ~17% of the targets.
# Physics check: the step-2000 line (rho, KE, |v|max) of every arm against the control's. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
ROWS="SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <sync|async> <extra app args> <env...>
  local pfx=$1 g="$2" ns=$3 mode=$4 extra=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300
  grep -A24 "SPH FAULT" $L | head -26 | cut -c1-200
  echo "  physics @2000: $(grep -E '^  step +2000:' $L | sed -E 's/.*(rho [0-9.]+ +KE [0-9.e+]+ +\|v\|max [0-9.]+).*/\1/'); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); send stalls: $(grep -c SENDSTALL $L); $(grep -m1 'boundary layer' $L | sed 's/^ *//')"
  grep -q "rc=0" $L
}
echo "=== band1 chain start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm b1c "8 4 8" 2000 async "" $ENVC; okc=$?
run_arm b1 "8 4 8" 2000 async "" $ENVC $ROWS SPH_INTERIOR_LAYERS=1; ok1=$?
run_arm b2 "8 4 8" 2000 async "" $ENVC $ROWS; ok2=$?
run_arm b1h "4 4 8" 2000 async "" $ENVC $ROWS SPH_INTERIOR_LAYERS=1; ok1h=$?
run_arm b1ch "4 4 8" 2000 async "" $ENVC; okch=$?
[ "$(left_s)" -gt 100 ] && SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 SPH_INTERIOR_LAYERS=1 SPH_SPLIT_STATS=1 GT_NOCOPY=1 TMO=100 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* 2>/dev/null | head -1); [ -n "$T" ] && { echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "largest gap" | cut -c1-160; python3 $D/trace_compare.py $(basename $T)=$T/trace_last1s.json 2>&1 | grep -E "busy|by the kernel|idle %" | cut -c1-200; }
grep -h "^\[SPLIT\]" $(ls -t $D/logs/gk200_asyncH_256_*.log | head -1) 2>/dev/null | awk '{p[$6]++; t[$6]+=$7; e[$6]+=$10} END{for(k in p) printf "  SPLIT %s targets %.0f%% of entries over %d prints\n", k, 100*t[k]/e[k], p[k]}'
run_arm b1s "8 4 8" 2000 sync "" SPH_STUCK_SECONDS=30 $ROWS SPH_INTERIOR_LAYERS=1; ok1s=$?
echo "=== band1 chain done $(date +%T), left $(squeue -j $JID -h -o %L); c=$okc b1=$ok1 b2=$ok2 b1h=$ok1h ch=$okch b1s=$ok1s; node KEPT"
