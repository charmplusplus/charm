#!/bin/bash
# n2_diff.sh <jobid> (2026-10-11): DiffusionLB ALONE under +LBAsync (few moves per event, to neighbours) against the
# MetisLB-alone controls of this pair (gpub[067,074]: 39.99-41.21 ms/step whole-run): period 25 lag 10, then a
# kernels-only trace of it (busy %, gaps = the migration bubbles), then period 10 lag 4. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="${DIFFX:-+LBDiffusionCommOn}"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <period> <lag> <env...>
  local pfx=$1 g="$2" ns=$3 P=$4 LAG=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 LB_BAL=DiffusionLB DIFF_FLAGS="$DF" LBF=$P LBB=$P LBLAG=$LAG XFLAGS="" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns async 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_async_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300
  echo "  physics @2000: $(grep -E '^  step +2000:' $L | sed -E 's/.*(rho [0-9.]+ +KE [0-9.e+]+ +\|v\|max [0-9.]+).*/\1/'); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); send stalls: $(grep -c SENDSTALL $L); slowest 25-step interval: $(grep -E '^  step +[0-9]+:' $L | sed -E 's/.*, ([0-9.]+) ms\/step.*/\1/' | sort -n | tail -1) ms"
  grep -q "rc=0" $L
}
A=$(ls -t $D/logs/off*_256_2000_p25_asyncT_*.log 2>/dev/null | head -1); [ -n "$A" ] && until grep -q "rc=" $A; do sleep 2; done
echo "=== diffusion chain start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), flags $DF"
run_arm D25 "8 4 8" 2000 25 10 $ENVC; ok25=$?
[ "$(left_s)" -gt 100 ] && LB_BAL=DiffusionLB DIFF_FLAGS="$DF" GT_NOCOPY=1 TMO=100 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* 2>/dev/null | head -1); [ -n "$T" ] && { echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "largest gap" | cut -c1-160; python3 $D/trace_compare.py $(basename $T)=$T/trace_last1s.json 2>&1 | grep -E "busy|idle %" | cut -c1-200; }
run_arm D10 "8 4 8" 2000 10 4 $ENVC; ok10=$?
echo "=== diffusion chain done $(date +%T), left $(squeue -j $JID -h -o %L); D25=$ok25 D10=$ok10; node KEPT"
