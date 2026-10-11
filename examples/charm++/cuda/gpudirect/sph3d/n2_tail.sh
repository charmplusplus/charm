#!/bin/bash
# n2_tail.sh <jobid> (2026-10-11): the rest of hold 22808905 after the quarantine replicates: wait for the arm in flight,
# relink against the final library, a kernels-only trace of the fixed build (busy %, gaps) and a 96-patch (4 3 8) control /
# interior-first pair: the split tied at 128 and led at 64 (where the landing gate stalled 121 s). Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
ROWS="SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <sync|async> <extra app args> <env...>
  local pfx=$1 g="$2" ns=$3 mode=$4 extra=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local P=25 LAG=10; [ $mode = sync ] && { P=100; LAG=50; }
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300
  grep -A24 "SPH FAULT" $L | head -26 | cut -c1-200
  echo "  send stalls over 5 ms: $(grep -c SENDSTALL $L); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); slowest 25-step interval: $(grep -E '^  step +[0-9]+:' $L | sed -E 's/.*, ([0-9.]+) ms\/step.*/\1/' | sort -n | tail -1) ms"
  grep -q "rc=0" $L
}
Q=$(ls -t $D/logs/q*_256_2000_p25_asyncT_*.log | head -1)
until grep -q "rc=" $Q; do sleep 2; done
echo "=== tail chain start $(date +%T), left $(squeue -j $JID -h -o %L); last replicate $(basename $Q): $(grep -o 'rc=[0-9]*' $Q | tail -1) $(grep -c 'check ok' $Q) checks"
cd $D && export LD_LIBRARY_PATH=/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH && rm -f sph3d && (make 2>&1 | grep -E "error|Error" | tail -2); echo "binary $(ls -la --time-style=+%H:%M:%S $D/sph3d | awk '{print $6}')"
[ "$(left_s)" -gt 100 ] && GT_NOCOPY=1 TMO=100 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* 2>/dev/null | head -1); [ -n "$T" ] && { echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "largest gap" | cut -c1-160; python3 $D/trace_compare.py $(basename $T)=$T/trace_last1s.json 2>&1 | grep -E "busy|idle %" | cut -c1-200; }
run_arm c96 "4 3 8" 2000 async "" $ENVC; okc=$?
run_arm r96 "4 3 8" 2000 async "" $ENVC $ROWS; okr=$?
echo "=== tail chain done $(date +%T), left $(squeue -j $JID -h -o %L); c96=$okc r96=$okr; node KEPT"
