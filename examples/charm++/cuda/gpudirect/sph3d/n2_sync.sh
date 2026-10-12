#!/bin/bash
# n2_sync.sh <jobid>: sync-LB counterpart of n2_conc.sh (2026-10-10, user: "capture a sync LB trace as well"). Same final
# configuration, sync LB at period 100 lag 50: control 2000 steps, -c 2000 steps (the concurrent exchange across a sync
# AtSync migration), then kernels-only +gputrace of sync control and sync -c (200 steps each, with the L records).
# Releases the node at the end.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <extra app args> <env...>
  local pfx=$1 ns=$2 extra=$3; shift 3
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns syncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p100_syncT_*.log | head -1)
  grep -E "CORRUPTION|Abort|abort|Fatal|Segmentation|did not consume|overflowed|expected" $L | head -3 | cut -c1-220
  grep -A12 "^Message latency" $L | grep -E "halo|migration" | head -6
  grep -q "rc=0" $L
}
echo "=== sync chain start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm s 2000 ""; oks=$?
run_arm Sc 2000 "-c"; okSc=$?
[ "$(left_s)" -gt 120 ] && GT_NOCOPY=1 bash $D/n2_gputrace.sh $JID 200 sync 2>&1 | grep -v "^HAPI\[" | cut -c1-200
[ "$(left_s)" -gt 120 ] && [ $okSc = 0 ] && GT_NOCOPY=1 XTRA_APP="-c" bash $D/n2_gputrace.sh $JID 200 sync 2>&1 | grep -v "^HAPI\[" | cut -c1-200
for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_sync_* 2>/dev/null | head -2); do echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "message latency|halo |migration |largest gap" | cut -c1-160; done
echo "=== sync chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: s=$oks Sc=$okSc"
echo "node's arms done: releasing $JID"; scancel $JID
