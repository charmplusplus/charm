#!/bin/bash
# n2_conc.sh <jobid>: the concurrent exchange (-c) and border-patch stream priority (SPH_PRIO_BORDER=1) against the
# control, same nodes, same binary, final 8-GPU configuration (256 patches, sort/100, Metis alone, async@25 lag 10,
# 32 GB pool). Arms: control 2000; C smoke 200; C 2000; P 2000; CP 2000; then a kernels-only +gputrace of C (200 steps)
# with the message-latency records. Every arm prints the message-latency summary. Releases the node at the end.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <extra app args> <env...>
  local pfx=$1 ns=$2 extra=$3; shift 3
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p25_asyncT_*.log | head -1)
  grep -E "CORRUPTION|Abort|abort|Fatal|Segmentation|did not consume|overflowed|expected" $L | head -3 | cut -c1-220
  grep -A12 "^Message latency" $L | grep -E "halo|migration" | head -6
  grep -q "rc=0" $L
}
echo "=== conc chain start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm c 2000 ""; okc=$?
run_arm Cs 200 "-c"; okCs=$?
if [ $okCs = 0 ]; then
  run_arm C 2000 "-c"; okC=$?
else
  echo "C smoke failed (rc ok=$okCs): skipping the long C arms"; okC=1
fi
run_arm P 2000 "" SPH_PRIO_BORDER=1; okP=$?
if [ $okC = 0 ]; then run_arm CP 2000 "-c" SPH_PRIO_BORDER=1; okCP=$?; fi
if [ $okC = 0 ] && [ "$(left_s)" -gt 120 ]; then
  GT_NOCOPY=1 XTRA_APP="-c" bash $D/n2_gputrace.sh $JID 200 async 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_async_* | head -1)
  python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "message latency|halo |migration |largest gap" | cut -c1-160
fi
echo "=== conc chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: c=$okc Cs=$okCs C=$okC P=$okP CP=${okCP:-skipped}"
echo "node's arms done: releasing $JID"; scancel $JID
