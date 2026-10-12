#!/bin/bash
# n2_rest.sh <jobid>: the rest of job 22797719 after the -c pup fix (2026-10-10): kernels-only sync trace (user: "capture a
# sync LB trace as well"), -c async smoke, -c async 2000, sync control 2000, -c sync 2000, kernels-only traces of -c in
# both modes. Guards on the time left; releases the node at the end.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() {  # <pfx> <steps> <mode sync|async> <extra app args> <env...>
  local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4
  local need=$(( ns > 500 ? 150 : 60 ))
  [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  env "$@" DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "CORRUPTION|Abort|abort|Fatal|Segmentation|did not consume|overflowed|expected|ERROR" $L | head -3 | cut -c1-220
  grep -A12 "^Message latency" $L | grep -E "halo|migration" | head -6
  grep -q "rc=0" $L
}
trace() {  # <mode> <extra app args>
  [ "$(left_s)" -gt 70 ] || { echo "skip trace $1 $2: $(left_s) s left"; return 1; }
  GT_NOCOPY=1 XTRA_APP="$2" bash $D/n2_gputrace.sh $JID 200 $1 2>&1 | grep -v "^HAPI\[" | cut -c1-200
}
echo "=== rest chain start $(date +%T), left $(squeue -j $JID -h -o %L)"
trace sync ""
run_arm Cs 200 async "-c"; okCs=$?
if [ $okCs = 0 ]; then run_arm C 2000 async "-c"; okC=$?; else echo "C smoke failed again: no long -c arms"; okC=1; fi
run_arm s 2000 sync ""; oks=$?
[ $okC = 0 ] && { run_arm Sc 2000 sync "-c"; okSc=$?; }
[ $okC = 0 ] && trace async "-c"
[ "${okSc:-1}" = 0 ] && trace sync "-c"
wait
for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_* 2>/dev/null | head -4); do echo "== $(basename $T)"; python3 $D/gap_report.py $T/trace_last1s.json 5 2>&1 | grep -E "message latency|halo |migration |largest gap" | cut -c1-160; done
echo "=== rest chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: Cs=$okCs C=$okC s=$oks Sc=${okSc:-skipped}"
echo "=== node KEPT: release only when the user says so"
