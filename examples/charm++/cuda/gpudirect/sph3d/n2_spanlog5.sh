#!/bin/bash
# n2_spanlog5.sh (2026-10-12): on job 22853463 after the periodic arms proved dead (CUPTI delivered nothing: device share 0%),
# hand), the +gputrace timeline of the last second for the control and the one-process layout with the span-log binary:
# GPU busy %, gaps by size and by the kernel that ended them, message latency by tier, the per-PE entry times during
# the largest gap. Then alternating plain async while time remains. Never scancel; node KEPT.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
JID=22853463
[ -n "$JID" ] || { echo "no jobid"; exit 1; }
cd $D
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
summary() {  # <log> <ns>
  local L=$1 ns=$2
  echo "$(date +%T) $(basename $L .log): $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | wall@25 $(grep -E '^  step +25:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') @50 $(grep -E '^  step +50:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$')"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|too small" $L | head -2 | cut -c1-160
}
run_arm() {  # <mode> <pfx> <PPN> <GPP> <suffix> <ns> <env...> -- <extra app args>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4 sfx=$5 ns=$6; shift 6
  local envs=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done; [ "$1" = "--" ] && shift
  local pes=$((4*gpp)) np=$((2*ppn)); local need=$(( ns >= 1000 ? 150 : 90 ))
  [ "$(left_s)" -gt $need ] || { echo "skip $pfx $mode $sfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}${sfx}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 "${envs[@]}" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF "$@" > /dev/null 2>&1
  summary $D/logs/$TAG.log $ns
  LASTLOG=$D/logs/$TAG.log
}
until ! ps -u bhosale -o cmd | grep -q "[r]un3d.sh 22853463"; do sleep 5; done   # the orphaned arm of the stopped chain
echo "=== spanlog5 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
echo "--- 1. GPU timeline of the last second (300 steps), ctl then s14"
declare -A OUTS
for p in ctl s14; do
  if [ $p = s14 ]; then ppn=1 gpp=4; else ppn=4 gpp=1; fi
  OUT=/scratch/mzu/bhosale/gputrace/sl5_${p}_$(date +%H%M%S); mkdir -p $OUT; OUTS[$p]=$OUT
  run_arm async $p $ppn $gpp Tr5 300 -- +gputrace $OUT +gputracenocopy
  echo "    files: $(ls $OUT | grep -c '\.proc$') proc; $(python3 $D/gputrace_merge.py --out $OUT/trace_last1s.json --last 1.0 $OUT 2>&1 | tail -1 | cut -c1-160)"
done
if [ -f ${OUTS[ctl]}/trace_last1s.json ] && [ -f ${OUTS[s14]}/trace_last1s.json ]; then
  echo "--- busy % and gaps (all GPUs, last 1 s):"
  python3 $D/trace_compare.py ctl=${OUTS[ctl]}/trace_last1s.json s14=${OUTS[s14]}/trace_last1s.json 2>&1 | grep -vE "^\s*$" | head -60 | cut -c1-220
  for p in ctl s14; do echo "--- $p gaps:"; python3 $D/gap_report.py ${OUTS[$p]}/trace_last1s.json 5 2>&1 | head -34 | cut -c1-220; done
  echo "dirs: ${OUTS[ctl]} ${OUTS[s14]}"
fi
echo "--- 2. sync ctl / s14 / s22"
run_arm sync ctl 4 1 Sl5 1000; run_arm sync s14 1 4 Sl5 1000; run_arm sync s22 2 2 Sl5 1000
echo "--- 3. alternating plain async ctl / s14 / s22 while time remains"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Sl5 1000; run_arm async s14 1 4 Sl5 1000; run_arm async s22 2 2 Sl5 1000; done
echo "=== spanlog5 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
