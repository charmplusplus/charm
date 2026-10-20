#!/bin/bash
# n2_devclose2.sh (2026-10-12): after the per-device window-close chain (devclose_chain.out) on the same two nodes,
# the arms that separate the one-process layout's steady-state penalty (s14 async 42-45 vs ctl 37-40 ms/step at two
# nodes; equal without LB) into its parts, with the binary the chain ran (no rebuild):
#   1. LB without CUPTI (CHARM_LB_NO_CUPTI=1): s14 and ctl async. If the gap vanishes, the cost is the tracing.
#   2. +LBDebug 2 with the close timing: per-call runtime API breakdown (HAPI[pe] shard N: runtime API by call) and
#      the launch-path tallies (gated/waited/pushes) per device, s14 and ctl async.
#   3. +gputrace of the last second, s14 and ctl async, 300 steps: GPU busy %, gaps, message latency by tier.
#   4. Alternating plain async ctl/s14 while time remains.
# Kill criterion: fault or checks mismatch. Keeps the node.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
until grep -q "^=== done" $D/logs/devclose_chain.out 2>/dev/null; do sleep 10; done
JID=$(grep -o 'jobid=[0-9]*' $D/logs/devclose_chain.out | head -1 | cut -d= -f2)
[ -n "$JID" ] || { echo "no jobid in devclose_chain.out"; exit 1; }
cd $D
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
summary() {  # <log> <ns>
  local L=$1 ns=$2
  echo "$(date +%T) $(basename $L .log): $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | wall@25 $(grep -E '^  step +25:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') @50 $(grep -E '^  step +50:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | close $(grep '^\[LBCUPTI pe=0\] flush' $L | sed -E 's/.*total=([0-9.]+)s.*/\1/' | awk '{t+=$1; n++} END{if(n) printf "%.1f ms x%d", 1000*t/n, n}')"
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
echo "=== devclose2 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
echo "--- 1. LB without CUPTI"
run_arm async s14 1 4 Nc 1000 CHARM_LB_NO_CUPTI=1 LBDEBUG=1
run_arm async ctl 4 1 Nc 1000 CHARM_LB_NO_CUPTI=1 LBDEBUG=1
echo "--- 2. per-call API breakdown + launch-path tallies"
run_arm async s14 1 4 Dbg 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=2
S14DBG=$LASTLOG
run_arm async ctl 4 1 Dbg 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=2
CTLDBG=$LASTLOG
for L in $S14DBG $CTLDBG; do
  echo "  $(basename $L .log):"
  echo "    attach: $(grep -h 'LBCUPTI pe=[0-9]*\] attach' $L | head -2 | sed -E 's/.*attach ([0-9.]+)s.*/\1/' | tr '\n' ' ')"
  echo "    launch path (first 3 closes on PE 0): $(grep -h '^\[LBCUPTI pe=0\]   records' $L | head -3 | sed -E 's/.*records //' | tr '\n' ';')"
  echo "    API by call (PE 0, first shard line of events 2-3):"; grep -h '^HAPI\[pe=0\] shard 0: runtime API by call' $L | sed -n 2,3p | cut -c1-330
  echo "    API by call (another device's rep, if any):"; grep -h 'shard 0: runtime API by call' $L | grep -v 'pe=0\]' | sed -n 2p | cut -c1-330
done
echo "--- 3. GPU timeline of the last second (300 steps)"
declare -A OUTS
for p in s14 ctl; do
  if [ $p = s14 ]; then ppn=1 gpp=4; else ppn=4 gpp=1; fi
  OUT=/scratch/mzu/bhosale/gputrace/gk2n_${p}_$(date +%H%M%S); mkdir -p $OUT; OUTS[$p]=$OUT
  run_arm async $p $ppn $gpp Tr 300 -- +gputrace $OUT +gputracenocopy
  echo "    files: $(ls $OUT | grep -c '\.proc$') proc; $(python3 $D/gputrace_merge.py --out $OUT/trace_last1s.json --last 1.0 $OUT 2>&1 | tail -1 | cut -c1-160)"
done
if [ -f ${OUTS[ctl]}/trace_last1s.json ] && [ -f ${OUTS[s14]}/trace_last1s.json ]; then
  echo "--- busy % and gaps (all GPUs, last 1 s):"
  python3 $D/trace_compare.py ctl=${OUTS[ctl]}/trace_last1s.json s14=${OUTS[s14]}/trace_last1s.json 2>&1 | grep -vE "^\s*$" | head -60 | cut -c1-200
  for p in ctl s14; do echo "--- $p gaps:"; python3 $D/gap_report.py ${OUTS[$p]}/trace_last1s.json 5 2>&1 | head -30 | cut -c1-200; done
  echo "dirs: ${OUTS[ctl]} ${OUTS[s14]}"
fi
echo "--- 4. alternating plain async"
i=0; while [ "$(left_s)" -gt 160 ] && [ $i -lt 6 ]; do i=$((i+1)); if [ $((i % 2)) = 1 ]; then run_arm async ctl 4 1 Dc2 1000; else run_arm async s14 1 4 Dc2 1000; fi; done
echo "=== devclose2 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
