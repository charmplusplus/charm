#!/bin/bash
# n2_spanlog11.sh (2026-10-13): validation of the one-close-per-process CUPTI window (cuptiActivityFlush(context) is
# deprecated and flushes the process: the (1,4,4) layout paid four process-wide forced flushes of 12-17 ms per LB
# event, job 22866653), the 1 MB host activity buffer (was 5 MB x one per thread, all handed back at every forced
# flush) and CUPTI's device-side profiling buffers in pinned host memory (HAPI_CUPTI_HOST_PINNED_BUFFERS, build-time).
# Binaries: sph3d = sph3d.pin (pinned, the default), sph3d.nopin (device memory). Timed arms first (close/flush/bufs
# per layout; s14 nopin and s14 at CHARM_LB_CUPTI_BUFKB=5120 isolate the two buffer changes), then 8 rounds async
# ctl/s14/s22, sync x2, alternating. Waits for job sph3d_spanlog11 RUNNING and logs/spanlog11.ready.
# Kill criterion: fault or checks mismatch. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
echo "=== $(date +%T): waiting for job sph3d_spanlog11"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_spanlog11 -h -t RUNNING -o %i | head -1); done
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N)"
w=0; until [ -f logs/spanlog11.ready ] || [ $w -ge 60 ]; do sleep 10; w=$((w+1)); done
[ -f logs/spanlog11.ready ] || { echo "=== $(date +%T) binary not marked ready after 10 min: nothing run; node KEPT"; exit 0; }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
DEVSHARE=0
summary() {  # <log> <ns>
  local L=$1 ns=$2
  DEVSHARE=$(grep -m1 -h '\[DiffusionLB\] measured loads explain' $L | sed -E 's/.*device ([0-9]+)%.*/\1/'); [ -n "$DEVSHARE" ] || DEVSHARE=0
  echo "$(date +%T) $(basename $L .log): $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | wall@25 $(grep -E '^  step +25:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') @50 $(grep -E '^  step +50:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | device-share ${DEVSHARE}% | launch $(grep -m1 -h '\[MetisLB\] measured loads explain' $L | sed -E 's/.*launch ([0-9]+)%.*/\1/')% | resume $(grep -h 'LB at step [0-9]*: resumed' $L | sed -n 2,39p | sed -E 's/.*resumed ([0-9.]+) s after.*/\1/' | awk '{s+=$1; n++} END{if(n) printf "%.3fs", s/n}') | close $(grep '^\[LBCUPTI pe=[0-9]*\] flush' $L | sed -E 's/.*total=([0-9.]+)s.*/\1/' | awk '{t+=$1; n++} END{if(n) printf "%.1f ms x%d", 1000*t/n, n}')"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|too small|CUPTI_ERROR" $L | head -2 | cut -c1-160
}
run_arm() {  # <mode> <pfx> <PPN> <GPP> <suffix> <ns> <env...>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4 sfx=$5 ns=$6; shift 6
  local pes=$((4*gpp)) np=$((2*ppn))
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx $mode $sfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}${sfx}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 LBDEBUG=1 "$@" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF > /dev/null 2>&1
  summary $D/logs/$TAG.log $ns
  LASTLOG=$D/logs/$TAG.log
}
failed() { ! grep -q "rc=0" <<< "$(grep -o "rc=[0-9]*" $LASTLOG | tail -1)" || [ "$(grep -ci "fatal\|abort\|Segmentation\|CORRUPTION\|Err 5\|CUPTI_ERROR" $LASTLOG)" -gt 0 ]; }
extras() {   # the registration and first-event fields of $LASTLOG
  echo "    startreg $(grep -c 'registered for RDMA at the mapping' $LASTLOG) lines $(grep -o 'at the mapping ([0-9.]* s' $LASTLOG | sed -E 's/.*\(//; s/ s//' | awk '{s+=$1; if($1>m)m=$1} END{printf "(sum %.1f s, max %.1f s)", s, m}') | lazyreg whole=$(grep -c 'registered once for RDMA' $LASTLOG) | metis-window $(grep -m1 'LB at step 25: resumed' $LASTLOG | sed -E 's/.*resumed ([0-9.]+) s.*/\1/')s | step25 $(grep -E '^  step +25:' $LASTLOG | sed -E 's/.*wall [0-9.]+ s, ([0-9.]+).*/\1/') ms/step | DiffLB1 at $(grep -m1 '\[DiffusionLB\] Load balancing step 1 starting at' $LASTLOG | sed -E 's/.*at ([0-9.]+).*/\1/')s"
}
flush_mean() { grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $1 | sed -E 's/.*flush=([0-9.]+)s.*/\1/' | awk '{s+=$1; n++} END{if(n) printf "%.1f", 1000*s/n; else print "NA"}'; }
timed_extras() {   # <closes per event>
  local per=$1
  grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*flush=([0-9.]+)s process=([0-9.]+)s normalize=([0-9.]+)s total=([0-9.]+)s.*wait=([0-9.]+)s retire=[0-9.]+s pass1=([0-9.]+)s pass2=[0-9.]+s merge=([0-9.]+)s snap=([0-9.]+)s\/([0-9]+)KB bufs=([0-9]+)\/([0-9]+)KB.*/\1 \2 \3 \4 \5 \6 \7 \8 \9 \10 \11/' | awk '{n++; f+=$1; p+=$2; no+=$3; t+=$4; w+=$5; a+=$6; m+=$7; sn+=$8; kb+=$9; bu+=$10; bkb+=$11; if($6>am)am=$6; if($1>fm)fm=$1} END{if(n) printf "    closes n=%d means(ms): total=%.1f flush=%.1f (max %.0f) process=%.1f wait=%.1f pass1=%.1f (max %.0f) merge=%.1f normalize=%.1f snap=%.1f (%.0f KB/close) | bufs %.1f/close holding %.0f KB\n", n, 1000*t/n, 1000*f/n, 1000*fm, 1000*p/n, 1000*w/n, 1000*a/n, 1000*am, 1000*m/n, 1000*no/n, 1000*sn/n, kb/n, bu/n, bkb/n}'
  echo "    flush max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*flush=([0-9.]+)s.*/\1/' | awk -v per=$per '{n++; if($1>p)p=$1; if(n%per==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  echo "    pass1 max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*pass1=([0-9.]+)s.*/\1/' | awk -v per=$per '{n++; if($1>p)p=$1; if(n%per==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  echo "    log sizes: $(grep -h '^\[LBCUPTI pe=[05]\] log:' $LASTLOG | sed -E 's/\[LBCUPTI pe=([0-9]+)\] log: arrival ([0-9]+) spans=([0-9]+) open=([0-9]+) bottom_open=([0-9]+) pruned=([0-9]+) gen=([0-9]+)/pe\1@\2:\3sp,o\4,b\5,g\7/' | tr '\n' ' ')"
  echo "    launch path (per close, first 2): $(grep -h 'launch path:' $LASTLOG | head -2 | sed -E 's/.*launch path: //' | tr '\n' ';')"
}
use_bin() { cp -f $D/sph3d.$1 $D/sph3d; echo "--- binary now sph3d.$1 ($(ls -la --time-style=+%H:%M $D/sph3d.$1 | awk '{print $6}'))"; }
echo "=== spanlog11 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binaries: $(ls -la --time-style=+%H:%M $D/sph3d $D/sph3d.pin $D/sph3d.nopin 2>/dev/null | awk '{print $NF" "$6}' | tr '\n' ' ')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
echo "--- 1. timed async (CHARM_LB_CUPTI_TIME=1): s14 pinned / ctl pinned / s14 nopin / s14 pinned BUFKB=5120 / s22 pinned"
use_bin pin
run_arm async s14 1 4 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 2; S14PIN=$LASTLOG
if failed; then
  echo "    FIRST ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|CUPTI_ERROR|device pool" $LASTLOG | head -8 | cut -c1-220
  run_arm async ctl 4 1 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras
  if failed; then
    echo "    control failed too: trying the nopin binary"; use_bin nopin
    run_arm async s14 1 4 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 2
    if failed; then echo "=== STARTUP FAILURE in both binaries $(date +%T): stopping; node KEPT for a fallback build"; grep -iE "fatal|abort|Err 5|CUPTI_ERROR" $LASTLOG | head -4 | cut -c1-220; exit 0; fi
    echo "    nopin runs: the rounds use sph3d.nopin"; ROUNDS=nopin
  fi
fi
run_arm async ctl 4 1 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 8
use_bin nopin
run_arm async s14 1 4 CfTnopin 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 2; S14NOPIN=$LASTLOG
use_bin pin
run_arm async s14 1 4 CfTbuf5m 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_BUFKB=5120; extras; timed_extras 2
run_arm async s22 2 2 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 4
if [ -z "$ROUNDS" ]; then
  FP=$(flush_mean $S14PIN); FN=$(flush_mean $S14NOPIN)
  echo "--- s14 flush mean: pinned $FP ms, nopin $FN ms"
  if [ "$FP" != "NA" ] && [ "$FN" != "NA" ] && awk -v a=$FP -v b=$FN 'BEGIN{exit !(b < a*0.8)}'; then ROUNDS=nopin; else ROUNDS=pin; fi
fi
echo "--- rounds use sph3d.$ROUNDS"; use_bin $ROUNDS
echo "--- 2. alternating async ctl / s14 / s22 (8 rounds)"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Cf 1000; extras; run_arm async s14 1 4 Cf 1000; extras; run_arm async s22 2 2 Cf 1000; extras; done
echo "--- 3. sync ctl / s14 / s22, twice"
for r in 1 2; do run_arm sync ctl 4 1 Cf 1000; extras; run_arm sync s14 1 4 Cf 1000; extras; run_arm sync s22 2 2 Cf 1000; extras; done
echo "--- 4. alternating async ctl / s14 while time remains"
while [ "$(left_s)" -gt 150 ]; do run_arm async ctl 4 1 Cf 1000; extras; run_arm async s14 1 4 Cf 1000; extras; done
echo "=== spanlog11 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
