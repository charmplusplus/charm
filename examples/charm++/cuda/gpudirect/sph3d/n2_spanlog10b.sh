#!/bin/bash
# n2_spanlog10b.sh (2026-10-13, relaunched without the salloc line: job 22866653 is already queued): validation of (a) pool chunk registration at the mapping (ckRegisterPoolSegments via
# hapiDevPoolSegmentHook) instead of at the first off-node transfer -- job 22856533 put ~150 x 10 ms registrations per
# process inside the (1,4,4) layout's Metis window (1.1-1.4 s vs the control's 0.6) -- and (b) the span-log prune
# compaction + end-patching refresh (the drain's first pass grew 9 -> 38 ms over the run, 6 -> 17 in the control).
# Timed arms first (CHARM_LB_CUPTI_TIME=1: close breakdown wait/pass1/snap, per-arrival log sizes), then 8 rounds async
# ctl/s14/s22, sync x2, alternating. The chain waits for logs/spanlog10.ready (binary built) before any arm.
# Kill criterion: fault or checks mismatch. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
echo "=== relaunched $(date +%T): waiting for job 22866653 (sph3d_spanlog10)"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_spanlog10 -h -t RUNNING -o %i | head -1); done
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N)"
w=0; until [ -f logs/spanlog10.ready ] || [ $w -ge 60 ]; do sleep 10; w=$((w+1)); done
[ -f logs/spanlog10.ready ] || { echo "=== $(date +%T) binary not marked ready after 10 min: nothing run; node KEPT"; exit 0; }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
DEVSHARE=0
summary() {  # <log> <ns>
  local L=$1 ns=$2
  DEVSHARE=$(grep -m1 -h '\[DiffusionLB\] measured loads explain' $L | sed -E 's/.*device ([0-9]+)%.*/\1/'); [ -n "$DEVSHARE" ] || DEVSHARE=0
  echo "$(date +%T) $(basename $L .log): $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | wall@25 $(grep -E '^  step +25:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') @50 $(grep -E '^  step +50:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | device-share ${DEVSHARE}% | resume $(grep -h 'LB at step [0-9]*: resumed' $L | sed -n 2,39p | sed -E 's/.*resumed ([0-9.]+) s after.*/\1/' | awk '{s+=$1; n++} END{if(n) printf "%.3fs", s/n}') | close $(grep '^\[LBCUPTI pe=0\] flush' $L | sed -E 's/.*total=([0-9.]+)s.*/\1/' | awk '{t+=$1; n++} END{if(n) printf "%.1f ms x%d", 1000*t/n, n}')"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|too small" $L | head -2 | cut -c1-160
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
extras() {   # the registration and first-event fields of $LASTLOG
  echo "    startreg $(grep -c 'registered for RDMA at the mapping' $LASTLOG) lines $(grep -o 'at the mapping ([0-9.]* s' $LASTLOG | sed -E 's/.*\(//; s/ s//' | awk '{s+=$1; if($1>m)m=$1} END{printf "(sum %.1f s, max %.1f s)", s, m}') | lazyreg whole=$(grep -c 'registered once for RDMA' $LASTLOG) window25-50=$(awk '/^  step +25:/{p=1} /^  step +50:/{p=0} p' $LASTLOG | grep -c 'registered once for RDMA') | metis-window $(grep -m1 'LB at step 25: resumed' $LASTLOG | sed -E 's/.*resumed ([0-9.]+) s.*/\1/')s | step25 $(grep -E '^  step +25:' $LASTLOG | sed -E 's/.*wall [0-9.]+ s, ([0-9.]+).*/\1/') ms/step"
}
timed_extras() {
  grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*process=([0-9.]+)s.*wait=([0-9.]+)s retire=[0-9.]+s pass1=([0-9.]+)s pass2=[0-9.]+s merge=([0-9.]+)s snap=([0-9.]+)s\/([0-9]+)KB.*/\1 \2 \3 \4 \5 \6/' | awk '{n++; p+=$1; w+=$2; a+=$3; m+=$4; sn+=$5; kb+=$6; if($3>am)am=$3} END{if(n) printf "    closes n=%d means(ms): process=%.1f wait=%.1f pass1=%.1f merge=%.1f snap=%.1f (%.0f KB/close) | max pass1=%.0f\n", n, 1000*p/n, 1000*w/n, 1000*a/n, 1000*m/n, 1000*sn/n, kb/n, 1000*am}'
  echo "    pass1 max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*pass1=([0-9.]+)s.*/\1/' | awk '{n++; if($1>p)p=$1; if(n%8==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  echo "    log sizes: $(grep -h '^\[LBCUPTI pe=[05]\] log:' $LASTLOG | sed -E 's/\[LBCUPTI pe=([0-9]+)\] log: arrival ([0-9]+) spans=([0-9]+) open=([0-9]+) bottom_open=([0-9]+) pruned=([0-9]+) gen=([0-9]+)/pe\1@\2:\3sp,o\4,b\5,g\7/' | tr '\n' ' ')"
}
echo "=== spanlog10 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
echo "--- 1. timed async s14 / ctl / s22 (CHARM_LB_CUPTI_TIME=1)"
run_arm async s14 1 4 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras
if ! grep -q "rc=0" <<< "$(grep -o "rc=[0-9]*" $LASTLOG | tail -1)" || [ "$(grep -ci "fatal\|abort\|Segmentation\|CORRUPTION\|Err 5" $LASTLOG)" -gt 0 ]; then echo "    FIRST ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|registered for RDMA at the mapping|device pool" $LASTLOG | head -8 | cut -c1-220; run_arm async ctl 4 1 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; if ! grep -q "rc=0" <<< "$(grep -o "rc=[0-9]*" $LASTLOG | tail -1)" || [ "$(grep -ci "fatal\|abort\|Segmentation\|CORRUPTION\|Err 5" $LASTLOG)" -gt 0 ]; then echo "=== STARTUP FAILURE in both layouts $(date +%T): stopping; node KEPT for a fallback build"; grep -iE "fatal|abort|Err 5" $LASTLOG | head -4 | cut -c1-220; exit 0; fi; fi
run_arm async ctl 4 1 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras
run_arm async s22 2 2 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras
echo "--- 2. alternating async ctl / s14 / s22 (8 rounds)"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Cf 1000; extras; run_arm async s14 1 4 Cf 1000; extras; run_arm async s22 2 2 Cf 1000; extras; done
echo "--- 3. sync ctl / s14 / s22, twice"
for r in 1 2; do run_arm sync ctl 4 1 Cf 1000; extras; run_arm sync s14 1 4 Cf 1000; extras; run_arm sync s22 2 2 Cf 1000; extras; done
echo "--- 4. alternating async ctl / s14 while time remains"
while [ "$(left_s)" -gt 150 ]; do run_arm async ctl 4 1 Cf 1000; extras; run_arm async s14 1 4 Cf 1000; extras; done
echo "=== spanlog10 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
