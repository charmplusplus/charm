#!/bin/bash
# n2_spanlog15.sh <jobid> (2026-10-17): the GPU timeline of an LB event in the (1,4,4) layout against the control.
# Chain 13's period-50 windows put the (1,4,4) event at ~30 ms of step time per event and the control's at ~0, with
# the window close itself 28 vs 11 ms (flush call 16 vs 6, parse 12 vs 5, all on the helper thread). Two kernel-only
# +gputracenocopy arms (async, LB every 25 steps, default binary sph3d.v14, CHARM_LB_CUPTI_PERIODIC=0), merged over
# the last 2.5 s (~2-3 events) and run through gap_report.py: per rank and GPU, the kernel gaps > 3 ms and what the
# PEs were doing in the largest. Then (2,2,4) traced if time allows, then alternating async ctl/s14 for the paired
# statistic. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
JID=$1; [ -n "$JID" ] || { echo "usage: n2_spanlog15.sh <jobid>"; exit 1; }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
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
timed_extras() {   # <closes per event>
  local per=$1
  grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | awk '{for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]} n++; f+=v["flush"]; g+=v["gate"]; sy+=v["sync"]; if(v["sync"]+0>sym)sym=v["sync"]+0; c+=v["call"]; if(v["call"]+0>cm)cm=v["call"]+0; p+=v["process"]; no+=v["normalize"]; t+=v["total"]; w+=v["wait"]; a+=v["pass1"]; if(v["pass1"]+0>am)am=v["pass1"]+0; m+=v["merge"]; split(v["snap"],sn,"/"); snt+=sn[1]; snk+=sn[2]; split(v["bufs"],bu,"/"); bun+=bu[1]; buk+=bu[2]; split(v["cbs"],cb,"/"); cbn+=cb[1]; cbk+=cb[2]} END{if(n) printf "    closes n=%d means(ms): total=%.1f flush=%.1f [gate=%.1f sync=%.1f (max %.0f) call=%.1f (max %.0f)] process=%.1f wait=%.1f pass1=%.1f (max %.0f) merge=%.1f normalize=%.1f snap=%.1f (%.0f KB/close) | bufs %.1f/close holding %.0f KB | flush delivered %.1f buffers/%.0f KB\n", n, 1000*t/n, 1000*f/n, 1000*g/n, 1000*sy/n, 1000*sym, 1000*c/n, 1000*cm, 1000*p/n, 1000*w/n, 1000*a/n, 1000*am, 1000*m/n, 1000*no/n, 1000*snt/n, snk/n, bun/n, buk/n, cbn/n, cbk/n}'
  echo "    sync max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.* sync=([0-9.]+)s.*/\1/' | awk -v per=$per '{n++; if($1>p)p=$1; if(n%per==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  echo "    call max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.* call=([0-9.]+)s.*/\1/' | awk -v per=$per '{n++; if($1>p)p=$1; if(n%per==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  grep -h 'flush call .* arrivals:' $LASTLOG | sed -n '2p;5p;6p' | cut -c1-420 | sed 's/^/    /'
  echo "    records (closes 2-4): $(grep -h 'records api=' $LASTLOG | sed -n 2,4p | sed -E 's/.*records (api=[0-9]+ kernels=[0-9]+).*/\1/' | tr '\n' ' ')"
  echo "    launch path (per close, 2nd and 3rd): $(grep -h 'launch path:' $LASTLOG | sed -n 2,3p | sed -E 's/.*launch path: //' | tr '\n' ';')"
}
windows50() {   # $LASTLOG: mean ms/step of the windows without an LB event (ending at steps = 0 mod 50, from 100) and with one (= 25 mod 50, from 125)
  grep -E '^  step +[0-9]+:' $LASTLOG | awk '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0; if (match($0,/ ([0-9.]+) ms\/step/,x)) { if (st>=100 && st%50==0) {c+=x[1]; cn++} else if (st>=125 && st%50==25) {e+=x[1]; en++} }} END{printf "    period-50 windows: clean %.2f ms/step (n=%d), event+lag %.2f (n=%d)\n", cn?c/cn:0, cn, en?e/en:0, en}'
}
# run_arm with extra application arguments ($XTRA_APP, e.g. the trace flags) appended after the LB flags
run_arm_x() {  # <mode> <pfx> <PPN> <GPP> <suffix> <ns> <env...>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4 sfx=$5 ns=$6; shift 6
  local pes=$((4*gpp)) np=$((2*ppn))
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx $mode $sfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}${sfx}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 LBDEBUG=1 "$@" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF $XTRA_APP > /dev/null 2>&1
  summary $D/logs/$TAG.log $ns
  LASTLOG=$D/logs/$TAG.log
}
trace_arm() {  # <pfx> <PPN> <GPP>: one kernel-only traced async arm, merged and gap-reported
  local pfx=$1 ppn=$2 gpp=$3
  local OUT=/scratch/mzu/bhosale/gputrace/gk15_${pfx}_P${ppn}G${gpp}_$(date +%H%M%S); mkdir -p $OUT
  XTRA_APP="+gputracenocopy +gputrace $OUT" run_arm_x async $pfx $ppn $gpp CfGT 1000 CHARM_LB_CUPTI_PERIODIC=0; extras
  echo "    trace files: $(ls $OUT | wc -l) in $OUT"
  ( cd $D && timeout 600 python3 gputrace_merge.py --out $OUT/trace_last2.5s.json --last 2.5 $OUT 2>&1 | tail -n 6 | sed 's/^/    merge: /' )
  [ -s $OUT/trace_last2.5s.json ] && { echo "    --- gap report ($pfx), gaps > 3 ms, per rank/GPU; then the largest gap's PE picture:"; timeout 300 python3 gap_report.py $OUT/trace_last2.5s.json 3 2>&1 | head -n 60 | cut -c1-240 | sed 's/^/    /'; }
  echo "    LB events in the traced window (resume lines, last 4): $(grep -h 'LB at step [0-9]*: resumed' $LASTLOG | tail -n 4 | sed -E 's/.*LB at step ([0-9]+): resumed ([0-9.]+) s after.*/\1:\2s/' | tr '\n' ' ')"
}
use_bin() { cp -f $D/sph3d.$1 $D/sph3d; echo "--- binary now sph3d.$1 ($(ls -la --time-style=+%H:%M $D/sph3d.$1 | awk '{print $6}'))"; }
while [ -n "$(squeue -j $JID -s -h -o %i 2>/dev/null | grep -v extern)" ]; do sleep 5; done
echo "=== spanlog15 start $(date +%T), job $JID, left $(left_s) s, end $(date -d @$END_EPOCH +%T)"
P0=(CHARM_LB_CUPTI_PERIODIC=0)
echo "--- 1. kernel-only GPU timelines with sph3d.v14: ctl then s14"
use_bin v14
trace_arm ctl 4 1
trace_arm s14 1 4
echo "--- 2. timed (CHARM_LB_CUPTI_TIME=1) with sph3d.v15 (CUPTI told our buffers are zeroed; recycled buffers zeroed on the helper): s14 / ctl"
use_bin v15
run_arm async s14 1 4 CfTz 1000 CHARM_LB_CUPTI_TIME=1 "${P0[@]}"; extras; timed_extras 2
if failed; then echo "    V15 ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|CUPTI_ERROR" $LASTLOG | head -6 | cut -c1-220; use_bin v14; fi
run_arm async ctl 4 1 CfTz 1000 CHARM_LB_CUPTI_TIME=1 "${P0[@]}"; extras; timed_extras 8
echo "--- 3. alternating async ctl / s14 while time remains ($(cmp -s $D/sph3d $D/sph3d.v15 && echo v15 || echo v14))"
while [ "$(left_s)" -gt 150 ]; do run_arm async ctl 4 1 Cf 1000 "${P0[@]}"; extras; run_arm async s14 1 4 Cf 1000 "${P0[@]}"; extras; done
echo "=== spanlog15 done $(date +%T), left $(left_s) s; node KEPT"
