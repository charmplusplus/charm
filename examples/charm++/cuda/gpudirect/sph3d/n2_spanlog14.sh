#!/bin/bash
# n2_spanlog14.sh (2026-10-17): runs on job 22953167 after chain 13 was stopped past its period-50 arms (the periodic
# worker flush did not shorten the forced flush call: 15.9 ms with it, 14.4 without; the arrivals line showed the call
# is a 2-42 ms wait BEFORE the first buffer, then ~0.3 ms per buffer). Two diagnostic binaries: sph3d.sync14 (timed
# arms cudaDeviceSynchronize every local device inside the gate before the flush and print sync= apart from call=: if
# the wait is the GPU queue drain, it moves into sync=), sph3d.soft14 (the NON-forced flush at the close: its call time
# and what it delivers, CHARM_LB_CUPTI_BUFKB=64 so buffers fill within the window; the balancer is blind to the open
# buffers so that arm's step time is not a result). Then period-50 rounds and alternating async ctl/s14 with the
# default binary (sph3d.v14 = v13 + arrivals printed at closes 20-22) for the paired statistic. CHARM_LB_CUPTI_PERIODIC=0
# everywhere (the period changed nothing). Takes the running job id as $1. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
JID=$1; [ -n "$JID" ] || { echo "usage: n2_spanlog14.sh <jobid>"; exit 1; }
echo "=== $(date +%T) spanlog14 on job $JID nodes $(squeue -j $JID -h -o %N), left $(squeue -j $JID -h -o %L)"
w=0; until [ -f logs/spanlog14.ready ] || [ $w -ge 60 ]; do sleep 10; w=$((w+1)); done
[ -f logs/spanlog14.ready ] || { echo "=== $(date +%T) binaries not marked ready after 10 min: nothing run; node KEPT"; exit 0; }
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
use_bin() { cp -f $D/sph3d.$1 $D/sph3d; echo "--- binary now sph3d.$1 ($(ls -la --time-style=+%H:%M $D/sph3d.$1 | awk '{print $6}'))"; }
while [ -n "$(squeue -j $JID -s -h -o %i 2>/dev/null | grep -v extern)" ]; do sleep 5; done
echo "=== spanlog14 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binaries: $(ls -la --time-style=+%H:%M $D/sph3d.sync14 $D/sph3d.soft14 $D/sph3d.v14 | awk '{print $NF" "$6}' | tr '\n' ' ')"
P0=(CHARM_LB_CUPTI_PERIODIC=0)
echo "--- 1. timed (CHARM_LB_CUPTI_TIME=1): s14 sync / s14 soft (BUFKB=64) / ctl sync / ctl soft / s22 sync"
use_bin sync14
run_arm async s14 1 4 CfTsync 1000 CHARM_LB_CUPTI_TIME=1 "${P0[@]}"; extras; timed_extras 2
if failed; then echo "    SYNC ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|CUPTI_ERROR" $LASTLOG | head -6 | cut -c1-220; fi
use_bin soft14
run_arm async s14 1 4 CfTsoft 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_BUFKB=64 "${P0[@]}"; extras; timed_extras 2
if failed; then echo "    SOFT ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|CUPTI_ERROR" $LASTLOG | head -6 | cut -c1-220; fi
use_bin sync14
run_arm async ctl 4 1 CfTsync 1000 CHARM_LB_CUPTI_TIME=1 "${P0[@]}"; extras; timed_extras 8
use_bin soft14
run_arm async ctl 4 1 CfTsoft 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_BUFKB=64 "${P0[@]}"; extras; timed_extras 8
use_bin sync14
run_arm async s22 2 2 CfTsync 1000 CHARM_LB_CUPTI_TIME=1 "${P0[@]}"; extras; timed_extras 4
use_bin v14
echo "--- 2. LB period 50 (LBF=50 LBB=50), async, 2 rounds (default binary)"
for r in 1 2; do
  run_arm async ctl 4 1 CfP50 1000 LBF=50 LBB=50 "${P0[@]}"; extras; windows50
  run_arm async s14 1 4 CfP50 1000 LBF=50 LBB=50 "${P0[@]}"; extras; windows50
  run_arm async s22 2 2 CfP50 1000 LBF=50 LBB=50 "${P0[@]}"; extras; windows50
done
echo "--- 3. alternating async ctl / s14 while time remains (default binary)"
while [ "$(left_s)" -gt 150 ]; do run_arm async ctl 4 1 Cf 1000 "${P0[@]}"; extras; run_arm async s14 1 4 Cf 1000 "${P0[@]}"; extras; done
echo "=== spanlog14 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
