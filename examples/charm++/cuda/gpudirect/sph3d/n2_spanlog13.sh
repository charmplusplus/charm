#!/bin/bash
# n2_spanlog13.sh (2026-10-17): the writer-preferring gates were validated and falsified as the cause (job 22868681:
# gate=0.0 everywhere, the cuptiActivityFlushAll CALL itself is 14.4 ms in the (1,4,4) process vs 5.2 in the control,
# 11.5 in (2,2,4); the (1,4,4) async layout was +1.00 +-0.33 ms/step over 22 paired rounds). Hypothesis under test:
# the call's cost is the device-buffer readback/record completion CUPTI's worker thread had not done, so a fixed
# worker flush period (cuptiActivityFlushPeriod, default 10 ms since this build; CHARM_LB_CUPTI_PERIODIC=0 restores
# the heuristics) moves it between closes and the forced flush at the close gets cheap in every layout. The binary
# also carries the parallel lane snapshot and the parallel per-device sweep (normalize) and the per-kind gated-call
# tallies. The timed closes print the arrival time of every buffer within the flush call (the call's structure).
# Timed arms (default / PERIODIC=0) decide the rounds' setting; then LB-period-50 arms x3 (clean vs event windows),
# 8 async rounds ctl/s14/s22, sync x2, alternating ctl/s14. One binary (sph3d = sph3d.v13).
# Kill criterion: fault or checks mismatch. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
echo "=== salloc 2 nodes $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:55:00 -J sph3d_spanlog13 > logs/spanlog13_salloc.out 2>&1 &
echo "=== $(date +%T): waiting for job sph3d_spanlog13"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_spanlog13 -h -t RUNNING -o %i | head -1); done
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N)"
w=0; until [ -f logs/spanlog13.ready ] || [ $w -ge 120 ]; do sleep 10; w=$((w+1)); done
[ -f logs/spanlog13.ready ] || { echo "=== $(date +%T) binary not marked ready after 20 min: nothing run; node KEPT"; exit 0; }
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
flush_mean() { grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $1 | sed -E 's/.* call=([0-9.]+)s.*/\1/' | awk '{s+=$1; n++} END{if(n) printf "%.1f", 1000*s/n; else print "NA"}'; }
timed_extras() {   # <closes per event>
  local per=$1
  grep -h '^HAPI> CUPTI worker flush period' $LASTLOG | head -1 | sed 's/^/    /'
  grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | awk '{for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]} n++; f+=v["flush"]; g+=v["gate"]; c+=v["call"]; if(v["gate"]+0>gm)gm=v["gate"]+0; if(v["call"]+0>cm)cm=v["call"]+0; p+=v["process"]; no+=v["normalize"]; t+=v["total"]; w+=v["wait"]; a+=v["pass1"]; if(v["pass1"]+0>am)am=v["pass1"]+0; m+=v["merge"]; split(v["snap"],sn,"/"); snt+=sn[1]; snk+=sn[2]; split(v["bufs"],bu,"/"); bun+=bu[1]; buk+=bu[2]; split(v["cbs"],cb,"/"); cbn+=cb[1]; cbk+=cb[2]} END{if(n) printf "    closes n=%d means(ms): total=%.1f flush=%.1f [gate=%.1f (max %.0f) call=%.1f (max %.0f)] process=%.1f wait=%.1f pass1=%.1f (max %.0f) merge=%.1f normalize=%.1f snap=%.1f (%.0f KB/close) | bufs %.1f/close holding %.0f KB | flush delivered %.1f buffers/%.0f KB\n", n, 1000*t/n, 1000*f/n, 1000*g/n, 1000*gm, 1000*c/n, 1000*cm, 1000*p/n, 1000*w/n, 1000*a/n, 1000*am, 1000*m/n, 1000*no/n, 1000*snt/n, snk/n, bun/n, buk/n, cbn/n, cbk/n}'
  echo "    call max per event (ms), events 1-4 ... last 4: $(grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.* call=([0-9.]+)s.*/\1/' | awk -v per=$per '{n++; if($1>p)p=$1; if(n%per==0){ev++; v[ev]=p; p=0}} END{for(i=1;i<=4&&i<=ev;i++) printf "%.0f ", 1000*v[i]; printf "... "; for(i=ev-3;i<=ev;i++) if(i>4) printf "%.0f ", 1000*v[i]}')"
  grep -h 'flush call .* arrivals:' $LASTLOG | sed -n 2,3p | cut -c1-420 | sed 's/^/    /'
  echo "    log sizes: $(grep -h '^\[LBCUPTI pe=[05]\] log:' $LASTLOG | sed -E 's/\[LBCUPTI pe=([0-9]+)\] log: arrival ([0-9]+) spans=([0-9]+) open=([0-9]+) bottom_open=([0-9]+) pruned=([0-9]+) gen=([0-9]+)/pe\1@\2:\3sp,o\4,b\5,g\7/' | tr '\n' ' ')"
  echo "    launch path (per close, 2nd and 3rd): $(grep -h 'launch path:' $LASTLOG | sed -n 2,3p | sed -E 's/.*launch path: //' | tr '\n' ';')"
}
windows50() {   # $LASTLOG: mean ms/step of the windows without an LB event (ending at steps = 0 mod 50, from 100) and with one (= 25 mod 50, from 125)
  grep -E '^  step +[0-9]+:' $LASTLOG | awk '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0; if (match($0,/ ([0-9.]+) ms\/step/,x)) { if (st>=100 && st%50==0) {c+=x[1]; cn++} else if (st>=125 && st%50==25) {e+=x[1]; en++} }} END{printf "    period-50 windows: clean %.2f ms/step (n=%d), event+lag %.2f (n=%d)\n", cn?c/cn:0, cn, en?e/en:0, en}'
}
echo "=== spanlog13 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binary: $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $NF" "$6}') $(cmp -s $D/sph3d $D/sph3d.v13 && echo '(= sph3d.v13)' || echo '(DIFFERS from sph3d.v13)')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
echo "--- 1. timed async (CHARM_LB_CUPTI_TIME=1): s14 default (period 10) / ctl default / s14 PERIODIC=0 / ctl PERIODIC=0 / s22 default"
run_arm async s14 1 4 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 2; S14P=$LASTLOG
if failed; then
  echo "    FIRST ARM FAILED:"; grep -iE "fatal|abort|Segmentation|Err 5|CUPTI_ERROR|device pool" $LASTLOG | head -8 | cut -c1-220
  run_arm async s14 1 4 CfTnoper 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_PERIODIC=0; extras; timed_extras 2
  if failed; then echo "=== STARTUP FAILURE with and without the periodic flush $(date +%T): stopping; node KEPT for a fallback build"; grep -iE "fatal|abort|Err 5|CUPTI_ERROR" $LASTLOG | head -4 | cut -c1-220; exit 0; fi
  echo "    PERIODIC=0 runs: the rounds use it"; PER=(CHARM_LB_CUPTI_PERIODIC=0); S14N=$LASTLOG
fi
run_arm async ctl 4 1 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 8
if [ -z "$S14N" ]; then run_arm async s14 1 4 CfTnoper 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_PERIODIC=0; extras; timed_extras 2; S14N=$LASTLOG; fi
run_arm async ctl 4 1 CfTnoper 1000 CHARM_LB_CUPTI_TIME=1 CHARM_LB_CUPTI_PERIODIC=0; extras; timed_extras 8
run_arm async s22 2 2 CfT 1000 CHARM_LB_CUPTI_TIME=1; extras; timed_extras 4
if [ ${#PER[@]} -eq 0 ]; then
  FP=$(flush_mean $S14P); FN=$(flush_mean $S14N)
  echo "--- s14 flush CALL mean: periodic 10 ms $FP ms, PERIODIC=0 $FN ms"
  if [ "$FP" != "NA" ] && [ "$FN" != "NA" ] && awk -v a=$FP -v b=$FN 'BEGIN{exit !(a < b*0.8)}'; then PER=(); echo "--- rounds use the default (periodic 10 ms)"; else PER=(CHARM_LB_CUPTI_PERIODIC=0); echo "--- periodic flush did not cut the call: rounds use CHARM_LB_CUPTI_PERIODIC=0"; fi
fi
echo "--- 1b. LB period 50 (LBF=50 LBB=50), async, 3 rounds: the windows without an event measure the step cost per layout"
for r in 1 2 3; do
  run_arm async ctl 4 1 CfP50 1000 LBF=50 LBB=50 "${PER[@]}"; extras; windows50
  run_arm async s14 1 4 CfP50 1000 LBF=50 LBB=50 "${PER[@]}"; extras; windows50
  run_arm async s22 2 2 CfP50 1000 LBF=50 LBB=50 "${PER[@]}"; extras; windows50
done
echo "--- 2. alternating async ctl / s14 / s22 (8 rounds)"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Cf 1000 "${PER[@]}"; extras; run_arm async s14 1 4 Cf 1000 "${PER[@]}"; extras; run_arm async s22 2 2 Cf 1000 "${PER[@]}"; extras; done
echo "--- 3. sync ctl / s14 / s22, twice"
for r in 1 2; do run_arm sync ctl 4 1 Cf 1000 "${PER[@]}"; extras; run_arm sync s14 1 4 Cf 1000 "${PER[@]}"; extras; run_arm sync s22 2 2 Cf 1000 "${PER[@]}"; extras; done
echo "--- 4. alternating async ctl / s14 while time remains"
while [ "$(left_s)" -gt 150 ]; do run_arm async ctl 4 1 Cf 1000 "${PER[@]}"; extras; run_arm async s14 1 4 Cf 1000 "${PER[@]}"; extras; done
echo "=== spanlog13 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
