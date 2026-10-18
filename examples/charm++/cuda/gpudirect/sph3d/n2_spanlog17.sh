#!/bin/bash
# n2_spanlog17.sh (2026-10-17): ONE arm per configuration (user: "one is fine"). The gate-free LB window close (E2):
# sph3d.e2 = non-forced CUPTI flush, no gate, 64 KB buffers, window stamps, per-thread coverage scaling and window
# clipping in the sweep; sph3d.v17 = the same build with the forced gated flush (the reference). Arms: timed e2 (1,4,4)
# and control (the close line's cover=/scale=/capped=, call=, total, and the balancer's device share and Metis moves
# must match the forced reference), e2 async (1,4,4)/control/(2,2,4), e2 sync (1,4,4)/control, v17 async (1,4,4) and
# control (the known +0.9 ms/step pair on this build), then the startup-registration probe: (1,4,4) with +gpupoolchunk
# 1024 and 2048 (fewer, larger RDMA registrations: 'startreg' per device against the 256 MB default), then leanmd e2
# async (1,4,8) and control (16x8x8, 100 steps). Kill criterion: a fault in the first e2 arm drops the e2 arms.
# 25-minute allocation. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
cd $D
echo "=== salloc 2 nodes, 16 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:16:00 -J sph3d_spanlog17 > logs/spanlog17_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_spanlog17 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T)"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
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
cover_line() {   # the coverage fields of the timed closes
  grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | awk '{for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]} n++; c+=v["cover"]; if(n==1||v["cover"]+0<cm)cm=v["cover"]+0; s+=v["scale"]; if(v["scale"]+0>sm)sm=v["scale"]+0; th+=v["threads"]; cp+=v["capped"]} END{if(n) printf "    coverage over %d closes: min-cover mean %.3f (worst %.2f) | max-scale mean %.3f (worst %.2f) | threads/close %.1f | capped total %d\n", n, c/n, cm, s/n, sm, th/n, cp}'
  echo "    gate waits (per close, 2nd and 3rd): $(grep -h 'launch path:' $LASTLOG | sed -n 2,3p | sed -E 's/.*gated=([0-9]+) waited=([0-9]+) \(([0-9.]+)s\).*/gated=\1 waited=\2 (\3 s)/' | tr '\n' ';')"
}
use_bin() { cp -f $D/sph3d.$1 $D/sph3d; echo "--- binary now sph3d.$1 ($(ls -la --time-style=+%H:%M $D/sph3d.$1 | awk '{print $6}'))"; }
lmd_mean() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <bin> <pfx> <PPN> <GPP>: one leanmd async arm, 16x8x8 100 steps
  local bin=$1 pfx=$2 ppn=$3 gpp=$4
  [ "$(left_s)" -gt 200 ] || { echo "skip leanmd $pfx: $(left_s) s left"; return 1; }
  cp -f $L/leanmd.$bin $L/leanmd
  local TAG=${pfx}_P${ppn}G${gpp}K8_16x8x8_async_${bin}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib STEPS=100 PERIOD=20 LAG=16 GRID="16 8 8" POOL=16384 TMO=600 LCIDEV=1 LB=async PPN=$ppn GPP=$gpp PES=$((8*gpp)) CHARM_LB_CUPTI_TIME=1 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) leanmd $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-100 $(lmd_mean $R 21 100) ms | mean42-100 $(lmd_mean $R 42 100) ms | total $(grep -oP 'Total application time \K[0-9.]+' $R 2>/dev/null | tail -1)s | migr=[$(grep -ah 'cross node migrations' $R 2>/dev/null | awk '{printf "%s ",$NF}')] | bad=$(cat $L/ranklogs/$TAG/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION')"
  grep -ah "measured loads explain" $R 2>/dev/null | head -n 2 | cut -c1-200 | sed 's/^/    /'
  grep -ah '^\[LBCUPTI pe=[0-9]*\] flush' $L/ranklogs/$TAG/rank_*.log 2>/dev/null | awk '{for(i=1;i<=NF;i++){split($i,kv,"="); v[kv[1]]=kv[2]} n++; t+=v["total"]; c+=v["call"]; cv+=v["cover"]; sc+=v["scale"]} END{if(n) printf "    closes n=%d: total %.1f ms call %.1f ms | min-cover mean %.3f max-scale mean %.3f\n", n, 1000*t/n, 1000*c/n, cv/n, sc/n}'
}
echo "=== spanlog17 start $(date +%T), job $JID, left $(left_s) s"
# One 4000-step arm per layout with the gate-free close (e2 = the default build): the 1000-step pairs resolve +-0.85
# ms/step at n=1 (sd ~0.6 per arm); four times the steps averages the per-step noise 2x. Order: (1,4,4), control, (2,2,4).
P0=(CHARM_LB_CUPTI_PERIODIC=0)
use_bin e2
echo "--- e2 async 4000 steps: s14 / ctl / s22"
[ "$(left_s)" -gt 240 ] && { run_arm async s14 1 4 Cfe2 4000 "${P0[@]}"; extras; }
[ "$(left_s)" -gt 240 ] && { run_arm async ctl 4 1 Cfe2 4000 "${P0[@]}"; extras; }
[ "$(left_s)" -gt 240 ] && { run_arm async s22 2 2 Cfe2 4000 "${P0[@]}"; extras; }
use_bin e2
echo "=== spanlog17 done $(date +%T), left $(left_s) s; node KEPT"
