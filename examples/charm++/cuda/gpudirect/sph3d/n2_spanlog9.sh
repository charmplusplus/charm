#!/bin/bash
# n2_spanlog9.sh (2026-10-12): the next two-node allocation (submitted once job 22853463 has ended; the interactive QOS
# allows one job per user), with the concurrent per-device flush (binary 18:52). Arms: alternating async ctl / s14 /
# s22 (8 rounds), sync ctl / s14 / s22, then the soft flush (existing CHARM_LB_CUPTI_SOFT_FLUSH=1: no forced flush, no
# process-wide CUDA stall at the close; check the balancer is not blind: device share, Metis moves) s14 and ctl timed,
# then alternating until the time is gone. All arms +LBDebug 1 for the lag-window lines (AtSyncStart to resume).
# Kill criterion: fault or checks mismatch. Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
until ! squeue -j 22853463 -h 2>/dev/null | grep -q .; do sleep 20; done
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:55:00 -J sph3d_spanlog9 > logs/spanlog9_salloc.out 2>&1 &
echo "=== salloc 2 nodes $(date +%T)"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_spanlog9 -h -t RUNNING -o %i | head -1); done
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N)"
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
echo "=== spanlog9 start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>&1 | grep -v '^srun'
echo "--- 1. alternating async ctl / s14 / s22"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Cf 1000; run_arm async s14 1 4 Cf 1000; run_arm async s22 2 2 Cf 1000; done
echo "--- 2. sync ctl / s14 / s22"
run_arm sync ctl 4 1 Cf 1000; run_arm sync s14 1 4 Cf 1000; run_arm sync s22 2 2 Cf 1000
echo "--- 3. soft flush (no forced flush at the close), s14 and ctl async, timed"
run_arm async s14 1 4 Sf 1000 CHARM_LB_CUPTI_SOFT_FLUSH=1 CHARM_LB_CUPTI_TIME=1
echo "    closes 2-6: $(grep -h '^\[LBCUPTI pe=0\] flush' $LASTLOG | sed -n 2,6p | sed -E 's/.*flush=/flush=/' | tr '\n' ';')"
run_arm async ctl 4 1 Sf 1000 CHARM_LB_CUPTI_SOFT_FLUSH=1 CHARM_LB_CUPTI_TIME=1
echo "    closes 2-6: $(grep -h '^\[LBCUPTI pe=0\] flush' $LASTLOG | sed -n 2,6p | sed -E 's/.*flush=/flush=/' | tr '\n' ';')"
echo "--- 4. alternating async ctl / s14 while time remains"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 8 ]; do i=$((i+1)); run_arm async ctl 4 1 Cf 1000; run_arm async s14 1 4 Cf 1000; done
echo "=== spanlog9 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
