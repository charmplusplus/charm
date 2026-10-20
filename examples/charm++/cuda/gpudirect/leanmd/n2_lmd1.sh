#!/bin/bash
# n2_lmd1.sh (2026-10-17): leanmd under the three process/GPU splits on two A40 nodes -- does the one-process layout
# step faster where it should (tens of thousands of small same-node Cell/Compute device messages per step, ~750 Compute
# migrations per LB event)? Strong configuration of scale.sh: 16x8x8 cells (128 per GPU on 8 GPUs), 100 steps, LB period
# 20, lag 16, -computemap local -density gradient, pool 16 GB vmm. Control (4,1,8) = 4 processes x 1 GPU x 8 PEs;
# s14 = (1,4,8) one process per node driving 4 GPUs with 32 PEs; s22 = (2,2,8). No-LB first (a per-step advantage shows
# without a balancer), two rounds; then sync and async Metis+Diffusion; then alternating while time remains.
# Kill criterion for a layout: a fault/abort in its first arm (that layout is dropped, the others continue).
# Never scancel; the node is KEPT at the end.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
cd $D
echo "=== salloc 2 nodes $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:55:00 -J lmd_split1 > logs/lmd1_salloc.out 2>&1 &
echo "=== $(date +%T): waiting for job lmd_split1"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_split1 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd binary $(ls -la --time-style=+%m-%d_%H:%M $D/leanmd | awk '{print $6}')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib STEPS=100 PERIOD=20 LAG=16 GRID="16 8 8" POOL=16384 TMO=600
declare -A DROP
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
summary() {  # <tag>
  local T=$1 R=$D/ranklogs/$1/rank_0.log L=$D/logs/$1.log
  echo "$(date +%T) $T: $(grep -o 'rc=[0-9]*' $L | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-100 $(mean_steps $R 21 100) ms | mean42-100 $(mean_steps $R 42 100) ms | total $(grep -oP 'Total application time \K[0-9.]+' $R 2>/dev/null | tail -1)s | first-step $(grep -m1 '^Step 1 Benchmark Time' $R 2>/dev/null | awk '{print $5}') | migr=[$(grep -ah 'cross node migrations' $R 2>/dev/null | awk '{printf "%s ",$NF}')] | bad=$(cat $D/ranklogs/$T/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|no free CUDA IPC event slot|out of memory|CORRUPTION')"
  cat $D/ranklogs/$T/rank_*.log 2>/dev/null | grep -aiE "fatal|abort|Segmentation|no free CUDA IPC event slot|out of memory|CORRUPTION|Err 5" | head -n 2 | cut -c1-200 | sed 's/^/    /'
  grep -ah "measured loads explain" $R 2>/dev/null | head -n 2 | cut -c1-200 | sed 's/^/    /'
  grep -ah "LB at step [0-9]*: resumed" $R 2>/dev/null | sed -n 2,6p | sed -E 's/.*LB at step ([0-9]+): resumed ([0-9.]+) s after.*/\1:\2s/' | tr '\n' ' ' | sed 's/^/    resume: /; s/$/\n/'
}
run_arm() {  # <LB> <pfx> <PPN> <GPP> [sfx]
  local lb=$1 pfx=$2 ppn=$3 gpp=$4 sfx=${5:-}
  [ -n "${DROP[$pfx]:-}" ] && { echo "skip $pfx $lb: layout dropped after a fault"; return 1; }
  [ "$(left_s)" -gt 240 ] || { echo "skip $pfx $lb: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K8_16x8x8_${lb}${sfx}_$(date +%H%M%S)
  env LCIDEV=1 LB=$lb PPN=$ppn GPP=$gpp PES=$((8*gpp)) $D/run_lmd.sh $JID $TAG > /dev/null 2>&1
  summary $TAG
  LASTTAG=$TAG
  if [ "$(cat $D/ranklogs/$TAG/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION')" -gt 0 ] || ! grep -q "rc=0" $D/logs/$TAG.log; then
    echo "    FAULT in $pfx $lb: $(grep -aiE 'fatal|abort|Segmentation|CORRUPTION|error' $D/logs/$TAG.log $D/ranklogs/$TAG/rank_*.log 2>/dev/null | head -n 3 | cut -c1-200 | tr '\n' ';')"
    return 2
  fi
  return 0
}
echo "=== lmd1 start $(date +%T), left $(left_s) s"
echo "--- 1. no LB: ctl (4,1,8) / s14 (1,4,8) / s22 (2,2,8), two rounds"
for r in 1 2; do
  run_arm none ctl 4 1; [ $? -eq 2 ] && [ $r -eq 1 ] && DROP[ctl]=1
  run_arm none s14 1 4; [ $? -eq 2 ] && [ $r -eq 1 ] && DROP[s14]=1
  run_arm none s22 2 2; [ $? -eq 2 ] && [ $r -eq 1 ] && DROP[s22]=1
done
echo "--- 2. sync Metis+Diffusion: ctl / s14 / s22"
run_arm sync ctl 4 1; run_arm sync s14 1 4; run_arm sync s22 2 2
echo "--- 3. async Metis+Diffusion (lag 16): ctl / s14 / s22"
run_arm async ctl 4 1; run_arm async s14 1 4; run_arm async s22 2 2
echo "--- 4. while time remains: async ctl / s14, then no-LB ctl / s14"
while [ "$(left_s)" -gt 240 ]; do run_arm async ctl 4 1; run_arm async s14 1 4; run_arm none ctl 4 1; run_arm none s14 1 4; done
echo "=== lmd1 done $(date +%T), left $(left_s) s; node KEPT"
