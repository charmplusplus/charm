#!/bin/bash
# n2_fix2b.sh (07:31): adopts the salloc the 07:26 chain submitted (job lmd_fix2); the probe5 build. Original header:
# n2_fix2.sh (2026-10-19 07:26). TARGET: channels vs control on the early-credit build at steady state, same
# session, runtime defaults, layout (4,1,8). VARIABLE: -channels. COUNT: 16x8x8 async ctl x2 / ch x2
# (interleaved), 8x4x4 async ctl x2 / ch x2. KILL CRITERION: a hang (rc=124) prints its watch/probe lines and
# stops the chain (the node is KEPT for a diagnosis arm). chantest first. Never scancel from here.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
cd $L; ulimit -c 0
echo "=== adopting the pending lmd_fix2 allocation $(date +%T)"
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_fix2 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}')"
echo "--- 1. chantest (step 3, 24 PEs)"
bash $G/chantest/test_chantest.sh $JID 24 3 2>&1 | grep -E 'PASS|FAIL|rc=' | cut -c1-100 | head -8
echo "--- 2. leanmd, layout (4,1,8), runtime defaults, mean42-100 ms/step"
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
arm() {  # <channels 0|1> <grid> <tag> <tmo>
  local ch=$1 grid=$2 tag=$3 tmo=$4; [ "$(left_s)" -gt $((tmo+20)) ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env CHANNELS=$ch LB=async GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | watch=$(cat $L/ranklogs/$TAG/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|exceed')"
  if ! grep -q 'rc=0' $L/logs/$TAG.log; then echo "    STOP: $tag did not complete"; cat $L/ranklogs/$TAG/rank_*.log | grep -a 'CHAN-WATCH\|CHAN-PROBE\|CHAN-MISDELIVERED\|CHAN-STALL' | sort -u | head -24 | cut -c1-420 | sed 's/^/    /'; return 2; fi
  return 0
}
arm 0 "16 8 8" fx2_ctl_async_16x8x8 150 || exit 0
arm 1 "16 8 8" fx2_ch_async_16x8x8 150 || exit 0
arm 0 "16 8 8" fx2_ctl_async_16x8x8 150 || exit 0
arm 1 "16 8 8" fx2_ch_async_16x8x8 150 || exit 0
arm 0 "8 4 4" fx2_ctl_async_8x4x4 100 || exit 0
arm 1 "8 4 4" fx2_ch_async_8x4x4 100 || exit 0
arm 0 "8 4 4" fx2_ctl_async_8x4x4 100 || exit 0
arm 1 "8 4 4" fx2_ch_async_8x4x4 100 || exit 0
echo "=== fix2 done $(date +%T), left $(left_s) s; node KEPT"
