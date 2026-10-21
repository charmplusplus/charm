#!/bin/bash
# conn1_chain.sh <jobid>: one A40 node. Validates the runtime-set CUDA_DEVICE_MAX_CONNECTIONS=32 (hapiPresetEnvironment
# from charm_main, 2026-10-20) and re-measures channels vs zero-copy under async LB with it in force.
# TARGET: the 2026-10-20 leanmd 8x4x4 async bands (8 queues: zc 225-233, ch 206-211) at 32 queues.
# ONE VARIABLE: the hardware queue count (runtime default 32 vs CONNECTIONS=8 in the environment).
# KILL CRITERION: connpreset "before" reports effective_queues != 32, or the first leanmd banner at the runtime default
# says anything but "32 hardware queue(s) per device (runtime default)" => the preset does not reach the driver; the
# chain stops and says so. It never sallocs and never scancels: read the results, then release.
#  1. connpreset none|before|after|context on one GPU: where an in-process setenv must sit relative to cuInit.
#  2. chantest, all 8 arms (must PASS as on 2026-10-20; nothing else in the runtime changed).
#  3. leanmd 8x4x4 async, mean42-100 ms/step: zc x3, ch x3 at the runtime default (a band each), then CONNECTIONS=8
#     zc x1, ch x1 (banner "from the environment") as the same-session 8-queue reference.
#  4. leanmd 16x8x8 async: zc x2, ch x2 at the runtime default. No one-node reference exists: the 2026-10-19 16x8x8
#     runs (tie, ch 838-854 vs zc 844-849, 8 queues) were TWO nodes; this is the one-node pair at 32 queues.
#  5. if under 45 min elapsed: leanmd 16x8x8 no LB, zc x1, ch x1 (2026-10-19, two nodes: channels lost 0.8%; cause
#     unknown).
JID=$1; [ -n "$JID" ] || { echo "usage: conn1_chain.sh <jobid>"; exit 1; }
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
CT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
SO=/u/bhosale/charm-reconverse/tests/util/streamorder
T0=$(date +%s); cd $L
echo "=== conn1 chain, job $JID, $(date +%T), nodes: $(squeue -j $JID -h -o %N)"
if [ -n "${SKIP_PRESET:-}" ]; then echo "--- 1. connpreset skipped (SKIP_PRESET): 14:39 run found no queue sharing in ANY mode, driver 595.71.05"; else
echo "--- 1. connpreset: in-process setenv placement, one GPU"
for m in none before after context; do
  env -u CUDA_DEVICE_MAX_CONNECTIONS srun --jobid=$JID --overlap -N 1 -n 1 --gpus-per-node=1 --exact \
    $SO/connpreset $m > $L/logs/conn1_preset_$m.log 2>&1
  echo "$(date +%T) connpreset $m: $(grep -a 'RESULT' $L/logs/conn1_preset_$m.log || grep -a 'CUDA \|CU \|srun' $L/logs/conn1_preset_$m.log | head -1)"
done
if ! grep -q 'effective_queues=32' $L/logs/conn1_preset_before.log; then
  echo "KILL CRITERION: setenv before cuInit did not give 32 queues; the charm_main preset cannot work. Chain stopped."
  echo "=== conn1 chain stopped $(date +%T)"; exit 1
fi
fi
echo "--- 2. chantest, all arms"
bash $CT/test_chantest_stable.sh $JID 24 3 2>&1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
BANNER=""
lmd() {  # <tag> <channels 0|1> <grid> <lb> [CONNECTIONS or empty]
  local tag=${1}_$(date +%H%M%S) ch=$2 grid="$3" lb=$4 conn=$5
  local envc=(); [ -n "$conn" ] && envc=(CUDA_DEVICE_MAX_CONNECTIONS=$conn)
  local tmo=300; [ "$grid" = "16 8 8" ] && tmo=480   # one node: ~1.7 s/step x 100 plus startup and the LB steps
  env "${envc[@]}" CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="$grid" STEPS=100 LB=$lb CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=$tmo bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local log=$L/logs/$tag.log; local R=$L/ranklogs/$tag/rank_0.log
  BANNER=$(grep -ah 'HAPI> Config:' $R $log 2>/dev/null | head -1 | sed -E 's/.*map (block|roundrobin), //')
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R 2>/dev/null | grep -oE '^[0-9]+' | head -6 | tr '\n' ' ')] | queues: ${BANNER:-NO BANNER}"
  cat $L/ranklogs/$tag/rank_*.log $log 2>/dev/null | grep -a '\[mig-window\]' | sed -E 's/pid=[0-9]+ //' | grep -a 'pack\|ship\|chan-hold\|decide' | sort | uniq -c | sort -k3 | head -16 | cut -c1-150
}
echo "--- 3. leanmd 8x4x4 async: runtime default (32), then CONNECTIONS=8"
lmd c1_zc32_r1 0 "8 4 4" async ""
case "$BANNER" in *"32 hardware queue(s) per device (runtime default)"*) ;; *)
  echo "KILL CRITERION: leanmd banner at the runtime default reads '$BANNER'. Chain stopped."; echo "=== conn1 chain stopped $(date +%T)"; exit 1;; esac
lmd c1_ch32_r1 1 "8 4 4" async ""
for r in 2 3; do lmd c1_zc32_r$r 0 "8 4 4" async ""; lmd c1_ch32_r$r 1 "8 4 4" async ""; done
lmd c1_zc8 0 "8 4 4" async 8
lmd c1_ch8 1 "8 4 4" async 8
echo "--- 4. leanmd 16x8x8 async at the runtime default"
for r in 1 2; do lmd c1_big_zc32_r$r 0 "16 8 8" async ""; lmd c1_big_ch32_r$r 1 "16 8 8" async ""; done
if [ $(( ($(date +%s) - T0) / 60 )) -lt 45 ]; then
  echo "--- 5. leanmd 16x8x8 no LB at the runtime default"
  lmd c1_big_zc32_nolb 0 "16 8 8" none ""; lmd c1_big_ch32_nolb 1 "16 8 8" none ""
else echo "--- 5. skipped: $(( ($(date +%s) - T0) / 60 )) min elapsed"; fi
echo "=== conn1 chain done $(date +%T), $(( ($(date +%s) - T0) / 60 )) min"
