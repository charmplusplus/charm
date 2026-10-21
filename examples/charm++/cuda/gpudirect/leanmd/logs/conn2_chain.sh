#!/bin/bash
# conn2_chain.sh <jobid>: one A40 node. Validates the channel move-protocol change of 2026-10-20 15:30 (staged
# announces: no sender hold; urgent control messages) against the 15:00 pairs on the same node type.
# TARGET: leanmd 16x8x8 async ONE node, channels 1568-1625 vs zero-copy 1538-1549 ms/step (42-100) before the change.
# ONE VARIABLE: the channel move protocol (zero-copy arms are the control; they do not use channels).
# KILL CRITERION: a chantest arm fails or hangs (TMO), or a leanmd channel run hangs (rc=124 with CHAN-STALL lines):
# the chain stops and says so. Never sallocs, never scancels.
#  1. chantest, all 8 arms (memcpy and ipc; lbx must still abort).
#  2. leanmd 16x8x8 async: zc, ch, zc, ch (mean42-100; the staged counters from CHARM_ZC_STATS at exit).
#  3. leanmd 8x4x4 async: zc, ch, zc, ch (the channel win there, 209 vs 203-219, must survive).
#  4. if under 40 min: leanmd 16x8x8 no LB: zc, ch with CHARM_ZC_STATS=1 (the 2.3% steady-state gap; chan-stats).
JID=$1; [ -n "$JID" ] || { echo "usage: conn2_chain.sh <jobid>"; exit 1; }
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
CT=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
T0=$(date +%s); cd $L
echo "=== conn2 chain, job $JID, $(date +%T), nodes: $(squeue -j $JID -h -o %N), binaries: leanmd $(date -r $L/leanmd +%m-%d_%H:%M) chantest $(date -r $CT/chantest +%m-%d_%H:%M)"
echo "--- 1. chantest, all arms"
bash $CT/test_chantest_stable.sh $JID 24 3 2>&1 | tee $L/logs/conn2_chantest.out
if grep -q 'FAIL' $L/logs/conn2_chantest.out; then echo "KILL CRITERION: a chantest arm failed. Chain stopped."; echo "=== conn2 chain stopped $(date +%T)"; exit 1; fi
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd() {  # <tag> <channels 0|1> <grid> <lb>
  local tag=${1}_$(date +%H%M%S) ch=$2 grid="$3" lb=$4
  local tmo=300; [ "$grid" = "16 8 8" ] && tmo=480
  CHARM_ZC_STATS=1 CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="$grid" STEPS=100 LB=$lb CHANNELS=$ch BIN=leanmd \
    POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=$tmo bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local log=$L/logs/$tag.log; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | pre-LB 1-20 $(mean_steps $R 1 20) | 25-40 $(mean_steps $R 25 40) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R 2>/dev/null | grep -oE '^[0-9]+' | head -4 | tr '\n' ' ')]"
  cat $L/ranklogs/$tag/rank_*.log $log 2>/dev/null | grep -a '\[mig-window\]' | sed -E 's/.*\[mig-window\] pid=[0-9]+ //' | awk '$1=="chan-hold"||$1=="stand-down"||$1=="pack"||$1=="ship"||$1=="land"{n=substr($2,3)+0; m=$4+0; mx=$8+0; N[$1]+=n; S[$1]+=n*m; if(mx>MX[$1])MX[$1]=mx} END{for(s in N) printf "   %-11s n=%6d mean=%7.1f ms max=%8.1f ms\n", s, N[s], (N[s]?S[s]/N[s]:0), MX[s]}' | sort
  cat $L/ranklogs/$tag/rank_*.log $log 2>/dev/null | grep -a 'chan-stats' | sed -E 's/pid=[0-9]+ //' | head -4 | cut -c1-200
  local stalls=$(cat $L/ranklogs/$tag/rank_*.log 2>/dev/null | grep -ac 'CHAN-STALL')
  [ "$stalls" -gt 0 ] && { echo "   CHAN-STALL lines: $stalls"; cat $L/ranklogs/$tag/rank_*.log | grep -a 'CHAN-STALL' | head -6 | cut -c1-220; }
  if [ $rc -ne 0 ] && [ $ch = 1 ]; then echo "KILL CRITERION: channel run rc=$rc. Chain stopped."; echo "=== conn2 chain stopped $(date +%T)"; exit 1; fi
}
echo "--- 2. leanmd 16x8x8 async"
lmd c2_big_zc 0 "16 8 8" async; lmd c2_big_ch 1 "16 8 8" async; lmd c2_big_zc 0 "16 8 8" async; lmd c2_big_ch 1 "16 8 8" async
echo "--- 3. leanmd 8x4x4 async"
lmd c2_zc 0 "8 4 4" async; lmd c2_ch 1 "8 4 4" async; lmd c2_zc 0 "8 4 4" async; lmd c2_ch 1 "8 4 4" async
if [ $(( ($(date +%s) - T0) / 60 )) -lt 40 ]; then
  echo "--- 4. leanmd 16x8x8 no LB"
  lmd c2_big_zc_nolb 0 "16 8 8" none; lmd c2_big_ch_nolb 1 "16 8 8" none
else echo "--- 4. skipped: $(( ($(date +%s) - T0) / 60 )) min elapsed"; fi
echo "=== conn2 chain done $(date +%T), $(( ($(date +%s) - T0) / 60 )) min"
