#!/bin/bash
# chan6b.sh <jobid>: replaces n2_chan6.sh's runner on the same allocation after its first channel arm died at
# startup (std::bad_array_new_length: the channel's copy constructor and assignment never copied owner_mgr, so the
# sender's copy from the handshake carried an uninitialized group id into the owner-keyed comm-graph call; the
# pup path carries it, which is why chantest passed). Fixed and relinked 12:26 (leanmd.ch under a new inode).
# Waits for the chain's running control arm to finish, then: ch 16x8x8 (default gate), ctl/ch 8x4x4 (default),
# ch 16x8x8 at +LBMetisMaxCutRise 2.0 (its control ran under the old runner), the open items (after_chan6.sh),
# then the 8x4x4 pair at 2.0 if time remains. Never sallocs, never scancels; node KEPT.
JID=$1; [ -n "$JID" ] || { echo "usage: chan6b.sh <jobid>"; exit 1; }
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
cd $L
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); [ -n "$END_EPOCH" ] || { echo "no end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
while pgrep -f "run_lmd.sh $JID" >/dev/null; do sleep 5; done
echo "=== chan6b $(date +%T), job $JID, left $(left_s) s, leanmd.ch $(ls -la --time-style=+%H:%M leanmd.ch | awk '{print $6}')"
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd.ch $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd.ch | awk '{print $6}')"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
echo "--- 1. the migration test"
[ "$(left_s)" -gt 600 ] && bash $G/chantest/test_chantest.sh $JID 24 3
echo "--- 2. leanmd -channels vs control, binary leanmd.ch, layout (4,1,8)"
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 TMO=300 LCIDEV=1 BIN=leanmd.ch
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
summary() {  # <tag>
  local T=$1 R=$L/ranklogs/$1/rank_0.log LG=$L/logs/$1.log
  echo "$(date +%T) $T: $(grep -o 'rc=[0-9]*' $LG | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-100 $(mean_steps $R 21 100) ms | mean42-100 $(mean_steps $R 42 100) ms | total $(grep -oP 'Total application time \K[0-9.]+' $R 2>/dev/null | tail -1)s | migr=[$(grep -ah 'cross node migrations' $R 2>/dev/null | awk '{printf "%s ",$NF}')] | metis-moves $(grep -a 'Objects migrating' $R 2>/dev/null | sed -E 's/.*migrating: ([0-9]+).*/\1/' | tr '\n' ',') | ext/int MB $(grep -a 'External comm BEFORE' $R 2>/dev/null | head -1 | awk '{printf "%.0f", $5}')/$(grep -a 'Internal comm BEFORE' $R 2>/dev/null | head -1 | awk '{printf "%.0f", $5}') | energy-drift $(grep -aoP 'drift \K[0-9.E+-]+' $R 2>/dev/null | tail -1) | bad=$(cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|CkDeviceChannel|exceed') | last: $(tail -1 $R 2>/dev/null | cut -c1-80)"
  cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -aiE "fatal|abort|Segmentation|CORRUPTION|out of memory|CkDeviceChannel|exceed|Err 5" | head -n 3 | cut -c1-200 | sed 's/^/    /'
  cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -a 'CHAN-STALL' | sed 's/^\[CHAN-STALL [0-9]*\] //' | sort | uniq -c | sort -rn | head -n 3 | cut -c1-240 | sed 's/^/    /'
}
DROP=0
arm() {  # <LB> <channels 0|1> <grid> <steps> <tag> [dbg 0|1] [extra LB flags]
  local lb=$1 ch=$2 grid=$3 steps=$4 tag=$5 dbg=${6:-0} extra="${7:-}"
  [ "$DROP" = 1 ] && [ "$ch" = 1 ] && { echo "skip $tag: channel arms dropped after a fault"; return 1; }
  [ "$(left_s)" -gt 200 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  local -a DB=(); [ "$dbg" = 1 ] && DB=(CHARM_DEBUG_MIGRATE=1)
  env "${DB[@]}" CHANNELS=$ch LB=$lb GRID="$grid" STEPS=$steps PPN=4 GPP=1 PES=8 LMD_EXTRA="$extra" $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  summary $TAG
  if ! grep -q 'rc=0' $L/logs/$TAG.log || [ "$(cat $L/ranklogs/$TAG/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION')" -gt 0 ]; then
    echo "    FAULT in $tag"; return 2
  fi
  return 0
}
arm async 1 "16 8 8" 100 ch_async_16x8x8_dbg 1
arm async 0 "8 4 4" 100 ctl_async_8x4x4
arm async 1 "8 4 4" 100 ch_async_8x4x4_dbg 1
arm async 1 "16 8 8" 100 ch_async_16x8x8_rise2 1 "+LBMetisMaxCutRise 2.0"
bash $L/after_chan6.sh $JID
arm async 0 "8 4 4" 100 ctl_async_8x4x4_rise2 0 "+LBMetisMaxCutRise 2.0"
arm async 1 "8 4 4" 100 ch_async_8x4x4_rise2 1 "+LBMetisMaxCutRise 2.0"
echo "=== chan6b done $(date +%T), left $(left_s) s; node KEPT"
