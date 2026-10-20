#!/bin/bash
# n2_chan6.sh (2026-10-18 10:10): persistent channels, sixth allocation. After chan5's first arm: the comm-graph
# edge of a put was recorded under the SENDER's object manager with the OWNER's element id, so a Cell->Compute
# put named the Cell of the same id and MetisLB saw a scrambled graph (7.5 GB external vs the control's 2.8;
# totals equal). Recorded under the owner's manager now, so both modes see the same graph and the same mapping.
# Also: chantest 6/6 on the epoch protocol (10:03). Waits for the queue to empty (chan5 holds its node to 10:33),
# installs the scratch leanmd.ch, sallocs 2 A40 nodes for 30 min, then chantest, then pairs at (4,1,8): 16x8x8
# async ctl/ch with the default Metis gate, then with +LBMetisMaxCutRise 2.0 (chan5: at 1.0 the control's cut rose
# 108.7% and was still refused; the 'not parsed by the RTS' warning for the flag is false, MetisLB's constructor reads
# it later), the same at 8x4x4, then 8x4x4 pairs at 2.0 to the end. Channel arms run with
# CHARM_DEBUG_MIGRATE=1 (stall scan + CHAN-STALE prints). Judge by mean42-100 ms/step and compare ext/int MB
# between the modes of a pair. Never scancel; the node is KEPT at the end.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
cd $L
until [ -z "$(squeue -u bhosale -h 2>/dev/null)" ]; do sleep 30; done
SCR=/tmp/claude-54324/-u-bhosale-charm-reconverse/73524aa5-0acc-479f-ab89-3199163b9f8f/scratchpad/lmdbuild/leanmd.ch
if pgrep -x leanmd.ch >/dev/null; then echo "a leanmd.ch still runs; not installing"; else cp $SCR $L/leanmd.ch && echo "installed leanmd.ch $(ls -la --time-style=+%H:%M $L/leanmd.ch | awk '{print $6}')"; fi
echo "=== salloc 2 nodes, 30 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:30:00 -J lmd_chan6 > logs/chan6_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_chan6 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
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
arm async 0 "16 8 8" 100 ctl_async_16x8x8
arm async 1 "16 8 8" 100 ch_async_16x8x8_dbg 1; [ $? -eq 2 ] && DROP=1
arm async 0 "16 8 8" 100 ctl_async_16x8x8_rise2 0 "+LBMetisMaxCutRise 2.0"
arm async 1 "16 8 8" 100 ch_async_16x8x8_rise2 1 "+LBMetisMaxCutRise 2.0"
arm async 0 "8 4 4" 100 ctl_async_8x4x4
arm async 1 "8 4 4" 100 ch_async_8x4x4_dbg 1
arm async 0 "8 4 4" 100 ctl_async_8x4x4_rise2 0 "+LBMetisMaxCutRise 2.0"
arm async 1 "8 4 4" 100 ch_async_8x4x4_rise2 1 "+LBMetisMaxCutRise 2.0"
arm sync 1 "16 8 8" 100 ch_sync_16x8x8_dbg 1
arm sync 0 "16 8 8" 100 ctl_sync_16x8x8
while [ "$(left_s)" -gt 300 ]; do arm async 1 "8 4 4" 100 ch_async_8x4x4_rise2 1 "+LBMetisMaxCutRise 2.0"; arm async 0 "8 4 4" 100 ctl_async_8x4x4_rise2 0 "+LBMetisMaxCutRise 2.0"; done
echo "=== chan6 done $(date +%T), left $(left_s) s; node KEPT"
