#!/bin/bash
# n2_chan11.sh (2026-10-18 21:50): the announce-and-read protocol at 16x8x8 still hangs after the first migration
# wave (8x4x4 completes). First arm = that run with the channel stall scan on (CHARM_DEBUG_MIGRATE=1) and a short
# timeout, to read the stuck state; then chantest; then, if the debug arm completed, the pairs. Binaries from the
# scratch build at start. 2 A40 nodes, 30 min. Never scancel; node KEPT.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; LCI=/u/bhosale/lci/build-cuda/lib
SCR=/tmp/claude-54324/-u-bhosale-charm-reconverse/73524aa5-0acc-479f-ab89-3199163b9f8f/scratchpad/lmdbuild/leanmd
cd $L
until [ -z "$(squeue -u bhosale -h 2>/dev/null)" ]; do sleep 30; done
if pgrep -x leanmd >/dev/null; then echo "a leanmd still runs; not installing"; else cp $SCR $L/leanmd.new && mv -f $L/leanmd.new $L/leanmd && echo "installed leanmd $(ls -la --time-style=+%H:%M:%S $L/leanmd | awk '{print $6}')"; fi
echo "=== salloc 2 nodes, 30 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:30:00 -J lmd_chan11 > logs/chan11_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_chan11 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}')"
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <LB> <channels> <grid> <tag> <tmo> [dbg]
  local lb=$1 ch=$2 grid=$3 tag=$4 tmo=$5 dbg=${6:-0}
  [ "$(left_s)" -gt $((tmo+30)) ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S); local -a DB=(); [ "$dbg" = 1 ] && DB=(CHARM_DEBUG_MIGRATE=1)
  env "${DB[@]}" NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=$tmo LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
  if [ "$dbg" = 1 ]; then
    cat $L/ranklogs/$TAG/rank_*.log | grep -a 'CHAN-STALL' | sed -E 's/^\[CHAN-STALL [0-9]+\] //; s/channel [0-9a-f]+/channel X/; s/token=[0-9a-f]+/token=X/; s/sender=[0-9]+\/[0-9a-f]+/sender=P\/T/; s/seq=[0-9]+/seq=N/; s/tag [0-9]+/tag N/; s/owner_pe [0-9]+/owner_pe P/' | sort | uniq -c | sort -rn | head -6 | cut -c1-300 | sed 's/^/    /'
    cat $L/ranklogs/$TAG/rank_*.log | grep -a -E 'CHAN-(STALE|CONTROL)|CkDeviceChannel|a read completed|exceed' | sed -E 's/[0-9a-f]{5,}/X/g; s/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -4 | cut -c1-200 | sed 's/^/    /'
  fi
  grep -q 'rc=0' $L/logs/$TAG.log
}
echo "--- 1. the hanging arm with the stall scan"
lmd_arm async 1 "16 8 8" dbg_ch_async_16x8x8 150 1; OK=$?
echo "--- 2. chantest"
bash $G/chantest/test_chantest.sh $JID 24 3
if [ "$OK" = 0 ]; then
  echo "--- 3. pairs"
  lmd_arm async 0 "16 8 8" ctl_async_16x8x8 180
  lmd_arm async 1 "8 4 4" ch_async_8x4x4 120
  lmd_arm async 0 "8 4 4" ctl_async_8x4x4 120
fi
echo "=== chan11 done $(date +%T), left $(left_s) s; node KEPT"
