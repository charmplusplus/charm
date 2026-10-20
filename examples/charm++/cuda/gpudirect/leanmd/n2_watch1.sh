#!/bin/bash
# n2_watch1.sh (2026-10-18 23:50). TARGET: the stall scan of the leanmd 16x8x8 -channels hang at step 46-47
# (sender-write protocol, runtime defaults; hung in chan4's repeat and in after_safe at 23:43, completed in
# chan5's debug arms, the 20:08 arm and after_v1: a timing race the CHARM_DEBUG_MIGRATE prints hide). The
# library now carries an always-on watchdog: after 20 s without a channel event on a PE it prints a tally and
# up to 12 stuck copies ([CHAN-WATCH], [CHAN-STALL] with the source stream's state and the tier). VARIABLE: none;
# the same arm up to 5 times. KILL CRITERION: the first hang (rc=124) -- its watch lines are the result and the
# arms stop; 5 completions = "did not reproduce in 5". First chantest (sanity of the relinked library).
# Never scancel from here; the node is KEPT at the end. ulimit -c 0 is set in run_lmd.sh.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
cd $L; ulimit -c 0
until [ -z "$(squeue -u bhosale -h 2>/dev/null)" ]; do sleep 30; done
echo "=== salloc 2 nodes, 30 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:30:00 -J lmd_watch1 > logs/watch1_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_watch1 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}')"
echo "--- 1. chantest (step 3, 24 PEs)"
[ "$(left_s)" -gt 900 ] && bash $G/chantest/test_chantest.sh $JID 24 3 2>&1 | grep -E 'PASS|FAIL|rc=' | head -8
echo "--- 2. leanmd -channels 16x8x8 async, layout (4,1,8), up to 5 arms, stop at the first hang"
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
summary() { local T=$1 R=$L/ranklogs/$1/rank_0.log LG=$L/logs/$1.log
  echo "$(date +%T) $T: $(grep -o 'rc=[0-9]*' $LG | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R 2>/dev/null | awk '{printf "%s ",$NF}')] | bad=$(cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|exceed') | last: $(tail -1 $R 2>/dev/null | cut -c1-60)"
  cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -aiE "fatal|abort|Segmentation|CORRUPTION|out of memory|exceed" | head -n 3 | cut -c1-200 | sed 's/^/    /'; }
watch_dump() { local T=$1; echo "    --- CHAN-WATCH tallies (per PE, first print):"
  cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -a 'CHAN-WATCH' | sed -E 's/\[CHAN-WATCH [0-9]+\] no channel event for [0-9]+ s \([0-9]+ handled\): //' | sort | uniq -c | sort -rn | head -n 12 | cut -c1-220 | sed 's/^/    /'
  echo "    --- CHAN-STALL examples (deduplicated on the sender/owner state):"
  cat $L/ranklogs/$T/rank_*.log 2>/dev/null | grep -a 'CHAN-STALL' | sed -E 's/^\[CHAN-STALL [0-9]+\] //; s/channel [0-9a-f]+ gen [0-9]+ owner_pe [0-9]+/channel X/; s/token=[0-9a-f]+/token=T/; s/seq=[0-9]+/seq=N/; s/epoch=[0-9]+/epoch=E/; s/closed=[0-9]+ pull/closed=C pull/; s/sender=[0-9]+\/[0-9a-f]+/sender=P\/T/; s/closing_gen=[0-9]+/closing_gen=G/; s/\(tag [0-9]+\)/(tag t)/' | sort | uniq -c | sort -rn | head -n 10 | cut -c1-300 | sed 's/^/    /'
  echo "    --- steps reached per rank: $(for f in $L/ranklogs/$T/rank_*.log; do grep -a '^Step' $f | tail -1 | awk '{printf "%s ", $2}'; done)"; }
hung=0; done_n=0
for i in 1 2 3 4 5; do
  [ "$(left_s)" -gt 220 ] || { echo "skip arm $i: $(left_s) s left"; break; }
  TAG=watch_ch_async_16x8x8_$(date +%H%M%S)
  env CHANNELS=1 LB=async GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=180 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  summary $TAG
  if grep -q 'rc=124' $L/logs/$TAG.log; then hung=1; watch_dump $TAG; echo "    HANG CAUGHT in arm $i: stopping the arms"; break; fi
  grep -q 'rc=0' $L/logs/$TAG.log && done_n=$((done_n+1))
done
[ $hung = 1 ] || echo "no hang in $done_n completed arms"
echo "=== watch1 done $(date +%T), left $(left_s) s; node KEPT"
