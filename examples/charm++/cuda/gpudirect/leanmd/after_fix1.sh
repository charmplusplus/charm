#!/bin/bash
# after_fix1.sh (00:25): the early-credit fix, one 16x8x8 channel arm on the kept node.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd
for i in 1; do
  [ "$(left_s)" -gt 200 ] || { echo "skip arm $i: $(left_s) s left"; break; }
  TAG=fix1_ch_async_16x8x8_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=1 LB=async GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=150 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | last: $(tail -1 $R/rank_0.log | cut -c1-60)"
  if grep -q 'rc=124' $L/logs/$TAG.log; then
    echo "--- WATCH:"; cat $R/rank_*.log | grep -a 'CHAN-WATCH' | cut -c1-300 | sed 's/^/  /'
    echo "--- STALL:"; cat $R/rank_*.log | grep -a 'CHAN-STALL' | sort -u | cut -c1-330 | sed 's/^/  /'
    echo "--- PROBE:"; cat $R/rank_*.log | grep -a 'CHAN-PROBE' | sort -u | cut -c1-400 | sed 's/^/  /'
    echo "--- HAPI:"; cat $R/rank_*.log | grep -a 'HAPI-QUEUES' | head -12 | cut -c1-250 | sed 's/^/  /'
    break
  fi
done
echo "=== fix1 done $(date +%T), left $(left_s) s; node KEPT"
