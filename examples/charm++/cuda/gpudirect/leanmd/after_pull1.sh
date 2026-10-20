#!/bin/bash
# after_pull1.sh (00:30): the receiver-read protocol (leanmd.pull: cross-host tier announces and reads) on the
# early-credit fix. Its 16x8x8 hang of 2026-10-18 showed in-process senders with in_flight=1 and no callback;
# the watchdog now dumps the HAPI queues for that. TARGET: does the read protocol complete 16x8x8 now, and at
# what ms/step. COUNT: 2 arms. KILL: first hang -> its watch lines. Node KEPT.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd.pull
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for i in 1 2; do
  [ "$(left_s)" -gt 170 ] || { echo "skip arm $i: $(left_s) s left"; break; }
  TAG=pull_ch_async_16x8x8_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=1 LB=async GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=150 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|exceed')"
  if ! grep -q 'rc=0' $L/logs/$TAG.log; then
    echo "--- WATCH:"; cat $R/rank_*.log | grep -a 'CHAN-WATCH' | sort -u | cut -c1-300 | sed 's/^/  /'
    echo "--- STALL (dedup):"; cat $R/rank_*.log | grep -a 'CHAN-STALL' | sed -E 's/^\[CHAN-STALL [0-9]+\] //; s/channel [0-9a-f]+ gen [0-9]+ owner_pe [0-9]+/channel X/; s/token=[0-9a-f]+/token=T/; s/seq=[0-9]+/seq=N/; s/epoch=[0-9]+/epoch=E/; s/closed=[0-9]+ pull/closed=C pull/; s/sender=[0-9]+\/[0-9a-f]+/sender=P\/T/; s/closing_gen=[0-9]+/closing_gen=G/; s/\(tag [0-9]+\)/(tag t)/' | sort | uniq -c | sort -rn | head -8 | cut -c1-330 | sed 's/^/  /'
    echo "--- PROBE:"; cat $R/rank_*.log | grep -a 'CHAN-PROBE' | sort -u | head -6 | cut -c1-400 | sed 's/^/  /'
    echo "--- HAPI:"; cat $R/rank_*.log | grep -a 'HAPI-QUEUES' | head -16 | cut -c1-250 | sed 's/^/  /'
    echo "--- faults:"; cat $R/rank_*.log | grep -aiE 'fatal|abort|Segmentation|CORRUPTION|exceed' | head -3 | cut -c1-200 | sed 's/^/  /'
    break
  fi
done
echo "=== pull1 done $(date +%T), left $(left_s) s; node KEPT"
