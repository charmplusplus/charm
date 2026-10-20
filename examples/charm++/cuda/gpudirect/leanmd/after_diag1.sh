#!/bin/bash
# after_diag1.sh <jobid> (15:15): does the immediate build (late init) take the shared-memory and immediate paths?
# 8x4x4 none, then 16x8x8 none if time remains; the [CHAN-INIT] lines and the per-thread post counts say.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd.imm
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
for g in "8 4 4" "16 8 8"; do
  tmo=60; [ "$g" = "16 8 8" ] && tmo=180
  [ "$(left_s)" -gt $((tmo+25)) ] || { echo "skip $g: $(left_s) s left"; continue; }
  TAG=diag_imm_none_$(echo $g | tr ' ' x)_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=1 LB=none GRID="$g" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
  echo "  --- CHAN-INIT:"; cat $R/rank_*.log | grep -a 'CHAN-INIT' | sed -E 's/\[CHAN-INIT [0-9]+\] //' | sort | uniq -c | sort -rn | head -8 | cut -c1-200 | sed 's/^/  /'
  echo "  --- posts per thread (rank 0 and 4):"; cat $R/rank_0.log $R/rank_4.log | grep -a '^\[am-send\]' | sed -E 's/\[am-send\] rank ([0-9]+) thread ([0-9]+): n=([0-9]+) post mean_us=([0-9.]+).*/r\1 t\2 n=\3 post=\4/' | sort | tr '\n' ' ' | fold -w 200 | sed 's/^/  /'; echo
  echo "  --- faults/notes:"; cat $R/rank_*.log | grep -a 'CHAN \|CHAN-WATCH\|Fatal\|abort' | head -4 | cut -c1-200 | sed 's/^/  /'
done
echo "=== diag1 done $(date +%T), left $(left_s) s; node KEPT"
