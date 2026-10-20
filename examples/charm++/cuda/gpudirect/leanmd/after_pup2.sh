#!/bin/bash
# after_pup2.sh <jobid> (14:17): the 1% without LB at 16x8x8 -- channel arm with MESSAGE credits/notes (leanmd.msgcredit),
# channel arm with SHARED-MEMORY credits/notes (leanmd), and the control arm, each with the
# channel host statistics (CHARM_ZC_STATS=1: put host time by tier, puts queued, events, credit send time) and
# the per-thread message-post statistics with LCI retry reasons (if libreconverse was rebuilt in time).
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; cd $L; ulimit -c 0
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1 BIN=leanmd CHARM_ZC_STATS=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
# imm: shared-memory same-host + the note as the data write's immediate across hosts (leanmd.imm)
for arm in imm ctl msg shmwrite; do
  [ "$(left_s)" -gt 230 ] || { echo "skip $arm: $(left_s) s left"; continue; }
  ch=1; bin=leanmd; [ $arm = msg ] && bin=leanmd.msgcredit; [ $arm = imm ] && bin=leanmd.imm; [ $arm = ctl ] && ch=0
  TAG=stats_${arm}_none_16x8x8_$(date +%H%M%S); R=$L/ranklogs/$TAG
  env CHANNELS=$ch LB=none BIN=$bin GRID="16 8 8" STEPS=100 PPN=4 GPP=1 PES=8 TMO=200 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms"
  echo "  --- zc-host (per process):"; cat $R/rank_*.log | grep -a '\[zc-host\]' | sed -E 's/pid=[0-9]+ //' | sort | uniq -c | sort -k2 | head -40 | cut -c1-200 | sed 's/^/  /'
  echo "  --- am-send, threads with the longest posts:"; cat $R/rank_*.log | grep -a '^\[am-send\]' | sort -t= -k3 -n -r | sort -k7 -t' ' | awk '{print}' | sort -k9.9 -n -r 2>/dev/null | head -0; cat $R/rank_*.log | grep -a '^\[am-send\]' | awk -F'post mean_us=' '{split($2,a," "); print a[1], $0}' | sort -n -r | head -8 | cut -d' ' -f2- | cut -c1-260 | sed 's/^/  /'
done
echo "=== pup2-stats done $(date +%T), left $(left_s) s; node KEPT"
