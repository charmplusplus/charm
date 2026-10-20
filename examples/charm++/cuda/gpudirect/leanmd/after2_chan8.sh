#!/bin/bash
# after2_chan8.sh <jobid>: after after_chan8 (calibration), 8x4x4 async pairs, channels vs control, to the end:
# the first pair on uffd was 144.7 vs 114.6 with a 393-object DiffusionLB wave in the channel run. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; LCI=/u/bhosale/lci/build-cuda/lib
until [ "$(squeue -j $JID -h -o %T 2>/dev/null)" = RUNNING ]; do sleep 15; done
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
while pgrep -f "^bash /u/bhosale/.*after_chan8.sh" >/dev/null; do sleep 5; done
while pgrep -f "run_lmd.sh $JID\|lbcalib" >/dev/null; do sleep 3; done
echo "=== after2_chan8 $(date +%T), left $(left_s) s"
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {
  local lb=$1 ch=$2 grid=$3 tag=$4
  [ "$(left_s)" -gt 120 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=200 LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
}
while [ "$(left_s)" -gt 120 ]; do lmd_arm async 1 "8 4 4" ch_async_8x4x4; lmd_arm async 0 "8 4 4" ctl_async_8x4x4; done
echo "=== after2_chan8 done $(date +%T), left $(left_s) s; node KEPT"
