#!/bin/bash
# after3_chan9.sh <jobid>: on chan9's node with the 20:46 builds: the announce-and-read protocol with the buffer
# marked busy from the read's completion (the 20:40 hang), the pushed resolution from reopen, a moved Compute's
# sender copies re-resolving at once, and live-atom pups with regrowth. chantest, the benchmark (two nodes, two
# processes), leanmd control vs channels at 16x8x8 and 8x4x4 async, then lbcalib if time. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel; LBC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/lbcalib; LCI=/u/bhosale/lci/build-cuda/lib
SCR=/tmp/claude-54324/-u-bhosale-charm-reconverse/73524aa5-0acc-479f-ab89-3199163b9f8f/scratchpad/lmdbuild/leanmd
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
if pgrep -x leanmd >/dev/null; then echo "a leanmd still runs; not installing"; else cp $SCR $L/leanmd.new && mv -f $L/leanmd.new $L/leanmd && echo "installed leanmd $(ls -la --time-style=+%H:%M:%S $L/leanmd | awk '{print $6}')"; fi
echo "=== after3_chan9 $(date +%T), left $(left_s) s"
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {
  local lb=$1 ch=$2 grid=$3 tag=$4 tmo=$5
  [ "$(left_s)" -gt $((tmo+20)) ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=$tmo LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
}
echo "--- 1. chantest"
bash $G/chantest/test_chantest.sh $JID 24 3
echo "--- 2. channel benchmark: two nodes (channel = receiver read), two processes"
for a in "rdma" "ipc"; do echo "--- channel $a"; bash $C/run_channel.sh $JID $a vmm -i 100 -l 20 -r 200 -R 20 2>&1 | grep -E 'channel .*rc=|one-way' | cut -c1-230 | sed 's/^/  /'; done
echo "--- 3. leanmd control vs channels, runtime defaults"
lmd_arm async 1 "16 8 8" ch_async_16x8x8 180
lmd_arm async 0 "16 8 8" ctl_async_16x8x8 180
lmd_arm async 1 "8 4 4" ch_async_8x4x4 120
lmd_arm async 0 "8 4 4" ctl_async_8x4x4 120
lmd_arm async 1 "8 4 4" ch_async_8x4x4 120
lmd_arm async 0 "8 4 4" ctl_async_8x4x4 120
if [ "$(left_s)" -gt 240 ]; then echo "--- 4. lbcalib on uffd"; cd $LBC && timeout $(( $(left_s) - 40 )) bash $LBC/calib2n.sh $JID 2>&1 | grep -E 'rc=|alpha|DONE' | head -12 | cut -c1-200; fi
echo "=== after3_chan9 done $(date +%T), left $(left_s) s; node KEPT"
