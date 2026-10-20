#!/bin/bash
# n2_chan7.sh (2026-10-18 17:20): channels vs zero-copy under the runtime's own strategy choice, plus the two
# microbenchmark defects. 1. sendcost: what one converse message costs (send call, handler, one-way) in one
# process, two processes on one node, two nodes -- the channel's credit is such a message and costs 12-13 us.
# 2. channel benchmark, two nodes, the put issued as 1/2/4/8 concurrent one-sided writes (+gpuputchunks): the
# single write ran at ~11 GB/s against the zero-copy path's ~14. 3. leanmd control vs -channels back to back on
# one binary and the runtime's defaults (no balancer flags): 16x8x8 async, 8x4x4 async, 8x4x4 none, then 8x4x4
# async pairs to the end. Waits for the queue to empty, sallocs 2 A40 nodes for 30 min. Never scancel; node KEPT.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
SC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/sendcost
LCI=/u/bhosale/lci/build-cuda/lib
cd $L
until [ -z "$(squeue -u bhosale -h 2>/dev/null)" ]; do sleep 30; done
echo "=== salloc 2 nodes, 30 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:30:00 -J lmd_chan7 > logs/chan7_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_chan7 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}') channel $(ls -la --time-style=+%m-%d_%H:%M $C/channel | awk '{print $6}')"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ENVL="LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so"
sendcost_arm() {  # <layout>
  local lay=$1 N NP PPN PPNARG
  case $lay in memcpy) N=1 NP=1 PPN=1 PPNARG="+ppn 2";; ipc) N=1 NP=2 PPN=2 PPNARG="+ppn 1";; rdma) N=2 NP=2 PPN=1 PPNARG="+ppn 1";; esac
  local LOG=$SC/logs/sendcost_${lay}_$(date +%H%M%S).log; mkdir -p $SC/logs
  env $ENVL timeout 120 srun --jobid=$JID --overlap --mpi=cray_shasta -N $N -n $NP --ntasks-per-node=$PPN --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$SC stdbuf -oL -eL ./sendcost $PPNARG > $LOG 2>&1
  echo "$(date +%T) sendcost $lay: rc=$? | $(grep -E '^(converse|entry) ' $LOG | tr '\n' '|' | cut -c1-300)"
}
ch_arm() {  # <layout> <args...>
  local lay=$1; shift
  [ "$(left_s)" -gt 90 ] || { echo "skip channel $lay: $(left_s) s left"; return 1; }
  echo "--- channel $lay $*"
  bash $C/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20 "$@" 2>&1 | grep -E 'channel .*rc=|one-way|credit' | cut -c1-230 | sed 's/^/  /'
}
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <LB> <channels> <grid> <tag>
  local lb=$1 ch=$2 grid=$3 tag=$4
  [ "$(left_s)" -gt 200 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=300 LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) applied, $(grep -a 'Objects migrating' $R | sed -E 's/.*migrating: ([0-9]+).*/\1/' | head -1) moves | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
}
echo "--- 1. one converse message: send call, handler, one-way"
sendcost_arm memcpy; sendcost_arm ipc; sendcost_arm rdma
echo "--- 2. channel benchmark, two nodes, put in 1/2/4/8 concurrent writes; then two processes on one node"
ch_arm rdma +gpuputchunks 1
ch_arm rdma +gpuputchunks 2
ch_arm rdma +gpuputchunks 4
ch_arm rdma +gpuputchunks 8
ch_arm ipc
echo "--- 3. leanmd control vs channels, the runtime's defaults, same binary"
lmd_arm async 0 "16 8 8" ctl_async_16x8x8
lmd_arm async 1 "16 8 8" ch_async_16x8x8
lmd_arm async 0 "8 4 4" ctl_async_8x4x4
lmd_arm async 1 "8 4 4" ch_async_8x4x4
lmd_arm none 0 "8 4 4" ctl_none_8x4x4
lmd_arm none 1 "8 4 4" ch_none_8x4x4
while [ "$(left_s)" -gt 300 ]; do lmd_arm async 1 "8 4 4" ch_async_8x4x4; lmd_arm async 0 "8 4 4" ctl_async_8x4x4; done
echo "=== chan7 done $(date +%T), left $(left_s) s; node KEPT"
