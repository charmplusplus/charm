#!/bin/bash
# n2_chan8.sh (2026-10-18 18:30): first full run on FI_MR_CACHE_MONITOR=uffd (the cache disabled cost 10 us per
# inter-process send: libfabric's CXI provider mapped, copied and unmapped the buffer every time) and the -O3 LCI.
# 1. chantest, all transports, with migration. 2. leanmd control vs -channels under the runtime's defaults (no
# balancer flags): 16x8x8 async, 8x4x4 async, 8x4x4 none. 3. sph3d async@25 and none. 4. sph2d moving load async
# and none. 5. pic2d async and none. 6. the channel benchmark: two nodes (put vs the receiver-pull variant if
# built) and two processes. Then 8x4x4 leanmd pairs to the end. 2 A40 nodes, 40 min. Never scancel; node KEPT.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
S3=$G/sph3d; S2=$G/sph2d; P=$G/pic2d; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
LCI=/u/bhosale/lci/build-cuda/lib
cd $L
until [ -z "$(squeue -u bhosale -h 2>/dev/null)" ]; do sleep 30; done
echo "=== salloc 2 nodes, 40 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:40:00 -J lmd_chan8 > logs/chan8_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n lmd_chan8 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T); leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}') sph3d $(ls -la --time-style=+%m-%d_%H:%M $S3/sph3d | awk '{print $6}') channel $(ls -la --time-style=+%m-%d_%H:%M $C/channel | awk '{print $6}')"
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ENVL="LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=uffd"
cfg() { grep -a -m1 -oE 'asynchronous balancing [a-z]+ by the application' "$1" | sed 's/^/    runtime: /'; grep -a 'MetisLB priced' "$1" | grep -c applied | sed 's/^/    metis applied: /'; }
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <LB> <channels> <grid> <tag>
  local lb=$1 ch=$2 grid=$3 tag=$4
  [ "$(left_s)" -gt 200 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=300 LCIDEV=1 LB=$lb CHANNELS=$ch GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis $(grep -a 'MetisLB priced' $R | grep -c applied)/$(grep -ac 'MetisLB priced' $R) applied | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CkDeviceChannel')"
}
sph3d_arm() {  # <LB> <tag>
  local lb=$1 tag=$2
  [ "$(left_s)" -gt 240 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=$LCI POOL=16384 LB=$lb LBDEBUG=1 TMO=400 $S3/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 > /dev/null 2>&1
  local LG=$S3/logs/$TAG.log
  echo "$(date +%T) sph3d $TAG: $(grep -o 'rc=[0-9]*' $LG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LG | tail -1) ms/step | metis $(grep -a 'MetisLB priced' $LG | grep -c applied)/$(grep -ac 'MetisLB priced' $LG) applied | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CANCELLED' $LG)"
}
pic_arm() {  # <mode> <iters>
  local mode=$1 iters=$2 lb
  [ "$(left_s)" -gt 90 ] || { echo "skip pic2d $mode: $(left_s) s left"; return 1; }
  local TAG=pic2d_${mode}_$(date +%H%M%S); local LOG=$P/logs/$TAG.log; mkdir -p $P/logs
  case $mode in none) lb="-f 99999";; async) lb="";; sync) lb="-A";; esac
  echo "[pic2d] $(date +%T) tag=$TAG LB=$mode iters=$iters args: $lb" > $LOG
  env $ENVL timeout 200 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$P \
    $P/numa_wrap.sh stdbuf -oL -eL ./pic2d -W 2048 -H 2048 -w 128 -h 128 -p 4 -d 1 -F 0.9 -s 0.1 -c 20 -r 32 -i $iters -u 10 $lb +LBDebug 1 +gpushm +gpupool +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[pic2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) pic2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LOG) ms/iter | metis $(grep -a 'MetisLB priced' $LOG | grep -c applied)/$(grep -ac 'MetisLB priced' $LOG) applied | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|Failed assert' $LOG)"
}
sph2d_arm() {  # <mode> <steps>
  local mode=$1 steps=$2 lb
  [ "$(left_s)" -gt 240 ] || { echo "skip sph2d $mode: $(left_s) s left"; return 1; }
  local TAG=sph2d_${mode}_$(date +%H%M%S); local LOG=$S2/logs/$TAG.log; mkdir -p $S2/logs
  case $mode in none) lb="-f 99999";; async) lb="";; sync) lb="-A";; esac
  echo "[sph2d] $(date +%T) tag=$TAG LB=$mode steps=$steps args: $lb" > $LOG
  env $ENVL timeout 300 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 -w $N0 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$S2 \
    $S2/numa_wrap_sph.sh stdbuf -oL -eL ./sph2d -x 16 -y 8 -s 0.00125 -r 2 -e 0.1 -V 10 -u 10 -S 100000 -i $steps $lb +LBDebug 1 +gpushm +gpupool +gpupoolsize 8192 +gpupoolalloc vmm +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[sph2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) sph2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | $(grep -aoP 'Average iteration time: [0-9.]+' $LOG | tail -1) ms | metis $(grep -a 'MetisLB priced' $LOG | grep -c applied)/$(grep -ac 'MetisLB priced' $LOG) applied | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|mismatch' $LOG)"
}
ch_arm() { local lay=$1; shift; [ "$(left_s)" -gt 90 ] || { echo "skip channel $lay"; return 1; }; echo "--- channel $lay $*"; bash $C/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20 "$@" 2>&1 | grep -E 'channel .*rc=|one-way' | cut -c1-230 | sed 's/^/  /'; }
echo "--- 1. chantest on uffd"
[ "$(left_s)" -gt 600 ] && bash $G/chantest/test_chantest.sh $JID 24 3
echo "--- 2. leanmd control vs channels, runtime defaults, uffd"
lmd_arm async 0 "16 8 8" ctl_async_16x8x8
lmd_arm async 1 "16 8 8" ch_async_16x8x8
lmd_arm async 0 "8 4 4" ctl_async_8x4x4
lmd_arm async 1 "8 4 4" ch_async_8x4x4
lmd_arm none 0 "8 4 4" ctl_none_8x4x4
lmd_arm none 1 "8 4 4" ch_none_8x4x4
echo "--- 3. sph3d on uffd: async@25 (was 38.1 on disabled), none (56.9)"
sph3d_arm async u_sph3d_async
sph3d_arm none u_sph3d_none
echo "--- 4. sph2d moving load on uffd: async (1.66), none (2.92)"
sph2d_arm async 60000
sph2d_arm none 60000
echo "--- 5. pic2d on uffd: async (27.7), none (27.8)"
pic_arm async 300
pic_arm none 300
echo "--- 6. channel benchmark on uffd: two nodes (put, and the receiver-pull variant if built), two processes"
ch_arm rdma
ch_arm rdma -g
ch_arm ipc
while [ "$(left_s)" -gt 300 ]; do lmd_arm async 1 "8 4 4" ch_async_8x4x4; lmd_arm async 0 "8 4 4" ctl_async_8x4x4; done
echo "=== chan8 done $(date +%T), left $(left_s) s; node KEPT"
