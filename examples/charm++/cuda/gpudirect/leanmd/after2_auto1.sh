#!/bin/bash
# after2_auto1.sh <jobid>: after after_auto1: sph2d async with the device dimension forced (+LBDiffusionGpuDim), the one difference from the hand-tuned run that the automatic chooser makes (it picks the launch dimension by criticality on sph2d). Never sallocs, never scancels.
# path fixed (-f 99999 no longer trips the lag check) and MetisLB re-evaluating every fourth event in the default
# chain. Order: sph3d none, sph3d async, sph2d none, sph2d async, then sph2d async with the old 10% floor as an
# experiment (XFLAGS) if time remains. Never sallocs, never scancels.
JID=$1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
S3=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
S2=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph2d
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d
LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); [ -n "$END_EPOCH" ] || { echo "no end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
echo "=== after_auto1 $(date +%T), left $(left_s) s; sph3d $(ls -la --time-style=+%H:%M $S3/sph3d | awk '{print $6}') sph2d $(ls -la --time-style=+%H:%M $S2/sph2d | awk '{print $6}')"
cfg() {  # <rank0 or main log>: the runtime's own account of what it chose
  grep -a -m1 -oE 'default strategy: [^(]*' "$1" | sed 's/^/    runtime: /'
  grep -a -m1 -oE 'transfer-cost model: [a-z-]+ table[^;]*' "$1" | sed 's/^/    runtime: /'
  grep -a -m1 -oE 'asynchronous balancing [a-z]+ by the application' "$1" | sed 's/^/    runtime: /'
  grep -a 'MetisLB priced' "$1" | sed -E 's/.*step bound ([0-9.]+) -> ([0-9.]+).*\(saves [^)]*\) ([0-9.]+%)[^;]*; .*([0-9]+) migration\(s\).*; (.*)$/    metis: bound \1 -> \2 (\3) \4 moves: \5/' | head -3
  grep -a -c 'MetisLB priced' "$1" | sed 's/^/    metis decisions: /'
}
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <LB> <grid> <tag>
  local lb=$1 grid=$2 tag=$3
  [ "$(left_s)" -gt 200 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI PERIOD=20 LAG=16 POOL=16384 TMO=300 LCIDEV=1 LB=$lb GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) leanmd $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R) | mean21-100 $(mean_steps $R 21 100) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R | awk '{printf "%s ",$NF}')] | metis-moves $(grep -a 'Objects migrating' $R | sed -E 's/.*migrating: ([0-9]+).*/\1/' | head -1) | drift $(grep -aoP 'drift \K[0-9.E+-]+' $R | tail -1) | bad=$(cat $L/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION')"
  cfg $R
}
sph3d_arm() {  # <LB> <tag>
  local lb=$1 tag=$2
  [ "$(left_s)" -gt 240 ] || { echo "skip $tag: $(left_s) s left"; return 1; }
  local TAG=${tag}_$(date +%H%M%S)
  env NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=$LCI POOL=16384 LB=$lb LBDEBUG=1 TMO=400 $S3/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 > /dev/null 2>&1
  local LG=$S3/logs/$TAG.log
  echo "$(date +%T) sph3d $TAG: $(grep -o 'rc=[0-9]*' $LG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LG | tail -1) ms/step | 2nd half $(awk '/^Step [0-9]+ .*ms/{n++; t[n]=$0} END{}' $LG) $(grep -aoP 'second half[^0-9]*\K[0-9.]+' $LG | tail -1) | LB events $(grep -ac '#Objects migrating' $LG) migr $(grep -aoE '#Objects migrating: [0-9]+' $LG | awk '{s+=$3} END{print s+0}') | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|CANCELLED' $LG)"
  cfg $LG
}
pic_arm() {  # <mode> <iters>
  local mode=$1 iters=$2 lb
  [ "$(left_s)" -gt 90 ] || { echo "skip pic2d $mode: $(left_s) s left"; return 1; }
  local TAG=pic2d_${mode}_$(date +%H%M%S); local LOG=$P/logs/$TAG.log; mkdir -p $P/logs
  case $mode in none) lb="-f 99999";; sync) lb="-f 50 -b 50";; async) lb="-f 50 -b 50 -a -l 40";; esac
  echo "[pic2d] $(date +%T) tag=$TAG LB=$mode iters=$iters args: $lb" > $LOG
  env LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout 200 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$P \
    $P/numa_wrap.sh stdbuf -oL -eL ./pic2d -W 2048 -H 2048 -w 128 -h 128 -p 4 -d 1 -F 0.9 -s 0.1 -c 20 -r 32 -i $iters -u 10 $lb +LBDebug 1 +gpushm +gpupool +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[pic2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) pic2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LOG) ms/iter | LB events $(grep -ac '#Objects migrating' $LOG) migr $(grep -aoE '#Objects migrating: [0-9]+' $LOG | awk '{s+=$3} END{print s+0}') | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|Failed assert' $LOG)"
  cfg $LOG
}
sph2d_arm() {  # <mode> <steps>: the 2026-09-13 moving-load benchmark (16x8 patches, 4 processes x 8 PEs on one node)
  local mode=$1 steps=$2 lb
  [ "$(left_s)" -gt 240 ] || { echo "skip sph2d $mode: $(left_s) s left"; return 1; }
  local TAG=sph2d_${mode}${XFLAGS:+_x}_$(date +%H%M%S); local LOG=$S2/logs/$TAG.log; mkdir -p $S2/logs
  case $mode in none) lb="-f 99999";; async) lb="-f 2000 -b 2000 -a -l 1800";; sync) lb="-f 2000 -b 2000";; esac
  echo "[sph2d] $(date +%T) tag=$TAG LB=$mode steps=$steps args: $lb" > $LOG
  env LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout 300 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 -w $N0 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$S2 \
    $S2/numa_wrap_sph.sh stdbuf -oL -eL ./sph2d -x 16 -y 8 -s 0.00125 -r 2 -e 0.1 -V 10 -u 10 -S 100000 -i $steps $lb ${XFLAGS:-} +LBDebug 1 +gpushm +gpupool +gpupoolsize 8192 +gpupoolalloc vmm +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[sph2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) sph2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | $(grep -aiE 'average|ms/step|ms per step' $LOG | tail -1 | cut -c1-90) | LB events $(grep -ac '#Objects migrating' $LOG) migr $(grep -aoE '#Objects migrating: [0-9]+' $LOG | awk '{s+=$3} END{print s+0}') | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|mismatch' $LOG)"
  cfg $LOG
}
while pgrep -f "^bash /u/bhosale/.*after_auto1.sh" >/dev/null; do sleep 3; done
echo "=== after2_auto1 start $(date +%T), left $(left_s) s"
XFLAGS="+LBDiffusionGpuDim" sph2d_arm async 60000
sph3d_arm async auto3_sph3d_async
echo "=== after2_auto1 done $(date +%T), left $(left_s) s; node KEPT"
