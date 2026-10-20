#!/bin/bash
# n2_head1.sh (2026-10-17): Step 2 of the branch consolidation -- reproduce the headline results with the HEAD binaries
# (e53a99c5d = multi-GPU runtime 9d315ab61 + apps 5d574c8b3 + the dead-switch removal; sph3d/leanmd/pic2d relinked 20:21),
# ONE arm per configuration, all inside one 2-node allocation so every number is compared in-session.
# Targets (references, steady state = second half):
#  sph3d 98M, 256 patches, 8 A40s, 2000 steps, control layout (4,1,4): noLB ~56.5 ms/step (10-09 2000 steps 56.5; 10-17
#    1000-step 56.1-56.8); async@25 lag 10 Metis-once-then-Diffusion ~37-39 (10-09 36.9; 10-17 1000-step 37.7-38.0 second
#    half); sync@25 same balancers ~41-43 (10-17 1000-step 42.6). TNL-SPH 98M 8 ranks 2000 steps: 39.55 (LB off) / 39.97 (LB on).
#  sph3d noLB (1,4,4) 2000 steps: the layout must tie the control's noLB at steady state AND its halo tiers must read the
#    control's (same process : same node : other node ~ 1 : 5.3 : 3.3; the 18b pre-fix arm read 347440 : 0 : 2969400).
#  leanmd 16x8x8, 100 steps, period 20, lag 16, (4,1,8), vmm 16 GB pool: mean42-100 none ~1496-1512 ms / sync ~995-1016 /
#    async ~973-1056 (lmd4 chain, 10-17).
#  pic2d 2048^2 in 128^2 patches (256 objects / 64 PEs), 300 iterations, Metis then Diffusion at 50/50 lag 40
#    (lb2node.sbatch recipe): nolb ~38.9 / sync 38.5-56.8 / async 36.4-42.0 ms/iter (job 21872670); the point here is
#    that HEAD runs all three clean (rc=0, no abort, Metis partitions exactly once).
# One variable per arm (the mode, or the layout). Kill criterion: a fault in the FIRST arm (sph3d noLB control) means the
# HEAD build is broken -> the chain stops and reports. Never scancel; the node is KEPT at the end. 25-minute allocation.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d
LCI=/u/bhosale/lci/build-cuda/lib
cd $D
ulimit -c 0
echo "=== binaries: sph3d $(ls -la --time-style=+%m-%d_%H:%M $D/sph3d | awk '{print $6}') leanmd $(ls -la --time-style=+%m-%d_%H:%M $L/leanmd | awk '{print $6}') pic2d $(ls -la --time-style=+%m-%d_%H:%M $P/pic2d | awk '{print $6}'); HEAD $(git -C $D rev-parse --short HEAD)"
echo "=== salloc 2 nodes, 25 min $(date +%T)"
salloc --no-shell -A mzu-delta-gpu -p gpuA40x4-interactive -N 2 --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-node=4 --exclusive --mem=0 --time=00:25:00 -J sph3d_head1 > logs/head1_salloc.out 2>&1 &
JID=""; until [ -n "$JID" ]; do sleep 15; JID=$(squeue -u bhosale -n sph3d_head1 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=""; for i in 1 2 3 4 5 6; do e=$(squeue -j $JID -h -o %e 2>/dev/null | head -1); [ -n "$e" ] && END_EPOCH=$(date -d "$e" +%s 2>/dev/null) && break; sleep 10; done
[ -n "$END_EPOCH" ] || { echo "cannot read the job's end time; stopping (node KEPT)"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "jobid=$JID granted $(date +%T) nodes $(squeue -j $JID -h -o %N), ends $(date -d @$END_EPOCH +%T)"
echo "--- clocksource per node:"; srun --jobid=$JID --overlap -N 2 -n 2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "    $(hostname) clocksource=$(cat /sys/devices/system/clocksource/clocksource0/current_clocksource)"' 2>/dev/null
export NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
LASTLOG=""
summary() {  # <log> <ns>
  local LG=$1 ns=$2 ds
  ds=$(grep -m1 -h '\[DiffusionLB\] measured loads explain' $LG | sed -E 's/.*device ([0-9]+)%.*/\1/'); [ -n "$ds" ] || ds=0
  echo "$(date +%T) $(basename $LG .log): $(grep -o 'rc=[0-9]*' $LG | tail -1) checks=$(grep -c 'check ok' $LG) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION\|outstanding for' $LG) | whole-run $(grep 'Average iteration' $LG | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $LG | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $LG | grep -oE '[0-9]+$') | device-share ${ds}% | launch $(grep -m1 -h '\[MetisLB\] measured loads explain' $LG | sed -E 's/.*launch ([0-9]+)%.*/\1/')% | resume $(grep -h 'LB at step [0-9]*: resumed' $LG | sed -n 2,79p | sed -E 's/.*resumed ([0-9.]+) s after.*/\1/' | awk '{s+=$1; n++} END{if(n) printf "%.3fs x%d", s/n, n}') | last: $(tail -1 $LG | cut -c1-80)"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|too small|CUPTI_ERROR|outstanding for" $LG | head -2 | cut -c1-160 | sed 's/^/    /'
}
run_arm() {  # <mode> <pfx> <PPN> <GPP> <ns> <env...>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4 ns=$5; shift 5
  local pes=$((4*gpp)) np=$((2*ppn))
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx $mode: $(left_s) s left"; return 1; }
  local TAG=head_${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 LBDEBUG=1 "$@" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF > /dev/null 2>&1
  LASTLOG=$D/logs/$TAG.log
  summary $LASTLOG $ns
}
failed() { ! grep -q "rc=0" <<< "$(grep -o "rc=[0-9]*" $LASTLOG | tail -1)" || [ "$(grep -ci "fatal\|abort\|Segmentation\|CORRUPTION\|Err 5\|CUPTI_ERROR" $LASTLOG)" -gt 0 ]; }
tiers() { echo "    halo tiers: $(grep -h 'halo      same process\|halo      same node\|halo      other node' $LASTLOG | sed -E 's/.*halo +(same process|same node|other node) +n= *([0-9]+).*/\1=\2/' | tr '\n' ' ')"; grep -h "MetisLB cross-group messages by host" $LASTLOG | head -1 | cut -c1-200 | sed 's/^/    /'; }
lmd_mean() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd_arm() {  # <mode>: one leanmd arm, control layout (4,1,8), 16x8x8 100 steps period 20 lag 16
  local mode=$1
  [ "$(left_s)" -gt 200 ] || { echo "skip leanmd $mode: $(left_s) s left"; return 1; }
  local TAG=head_P4G1K8_16x8x8_${mode}_$(date +%H%M%S)
  env NODES=2 POOL_ALLOC=vmm LCI_LIB=$LCI STEPS=100 PERIOD=20 LAG=16 GRID="16 8 8" POOL=16384 TMO=600 LCIDEV=1 LB=$mode PPN=4 GPP=1 PES=8 $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  local R=$L/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) leanmd $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean21-100 $(lmd_mean $R 21 100) ms | mean42-100 $(lmd_mean $R 42 100) ms | total $(grep -oP 'Total application time \K[0-9.]+' $R 2>/dev/null | tail -1)s | migr=[$(grep -ah 'cross node migrations' $R 2>/dev/null | awk '{printf "%s ",$NF}')] | bad=$(cat $L/ranklogs/$TAG/rank_*.log 2>/dev/null | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|out of memory|outstanding for') | last: $(tail -1 $R 2>/dev/null | cut -c1-80)"
  cat $L/ranklogs/$TAG/rank_*.log 2>/dev/null | grep -aiE "fatal|abort|Segmentation|CORRUPTION|out of memory|outstanding for|Err 5" | head -n 2 | cut -c1-200 | sed 's/^/    /'
  grep -ah "measured loads explain" $R 2>/dev/null | head -n 2 | cut -c1-200 | sed 's/^/    /'
}
pic_arm() {  # <mode>: one pic2d arm, the lb2node.sbatch recipe (8 processes x 8 PEs, 256 patches, 300 iterations)
  local mode=$1 lb
  [ "$(left_s)" -gt 100 ] || { echo "skip pic2d $mode: $(left_s) s left"; return 1; }
  local TAG=head_pic2d_${mode}_$(date +%H%M%S) LOG=$P/logs/$TAG.log; mkdir -p $P/logs
  local BAL="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBCommOn +LBDebug 1"
  case $mode in none) lb="-f 99999";; sync) lb="-f 50 -b 50 $BAL";; async) lb="-f 50 -b 50 -a -l 40 $BAL +LBAsync";; esac
  echo "[pic2d] $(date +%T) tag=$TAG LB=$mode args: $lb" > $LOG
  env X=1 LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    timeout 240 srun --jobid=$JID --overlap --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$P \
    $P/numa_wrap.sh stdbuf -oL -eL ./pic2d -W 2048 -H 2048 -w 128 -h 128 -p 4 -d 1 -F 0.9 -s 0.1 -c 20 -r 32 -i 300 -u 10 $lb +gpushm +gpupool +gpuipceventpool 256 +ppn 8 >> $LOG 2>&1
  echo "[pic2d] rc=$? $(date +%T)" >> $LOG
  echo "$(date +%T) pic2d $TAG: $(grep -o 'rc=[0-9]*' $LOG | tail -1) | avg $(grep -aoP 'Average iteration time: \K[0-9.]+' $LOG) ms/iter | metis-partitions $(grep -ac 'Metis partitioning' $LOG) | migrating $(grep -aoE '#Objects migrating: [0-9]+' $LOG | awk '{s+=$NF; n++} END{printf "%d over %d LB steps", s, n}') | bad=$(grep -aciE 'fatal|abort|Segmentation|CORRUPTION|mismatch|exceed capacity|outstanding for' $LOG) | last: $(tail -1 $LOG | cut -c1-80)"
  grep -aiE 'fatal|abort|Segmentation|CORRUPTION|mismatch|exceed capacity|reason:|outstanding for' $LOG | head -2 | cut -c1-200 | sed 's/^/    /'
}
echo "=== head1 start $(date +%T), job $JID, left $(left_s) s"
P0=(CHARM_LB_CUPTI_PERIODIC=0)
echo "--- 1. sph3d control (4,1,4), 2000 steps: noLB (kill-criterion arm), async@25 lag 10, sync@25"
run_arm none ctl 4 1 2000; tiers
if failed; then echo "!!! the HEAD sph3d noLB control arm FAILED: the chain stops here (node KEPT)"; echo "=== head1 stopped $(date +%T), left $(left_s) s"; exit 2; fi
run_arm async ctl 4 1 2000 "${P0[@]}"; tiers
run_arm sync ctl 4 1 2000 "${P0[@]}"; tiers
echo "--- 2. sph3d noLB (1,4,4), 2000 steps: the layout tie and the tier accounting"
run_arm none s14 1 4 2000; tiers
echo "--- 3. leanmd (4,1,8) 16x8x8 100 steps: none / async / sync"
lmd_arm none
lmd_arm async
lmd_arm sync
echo "--- 4. pic2d 2 nodes, 300 iterations: none / async / sync"
pic_arm none
pic_arm async
pic_arm sync
echo "=== head1 done $(date +%T), left $(left_s) s; node KEPT"
