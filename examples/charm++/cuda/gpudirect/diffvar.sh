#!/bin/bash
# diffvar.sh <JOBID> <NODES> <REPEATS> -- is the DiffusionLB win at 2 nodes bimodal?
#
# leanmd weak, the same arm repeated, each run in its OWN log directory so a
# fast and a slow instance can be compared afterwards (scale.sh reuses one
# directory per arm and loses the earlier run). Observed so far at N=2, all on
# gpub[002,005], all pool=1(+0), noLB 1589.8-1597.0:
#   sync  1269.6  migr 1935 454 959 2
#   sync  1296.8  migr 1928 570 802 2
#   sync  1528.6  migr 1907   2   3 455
#   async 1557.5  migr 1695 474 900 3
# Neither the load share (device 23-25% in both) nor the migration pattern
# separates them, so this samples the distribution and keeps the evidence.
JID=$1; NODES=${2:-2}; REP=${3:-5}
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192; ulimit -c 0
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs; cd $L || exit 1
PES=$((32*NODES)); GRID="$((8*NODES)) 8 8"
SRUN="srun --jobid=$JID --mpi=cray_shasta -N $NODES -n $((4*NODES)) --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=${SLURM_CPUS_PER_TASK:-16} --cpu-bind=none --exact --kill-on-bad-exit=1"
C="$GRID 100 20 20 -computemap local -density gradient +pe $PES +setcpuaffinity +gpushm +gpuipceventpool 256 +gpupool +gpupoolsize 2048"
MD="+balancer MetisLB +balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
wm(){ awk -v lo=$2 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=100){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1"; }
one(){ local tag=$1; shift; local dir=$RL/diffvar_N${NODES}_$tag; rm -rf $dir; mkdir -p $dir
  RANKLOG_DIR=$dir env RANKLOG_DIR=$dir timeout 500 $SRUN stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd $C "$@" >$dir/srun.err 2>&1
  local rc=$? m42; m42=$(wm $dir/rank_0.log 42)
  printf "  %-12s mean42=%-8s rc=%s migr=[%s] imb=[%s] dir=%s\n" "$tag" "$m42" "$rc" \
    "$(grep -ah 'cross node migrations' $dir/rank_0.log | awk '{printf "%s ",$NF}')" \
    "$(grep -ah 'max ratio' $dir/rank_0.log | grep -oE 'max ratio [0-9.]+' | awk '{printf "%s ",$3}')" "$(basename $dir)"; }
echo "=== diffvar N=$NODES, $REP repeats of leanmd weak sync + 1 noLB control, $(date +%T)"
one noLB_ctl
for i in $(seq 1 $REP); do one sync_$i $MD +LBDebug 1; done
echo "=== spread: $(grep -h . /dev/null; echo)"
echo "=== DONE $(date +%T)"
