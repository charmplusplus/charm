#!/bin/bash
# pool_cmp.sh <jobid> <reps> [tags...]   tags: <lb>_<mode>, lb in noLB|sync|async, mode in nopool|pool
J=$1; REPS=$2; shift 2
B=/u/bhosale/charm-reconverse/examples/charm++/barnes; cd $B || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled CHARM_ZC_STATS=1 BARNES_MASS_CHECK=1
BASE="-in=clustered.bin -killat=15 -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1 -lbperiod=5"
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
NOPOOL="+gpushm +gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 512 +gpuipceventpool 256"
POOL="+gpushm +gpupool +gpuipceventpool 256"
OUT=$B/check/pool_cmp_results.txt; : > $OUT
for r in $(seq 1 $REPS); do for tag in "$@"; do
  lb=${tag%_*}; mode=${tag#*_}
  case $lb in noLB) LB="";; sync) LB="$D";; async) LB="-lbasync=1 $D +LBAsync";; esac
  [ $mode = pool ] && IPC="$POOL" || IPC="$NOPOOL"
  L=$B/check/bh_${tag}_$r.log
  PES=4 timeout 200 srun --unbuffered --jobid=$J --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL ./numa_wrap.sh ./barnes $BASE +p 4 $LB $IPC > $L 2>&1; rc=$?; sleep 1
  avg=$(grep -oP 'finished all [0-9]+ iterations with avg time \K[0-9.]+' $L | tail -1)
  en=$(grep -oP 'energy \K[-0-9.]+' $L | tail -1)
  mass=$(grep -aoiP 'mass[^\n]{0,60}' $L | tail -1)
  mig=$(grep -oP 'Objects migrating: \K[0-9]+' $L | paste -sd+ | bc)
  staged=$(grep -a ipc-stats $L | grep -o 'staged=[0-9]*' | cut -d= -f2 | paste -sd,)
  ok=1; [ "$rc" = 0 ] && [ -n "$avg" ] || ok=0
  echo "$tag rep$r ok=$ok rc=$rc avg_iter=${avg:-NONE} energy=${en:-NONE} migrated=${mig:-0} staged=${staged:-?} mass='${mass}'" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag rep$r"; grep -anE 'Reason:|Fatal|Abort|outstanding for|WARNING|Need more|Segmentation' $L | head -6 | cut -c1-200; echo "== tail: $(tail -2 $L | cut -c1-120 | tr '\n' '|')"; exit 1; fi
done; done
echo "ALL PASSED"
