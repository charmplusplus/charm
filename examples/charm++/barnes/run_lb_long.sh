#!/bin/bash
# run_lb_long.sh <jobid> <reps>
# Long run, rare LB. -killat=400 -lbperiod=100 puts balancing steps at
# iterations 2, 102, 202, 302, so there are ~100 clean iterations between them
# and the load drift between steps is visible instead of being buried in a
# sawtooth. +LBDebug 1 so the balancing windows are detected from the log
# rather than inferred from the schedule.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes || exit 1
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=$1; REPS=${2:-3}
K=${K:-400}; PER=${PER:-100}
IN=${IN:-galaxy-peri.bin}
COMMON="-killat=$K -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1 -eps=0.0249 -dtime=0.005941"
LBOPT="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 1"
R="srun --jobid=$J -n 4 --cpus-per-task=8 --exact --unbuffered ./numa_wrap.sh ./barnes"
for rep in $(seq 1 $REPS); do
  for arm in noLB sync async; do
    case $arm in
      noLB)  LB="";;
      sync)  LB="-lbperiod=$PER $LBOPT";;
      async) LB="-lbperiod=$PER -lbasync=1 $LBOPT +LBAsync";;
    esac
    L=check/long_${arm}_$rep.log
    timeout 900 $R -in=$IN $COMMON $LB +p 4 > $L 2>&1; rc=$?
    n=$(grep -ac 'prev time' $L)
    echo "$arm rep$rep rc=$rc iterations_timed=$n"
    [ "$rc" != 0 ] && { echo "STOP"; grep -aiE 'abort|fatal|reason:' $L | head -3; exit 1; }
  done
done
echo "LONG DONE"
