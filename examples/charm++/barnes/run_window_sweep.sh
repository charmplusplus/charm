#!/bin/bash
# run_window_sweep.sh <jobid> [reps]
# Does the second balancing step degrade the placement because it is fitting to
# noise? At -lbperiod=50 the default -lbwindow=2 lets the strategy decide from
# two instrumented iterations out of fifty. Widen the window, and separately
# raise the noise floor, and see whether step 1 stops moving ~1000 objects and
# stops making the GPU imbalance worse.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes || exit 1
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=$1; REPS=${2:-2}
R="srun --jobid=$J -n 4 --cpus-per-task=8 --exact --unbuffered ./numa_wrap.sh ./barnes"
B="-in=galaxy-peri.bin -killat=150 -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1 -eps=0.0249 -dtime=0.005941 -lbperiod=50 +p 4"
LB="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 1"
run(){ # tag  extra-app-args  extra-lb-args
  local tag=$1; shift; local app=$1; shift
  for r in $(seq 1 $REPS); do
    L=check/win_${tag}_$r.log
    BARNES_TP_MAP=1 timeout 300 $R $B $app $LB "$@" > $L 2>&1
    echo "$tag rep$r rc=$? iters=$(grep -ac 'prev time' $L)"
  done
}
run w2    "-lbwindow=2"
run w10   "-lbwindow=10"
run w25   "-lbwindow=25"
run w2f25 "-lbwindow=2"  +LBDiffusionMinImbalance 0.25
echo "SWEEP DONE"
