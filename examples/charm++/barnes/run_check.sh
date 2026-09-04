#!/bin/bash
# Correctness sweep. Everything at -killat=1 so the acceleration dump lands on
# iteration 0, where every run still has identical positions -- at any later
# iteration the runs have drifted and the comparison would measure trajectory
# divergence rather than force error.
#
# The reference is the code itself at -theta=0.05, which opens almost every
# cell, so the walk degenerates to direct summation.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1
export FI_MR_CACHE_MONITOR=disabled

D=${DUMPDIR:?}
IN=${IN:?}
JOB=${JOB:?}
COMMON="-in=$IN -killat=1 -b=8 -ppc=256"
RUN="srun --jobid=$JOB -n 1 --cpus-per-task=8 --exact"

run(){  # name binary extra-args
  local name=$1; shift
  local bin=$1; shift
  echo "########## $name ##########"
  BARNES_ACCEL_DUMP=$D/$name timeout 600 $RUN ./$bin $COMMON "$@" +p2
  echo "=== $name exit: $? ==="
}

run ref       barnes-cpu       -theta=0.05
run mono050   barnes-cpu-mono  -theta=0.5
run quad050   barnes-cpu       -theta=0.5
run quad070   barnes-cpu       -theta=0.7
run gpu050    barnes           -theta=0.5
run quad050d3 barnes-cpu       -theta=0.5 -decomplevels=3
echo "SWEEP DONE"
