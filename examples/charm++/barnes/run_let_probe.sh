#!/bin/bash
# run_let_probe.sh <jobid>
# Why does a second "balance" step make barnes slower? Particle spread does not
# explain it (the fastest segment has the worst spread), so measure the other
# candidate: the locally essential tree. Its volume is set by how wide each
# PE's SFC interval is, which is exactly what the balancer reshapes, and it is
# not in the balancer's objective.
#
# 150 iterations at -lbperiod=50 gives three segments under three placements:
# after step 0 (good), after step 1 (bad), after the revert (good again).
# BARNES_LET_SIZE walks the tree once per destination per iteration, so this
# run is NOT timing-comparable -- it is measuring volume.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes || exit 1
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=$1
R="srun --jobid=$J -n 4 --cpus-per-task=8 --exact --unbuffered ./numa_wrap.sh ./barnes"
F="-in=galaxy-peri.bin -killat=150 -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1 -eps=0.0249 -dtime=0.005941 -lbperiod=50 +p 4"
LB="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +LBDebug 1"
BARNES_LET_SIZE=1 BARNES_TP_MAP=1 timeout 900 $R $F $LB > check/letprobe.log 2>&1
echo "rc=$? iters=$(grep -ac 'prev time' check/letprobe.log) LETlines=$(grep -ac '^\[LET\]' check/letprobe.log) TPMAPlines=$(grep -ac '^\[TPMAP\]' check/letprobe.log)"
grep -aoP "\[DiffusionLB\] step \d+:.*" check/letprobe.log
