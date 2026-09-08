#!/bin/bash
# The LB comparison re-run at a tree-piece count the decomposition actually
# fills. -p=736 against ~705 populated pieces leaves 4% empty instead of 66%,
# so every PE holds work and the balancer's cut is not degenerate.
# Does not stop the whole matrix on one dataset's failure: a dataset whose
# decomposition needs more than 736 pieces aborts, and that is a result.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes || exit 1
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=$1; REPS=${2:-2}; P=${P:-736}; K=${K:-60}
COMMON="-killat=$K -b=128 -p=$P -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1"
LBOPT="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
R="srun --jobid=$J -n 4 --cpus-per-task=8 --exact --unbuffered ./numa_wrap.sh ./barnes"
SETS=("galaxy-peri.bin:-eps=0.0249 -dtime=0.005941"
      "galaxy.bin:-eps=0.0249 -dtime=0.005941"
      "clustered.bin:")
for rep in $(seq 1 $REPS); do
for s in "${SETS[@]}"; do
  IN=${s%%:*}; EXTRA=${s#*:}
  for arm in noLB sync async; do
    case $arm in
      noLB)  LB="";;
      sync)  LB="-lbperiod=5 $LBOPT";;
      async) LB="-lbperiod=5 -lbasync=1 $LBOPT +LBAsync";;
    esac
    L=check/m736_${IN%.bin}_${arm}_$rep.log
    timeout 200 $R -in=$IN $COMMON $EXTRA $LB +p 4 > $L 2>&1; rc=$?
    used=$(grep -aoP 'used treepieces \K[0-9]+' $L | head -1)
    echo "$IN $arm rep$rep rc=$rc used=${used:-NONE} iters=$(grep -ac 'prev time' $L)"
  done
done
done
echo "M736 DONE"
