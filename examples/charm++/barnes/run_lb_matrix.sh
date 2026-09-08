#!/bin/bash
# run_lb_matrix.sh <jobid> <reps>
# noLB / sync / async on each dataset, same iteration count, same flags.
# Prints one line per run as it finishes and leaves the logs in check/.
cd /u/bhosale/charm-reconverse/examples/charm++/barnes || exit 1
ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export IPATH_NO_BACKTRACE=1 FI_MR_CACHE_MONITOR=disabled
J=$1; REPS=${2:-3}
K=${K:-60}
COMMON="-killat=$K -b=128 -p=2048 -blockmap=1 -qd=0 -let=1 -devwalk=1 -devexch=1 -devlet=1"
LBOPT="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
OUT=check/lbmatrix.txt; : > $OUT
R="srun --jobid=$J -n 4 --cpus-per-task=8 --exact --unbuffered ./numa_wrap.sh ./barnes"

# dataset:extra-args
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
    L=check/lbm_${IN%.bin}_${arm}_$rep.log
    timeout 400 $R -in=$IN $COMMON $EXTRA $LB +p 4 > $L 2>&1; rc=$?
    avg=$(grep -aoP 'finished all \d+ iterations with avg time \K[0-9.]+' $L | tail -1)
    mig=$(grep -aoP 'cross node migrations AFTER LB: \K[0-9]+' $L | paste -sd+ | bc)
    # particle-count spread across PEs on the last iteration reported
    imb=$(grep -a FRONTIER $L | tail -6 | grep -aoP 'myParts=\K[0-9]+' | \
          awk '{v[NR]=$1; s+=$1} END{m=v[1]; for(i=1;i<=NR;i++) if(v[i]>m) m=v[i]; if(NR>0) printf "%.3f", m/(s/NR)}')
    echo "$IN $arm rep$rep rc=$rc avg_ms=$(awk -v a="${avg:-0}" 'BEGIN{printf "%.1f", a*1000}') migrations=${mig:-0} parts_max_over_mean=${imb:-NA}" | tee -a $OUT
    [ "$rc" != 0 ] && { echo "  STOP: rc=$rc"; grep -aiE 'abort|fatal|reason:' $L | head -3; exit 1; }
  done
done
done
echo "MATRIX DONE"
