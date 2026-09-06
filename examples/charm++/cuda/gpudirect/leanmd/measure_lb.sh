#!/bin/bash
# Compare leanmd step times with and without load balancing.
#
# Reports the mean step time over a post-LB window rather than total wall time:
# the first steps include one-off device allocation and cold-start IPC setup,
# and the LB steps themselves carry migration cost, so totals blur the thing we
# want to see (does the resulting placement run faster?). Both are printed.
#
# usage: measure_lb.sh <jobid> <reps> [extra leanmd args...]
set -u
JOBID=$1; REPS=$2; shift 2
EXTRA=("$@")

export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}"
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd

# Mean of "Step N Benchmark Time X ms/step" for N in [lo,hi]
window_mean() {
  awk -v lo="$2" -v hi="$3" '
    /^Step [0-9]+ Benchmark Time/ { n=$2+0; t=$5+0; if (n>=lo && n<=hi) { s+=t; c++ } }
    END { if (c>0) printf "%.2f", s/c; else printf "NA" }' "$1"
}

run_one() {
  local tag=$1 rep=$2; shift 2
  local log="/tmp/meas_${tag}_${rep}.log"
  timeout 300 srun --jobid="$JOBID" --mpi=cray_shasta -n 4 --cpus-per-task=8 \
    --cpu-bind=cores --exact ./leanmd "${EXTRA[@]}" "$@" > "$log" 2>&1
  local rc=$?
  local total; total=$(grep -oP 'Total application time \K[0-9.]+' "$log" | tail -1)
  local post; post=$(window_mean "$log" 21 100)
  local drift; drift=$(grep -oP 'drift \K[0-9.E+-]+' "$log" | tail -1)
  printf "%-14s rep%-2s rc=%-3s total=%-10s mean_step_21_100=%-9s drift=%s\n" \
    "$tag" "$rep" "$rc" "${total:-NA}" "${post:-NA}" "${drift:-NA}"
  rm -f core.leanmd.*
}

for r in $(seq 1 "$REPS"); do run_one "noLB"    "$r"; done
for r in $(seq 1 "$REPS"); do run_one "MetisLB" "$r" +balancer MetisLB; done
for r in $(seq 1 "$REPS"); do run_one "Metis+Diff" "$r" +balancer MetisLB +balancer DiffusionLB; done
