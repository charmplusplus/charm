#!/bin/bash
# compare.sh <jobid> [reps]   app (noLB, sync, async) vs reference (none, sync, async; greedy and lpt)
# on one A40 node with identical arguments; prints a table of step times and
# the checksum agreement. BASE= the shared arguments (defaults below).
J=$1; REPS=${2:-1}
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/moe; cd $P || exit 1
BASE=${BASE:-"-e 64 -m 2048 -h 8192 -t 8192 -i 30 -u 3 -z 1.0 -p 10 -C 5"}
REFBASE=$(echo "$BASE" | sed 's/ -h / -H /')
OUT=$P/ranklogs/compare_results.txt; : > $OUT
row() { # tag log
  avg=$(grep -ho -P 'Average step time: \K[0-9.]+' $2 | head -1)
  out2=$(grep -ho -P 'out2 \K[0-9.e+-]+' $2 | tail -1); w2=$(grep -ho -P 'w2 \K[0-9.e+-]+' $2 | tail -1)
  mv=$(grep -aho -P 'emigrate n=\K[0-9]+' $(dirname $2)/rank_*.log 2>/dev/null | paste -sd+ | bc)
  [ -z "$mv" ] && mv=$(grep -ho -P 'experts moved: \K[0-9]+' $2)
  printf "%-22s avg_ms=%-9s out2=%-24s w2=%-24s moved=%s\n" "$1" "${avg:-NONE}" "${out2:-NONE}" "${w2:-NONE}" "${mv:-0}" | tee -a $OUT
}
for r in $(seq 1 $REPS); do
  BASE="$BASE" $P/ranklogs/lbcmp.sh $J 1 noLB_pool sync_pool async_pool >/dev/null 2>&1
  for t in noLB sync async; do row "app_$t" $P/ranklogs/lbc_${t}_pool_1/rank_0.log; done
  for cfg in "none:none" "sync_greedy:sync --place greedy" "async_greedy:async --place greedy" "sync_lpt:sync --place lpt" "async_lpt:async --place lpt"; do
    tag=${cfg%%:*}; lb=${cfg#*:}
    $P/ref/run_ref.sh $J $tag $REFBASE -f 5 -b 5 --lb $lb >/dev/null 2>&1
    row "ref_$tag" $P/ranklogs/ref_$tag/rank_0.log
  done
done
