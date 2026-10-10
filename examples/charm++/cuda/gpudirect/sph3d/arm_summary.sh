#!/bin/bash
# arm_summary.sh <log>...: per-arm LB-event summary for the async idle-gap arms (2026-10-10).
for L in "$@"; do
  [ -f "$L" ] || continue
  rc=$(grep -o 'rc=[0-9]*' $L | tail -1)
  n2=$(grep -E "^  step +[0-9]+:" $L | awk '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0;} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,m); if (st>1000){s+=m[1];n++}} END{if(n) printf "%.1f", s/n; else print "NA"}')
  avg=$(grep "Average iteration time" $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/')
  res=$(grep -E "LB at step [0-9]+: resumed" $L | sed -E 's/.*resumed ([0-9.]+) s after.*/\1/' | awk '{s+=$1;n++;if($1>mx)mx=$1} END{if(n) printf "n=%d avg=%.0f max=%.0f ms", n, 1000*s/n, 1000*mx}')
  wait=$(grep -E "AtSyncWait entered" $L | sed -E 's/.*entered ([0-9.]+) s after.*/\1/' | awk '{s+=$1;n++} END{if(n) printf "avg=%.0f ms", 1000*s/n}')
  strat=$(grep -E "strategy finished at" $L | sed -E 's/.*duration ([0-9.]+) s.*/\1/' | awk '{s+=$1;n++;if($1>mx)mx=$1} END{if(n) printf "n=%d avg=%.1f max=%.1f ms", n, 1000*s/n, 1000*mx}')
  helper=$(grep -E "strategy ran [0-9.]+ ms on the helper" $L | sed -E 's/.*strategy ran ([0-9.]+) ms.*/\1/' | awk '{s+=$1;n++;if($1>mx)mx=$1} END{if(n) printf "n=%d avg=%.1f max=%.1f ms", n, 1000*s/n/1000, mx}')
  cupti=$(grep -h "LBCUPTI" $L | sed -E 's/.*total=([0-9.]+)s/\1/' | awk '{s+=$1;n++;if($1>mx)mx=$1} END{if(n) printf "n=%d avg=%.1f max=%.1f ms", n, 1000*s/n, 1000*mx}')
  bad=$(grep -ciE "Segmentation|Fatal|Abort|did not consume" $L)
  echo "$(basename $L): $rc bad=$bad | 2nd-half $n2 ms, avg $avg ms | resume $res | atsyncwait $wait | strategy(inline) $strat | strategy(helper) $helper | cupti-build $cupti"
done
