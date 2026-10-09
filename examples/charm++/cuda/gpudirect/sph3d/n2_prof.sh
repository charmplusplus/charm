#!/bin/bash
# n2_prof.sh <jobid>  -- 98M, 2 nodes, SPH_HOST_PROFILE=1: sync x2, async x1. Prints the arm summary
# and the hottest/median PE's HOSTPROF line per arm; full per-PE tables stay in the logs.
JID=$1
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
one() {  # one <LB> <rep>
  local LBM=$1 REP=$2
  SPH_HOST_PROFILE=1 CHARM_ZC_STATS=1 TAGP=prof$REP LB=$LBM $D/arms_quiet.sh $JID both > /dev/null 2>&1
  local L=$D/logs/prof${REP}_both_${LBM}.log
  local rc=$(grep -o 'rc=[0-9]*' $L | tail -1)
  local ok=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|quiet-level\|ran more than' $L)
  local ms=$(grep -E "^  step +(325|350|375|400|425|450|475|500|525|550|575|600):" $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n; else print "NA"}')
  local avg=$(grep "Average iteration time" $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/')
  local hm=$(for S in 2 3 4; do grep "^\[PELOAD\] step=$S pe=" $L | awk '{split($4,b,"="); h=b[2]+0; sum+=h; n++; if(h>mx)mx=h} END{if(n) printf "%.2f/%.2f ", mx, sum/n}'; done)
  echo "$(date +%T) $LBM rep$REP: $rc checks=$ok bad=$bad | steps300-600 ${ms} ms, whole-run avg ${avg} ms | host max/avg @LB2-4: ${hm}| HOSTPROF lines $(grep -c '^HOSTPROF' $L)"
  grep -i "fatal\|abort\|did not consume\|quiet-level\|ran more than" $L | head -2 | cut -c1-160
  [ "$bad" != "0" ] && { echo "STOP: fault in $L"; exit 3; }
}
one sync 1; one sync 2; one async 1
echo "done $(date +%T); node time $(squeue -j $JID -h -o %M)"
