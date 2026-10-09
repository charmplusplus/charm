#!/bin/bash
# n2_pairs.sh <jobid>  -- 98M on 2 nodes: alternating base/both pairs, sync x3 then async x2.
# Prints, per arm: ms/step at steps 300-600, per-PE host max/avg at LB steps 2-4, device busy %.
JID=$1
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib TAGP=n2
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
one() {  # one <LB> <arm> <rep>
  local LBM=$1 ARM=$2 REP=$3
  TAGP=n2r$REP LB=$LBM $D/arms_quiet.sh $JID $ARM > /dev/null 2>&1
  local L=$D/logs/n2r${REP}_${ARM}_${LBM}.log
  local rc=$(grep -o 'rc=[0-9]*' $L | tail -1)
  local ok=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|quiet-level\|ran more than' $L)
  local ms=$(grep -E "^  step +(325|350|375|400|425|450|475|500|525|550|575|600):" $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n; else print "NA"}')
  local hm=$(for S in 2 3 4; do grep "^\[PELOAD\] step=$S pe=" $L | awk '{split($4,b,"="); h=b[2]+0; sum+=h; n++; if(h>mx)mx=h} END{if(n) printf "%.2f/%.2f ", mx, sum/n}'; done)
  local dv=$(grep "measured loads explain" $L | grep Diffusion | sed -E 's/.*device ([0-9]+)%.*/\1/' | sed -n 2,4p | tr '\n' ' ')
  local cut=$(grep -o "cross-group cut [0-9.]* -> [0-9.]*" $L | head -1 | sed 's/cross-group cut //')
  local sk=$(grep -E "^  step +600:" $L | sed -E 's/.*quiet-skipped ([0-9]+).*/\1/')
  echo "$(date +%T) $LBM $ARM rep$REP: $rc checks=$ok bad=$bad | steps300-600 ${ms} ms | host max/avg s per interval @LB2-4: ${hm}| device% ${dv}| Metis cut ${cut} | skipped ${sk}"
  grep -i "fatal\|abort\|did not consume\|quiet-level\|ran more than" $L | head -2 | cut -c1-160
  [ "$bad" != "0" ] && { echo "STOP: fault in $L"; exit 3; }
}
for REP in 1 2 3; do one sync base $REP; one sync both $REP; done
for REP in 1 2; do one async base $REP; one async both $REP; done
echo "done $(date +%T); node time $(squeue -j $JID -h -o %M)"
