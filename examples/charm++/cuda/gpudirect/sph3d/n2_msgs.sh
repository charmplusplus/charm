#!/bin/bash
# n2_msgs.sh <jobid>  -- 98M on 2 nodes, both host-event cuts on in every arm:
# "meas" = measured host weights; "msgs" = +LBMetisHostByMsgs. sync x3 pairs, async x2.
JID=$1
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
one() {  # one <LB> <variant> <rep>
  local LBM=$1 V=$2 REP=$3 EXTRA=""
  [ "$V" = msgs ] && EXTRA="+LBMetisHostByMsgs $XTRA"; [ "$V" = meas ] && EXTRA="$XTRA"
  TAGP=${TP:-n2m}${REP}${V} LB=$LBM $D/arms_quiet.sh $JID both $EXTRA > /dev/null 2>&1
  local L=$D/logs/${TP:-n2m}${REP}${V}_both_${LBM}.log
  local rc=$(grep -o 'rc=[0-9]*' $L | tail -1)
  local ok=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|quiet-level\|ran more than' $L)
  local ms=$(grep -E "^  step +(325|350|375|400|425|450|475|500|525|550|575|600):" $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n; else print "NA"}')
  local hm=$(for S in 2 3 4; do grep "^\[PELOAD\] step=$S pe=" $L | awk '{split($4,b,"="); h=b[2]+0; sum+=h; n++; if(h>mx)mx=h} END{if(n) printf "%.2f/%.2f ", mx, sum/n}'; done)
  local dv=$(grep "measured loads explain" $L | grep Diffusion | sed -E 's/.*device ([0-9]+)%.*/\1/' | sed -n 2,4p | tr '\n' ' ')
  local tol=$(grep -o "host weights by message count, [0-9.]* us per message" $L | head -1); local rep=$(grep -o "[0-9]* repair move(s)" $L | head -1)
  local hpred=$(grep -o "host dimension max [0-9.]* -> [0-9.]* against limit [0-9.]*" $L | head -1 | sed 's/host dimension //')
  echo "$(date +%T) $LBM $V rep$REP: $rc checks=$ok bad=$bad | steps300-600 ${ms} ms | host max/avg @LB2-4: ${hm}| device% ${dv}| ${tol} ${rep} | ${hpred}"
  grep -i "fatal\|abort\|did not consume\|quiet-level\|ran more than" $L | head -2 | cut -c1-160
  [ "$bad" != "0" ] && { echo "STOP: fault in $L"; exit 3; }
}
for REP in 1 2; do one sync msgs $REP; one sync meas $REP; done
one async msgs 1
echo "done $(date +%T); node time $(squeue -j $JID -h -o %M)"
