#!/bin/bash
# n2_patches.sh <jobid> [LB] -- 98M on 2 nodes at 480 (6x4x20), 256 (8x4x8) and 128 (4x4x8) patches,
# message-count host weights on; prints step, host max/avg, device busy per arm.
JID=$1; LBM=${2:-sync}
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
for G in "6 4 20" "8 4 8" "4 4 8"; do
  set -- $G; TAG=p$(( $1*$2*$3 ))_${LBM}
  POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x $1 -y $2 -z $3 -r 2 -e 0.05 -i 600 -S 25 -k 1 +LBMetisHostByMsgs > /dev/null 2>&1
  L=$D/logs/$TAG.log
  ms=$(grep -E "^  step +(325|350|375|400|425|450|475|500|525|550|575|600):" $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n; else print "NA"}')
  avg=$(grep "Average iteration time" $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/')
  hm=$(for S in 2 3 4; do grep "^\[PELOAD\] step=$S pe=" $L | awk '{split($4,b,"="); h=b[2]+0; sum+=h; n++; if(h>mx)mx=h} END{if(n) printf "%.2f/%.2f ", mx, sum/n}'; done)
  dv=$(grep "measured loads explain" $L | grep Diffusion | sed -E 's/.*device ([0-9]+)%.*/\1/' | sed -n 2,4p | tr '\n' ' ')
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume' $L) | steps300-600 ${ms} ms, whole-run ${avg} ms | host max/avg @LB2-4: ${hm}| device% ${dv}| $(grep -o 'active [0-9]*/[0-9]*' $L | tail -1)"
  grep -i "fatal\|abort\|did not consume" $L | head -2 | cut -c1-160
done
echo "done $(date +%T); node time $(squeue -j $JID -h -o %M)"
