#!/bin/bash
# n2_final.sh <jobid> <patches: "8 4 8"|"4 4 8"> <steps> <arm>...   arm = none | sync | async | syncT | asyncT  (T = +LBMetisHostByMsgs)
JID=$1; G=$2; NS=$3; shift 3
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
set -- $G "$@"; GX=$1; GY=$2; GZ=$3; shift 3
NPATCH=$((GX*GY*GZ))
for ARM in "$@"; do
  LBM=${ARM%T}; EXTRA=""; [ "$ARM" != "$LBM" ] && EXTRA="+LBMetisHostByMsgs"
  TAG=${TAGPFX:-f}${NPATCH}_${NS}_p${LBF:-100}_${ARM}_$(date +%H%M%S)
  POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x $GX -y $GY -z $GZ -r 2 -e 0.05 -i $NS -S 25 -k 1 $EXTRA ${XFLAGS:-} > /dev/null 2>&1
  L=$D/logs/$TAG.log
  ms=$(grep -E "^  step +[0-9]+:" $L | awk -v ns=$NS '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0; } /ms\/step/ { match($0,/ ([0-9.]+) ms\/step/,m); if (st > ns/2) {s+=m[1]; n++} } END{if(n) printf "%.1f", s/n; else print "NA"}')
  avg=$(grep "Average iteration time" $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/')
  hm=$(for S in 2 3; do grep "^\[PELOAD\] step=$S pe=" $L | awk '{split($4,b,"="); h=b[2]+0; sum+=h; n++; if(h>mx)mx=h} END{if(n) printf "%.2f/%.2f ", mx, sum/n}'; done)
  dv=$(grep "measured loads explain" $L | grep Diffusion | sed -E 's/.*device ([0-9]+)%.*/\1/' | sed -n 2,3p | tr '\n' ' ')
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume') | 2nd-half ${ms} ms, whole-run ${avg} ms | host max/avg ${hm}| device% ${dv}"
  grep -i "fatal\|abort\|did not consume" $L | head -2 | cut -c1-160
done
echo "done $(date +%T); node time $(squeue -j $JID -h -o %M)"
