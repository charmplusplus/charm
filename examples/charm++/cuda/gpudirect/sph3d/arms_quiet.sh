#!/bin/bash
# arms_quiet.sh <jobid> <arm>   -- one arm per call on a held 1-node allocation.
# Env for 2 nodes: NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib TAGP=n2 (run3d.sh reads them).
# Arms (98M, 960 patches): both = new default; noskip = SPH_NO_QUIET_SKIP=1;
# msg = SPH_SENDDONE_MSG=1; base = both switches off (old behaviour). LB via env LB.
JID=$1; ARM=$2; shift 2
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
A98="-X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 12 -y 4 -z 20 -r 2 -e 0.05 -i 600 -S 25 -k 1"
case $ARM in
  both)   ENV="";;
  noskip) ENV="SPH_NO_QUIET_SKIP=1";;
  msg)    ENV="SPH_SENDDONE_MSG=1";;
  base)   ENV="SPH_NO_QUIET_SKIP=1 SPH_SENDDONE_MSG=1";;
  *) echo "unknown arm $ARM"; exit 2;;
esac
TAG=${TAGP:-q}_${ARM}_${LB:-none}
env $ENV POOL=16384 LB=${LB:-none} LBDEBUG=${LBDEBUG:-1} $D/run3d.sh $JID $TAG $A98 "$@" > /dev/null 2>&1
L=$D/logs/$TAG.log
echo "== $TAG: exit $? ; $(grep -c 'check ok' $L) checks ok; $(grep -ci 'fatal\|abort\|error\|did not consume' $L) bad lines"
grep -E "^  step +(100|200|300|400|500|600):" $L | sed -E 's/.*active ([0-9]+\/[0-9]+).*, ([0-9.]+) ms\/step.*quiet-skipped ([0-9]+).*/  step \1 \2 ms\/step skipped \3/'
grep -i "fatal\|abort\|did not consume\|quiet-level proof\|ran more than" $L | head -3
