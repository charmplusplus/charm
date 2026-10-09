#!/bin/bash
# n2_chain2.sh <jobid>: allocation 2 -- TNL 98M regen on node A while sph3d 1-node references run on node B;
# then TNL 98M 8 ranks (LB on / off, 2000 steps); then sph3d 98M 256 patches Metis-only sync/async; then 25M.
JID=$1
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
T=/u/bhosale/tnl-sph/examples/WCSPH-DBC/tank3D_d8
NODES_LIST=$(squeue -j $JID -h -o %N); A=$(scontrol show hostnames $NODES_LIST | sed -n 1p); B=$(scontrol show hostnames $NODES_LIST | sed -n 2p)
echo "chain2 start $(date +%T) nodes A=$A B=$B"
# phase L: LeanMD non-regression first (both nodes, ~8 min): weak N=2 nolb/sync/async vs 1515.7/909.4/904.4 (2026-09-17)
( cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect && STEPS=100 PERIOD=20 LAG=16 bash scale.sh $JID 2 leanmd weak "nolb sync async" 2>&1 | grep -E "^leanmd|rc=|FAIL|Fatal|abort|Cannot" | cut -c1-200 )
echo "leanmd done $(date +%T) node time $(squeue -j $JID -h -o %M)"
# phase 0: TNL 98M init on A (background), sph3d 1-node refs on B
( cd $T && INIT_NODE=$A TMO=1500 ./run_ranks.sh $JID tnl98_init 0.0022 2000 8 2 init-only > $D/logs/tnl98_init.out 2>&1; echo "tnl init rc=$? $(date +%T)" ) &
INITPID=$!
export POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib
for ARM in syncT asyncT; do
  LBM=${ARM%T}
  TAG=ref1n_98M_p128_${ARM}
  NODES=1 NP=4 NODELIST=$B POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 4 -r 2 -e 0.05 -i 600 -S 25 -k 1 +LBMetisHostByMsgs > /dev/null 2>&1
  L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | 2nd-half $(grep -E '^  step +(325|350|375|400|425|450|475|500|525|550|575|600):' $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{printf "%.1f", s/n}') ms whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms"
done
for ARM in none syncT asyncT; do
  LBM=${ARM%T}; EX=""; [ "$ARM" != "$LBM" ] && EX="+LBMetisHostByMsgs"
  TAG=ref1n_25M_p64_${ARM}
  NODES=1 NP=4 NODELIST=$B POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0035 -V 10 -x 4 -y 4 -z 4 -r 2 -e 0.05 -i 600 -S 25 -k 1 $EX > /dev/null 2>&1
  L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | 2nd-half $(grep -E '^  step +(325|350|375|400|425|450|475|500|525|550|575|600):' $L | sed -E 's/.* ([0-9.]+) ms\/step.*/\1/' | awk '{s+=$1;n++} END{printf "%.1f", s/n}') ms whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms"
done
echo "waiting for TNL init $(date +%T)"; wait $INITPID; tail -2 $D/logs/tnl98_init.out | cut -c1-160
# phase 1: TNL 98M 8 ranks, 2000 steps, LB on / off (config variants regenerated from the NEW config)
python3 - <<'PY'
import re
p="/scratch/mzu/bhosale/tnl-sph/d8_sources/config.jsonc"; s=open(p).read(); open(p+".orig98","w").write(s)
dt=float(re.search(r'"initial-time-step":\s*([0-9.eE+-]+)', s).group(1)); steps=2000
for name, lb in (("lb100_2000", 100), ("nolb_2000", 1000000)):
    t=re.sub(r'"final-time":\s*[0-9.eE+-]+', f'"final-time": {steps*dt:.8e}', s)
    t=re.sub(r'"load-balancing-step-inteval":\s*\d+', f'"load-balancing-step-inteval": {lb}', t); open(p+"."+name,"w").write(t)
print("tnl 98M configs for", steps, "steps, dt", dt)
PY
sed -i 's|cp $C.orig $C|cp $C.orig98 $C|' $T/run_variant.sh
$T/run_variant.sh $JID tnl98_lb100_2000 lb100_2000
$T/run_variant.sh $JID tnl98_nolb_2000 nolb_2000
echo "tnl done $(date +%T) node time $(squeue -j $JID -h -o %M)"
# phase 2: sph3d 98M 256 patches, 2000 steps, Metis alone at every event (period 100), sync/async x2
DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 $D/n2_final.sh $JID "8 4 8" 2000 syncT asyncT syncT asyncT
# phase 3: 25M on 2 nodes, 128 patches, 600 steps
export NODES=2 NP=8
for ARM in none syncT asyncT; do
  LBM=${ARM%T}; EX=""; [ "$ARM" != "$LBM" ] && EX="+LBMetisHostByMsgs"
  TAG=f25M_p128_${ARM}
  POOL=16384 LB=$LBM LBDEBUG=1 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0035 -V 10 -x 4 -y 4 -z 8 -r 2 -e 0.05 -i 2000 -S 25 -k 1 $EX > /dev/null 2>&1
  L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms"
done
echo "chain2 done $(date +%T) node time $(squeue -j $JID -h -o %M)"
