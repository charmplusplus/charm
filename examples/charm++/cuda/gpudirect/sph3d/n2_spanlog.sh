#!/bin/bash
# n2_spanlog.sh (2026-10-12): the entry-span log (hapi_impl.cpp CuptiEntryLog) replacing CUPTI's per-entry
# external-correlation push/pop, on the same two nodes as devclose*. launchcost (devclose3_chain.out): a driver call
# under tracing costs 32.8 us in a 16-thread process with the push/pop, 7.1 us without it, 5.0 us in a 4-thread
# process. Arm 1 checks attribution still works (device share of the measured load, Metis moves) and prints the
# launch-path tally; then alternating plain async ctl / s14 / s22 while time remains. Kill criterion: fault, checks
# mismatch, or device share < 30% in arm 1 (attribution broken). Keeps the node.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
JID=$(grep -o 'jobid=[0-9]*' $D/logs/devclose_chain.out | head -1 | cut -d= -f2)
[ -n "$JID" ] || { echo "no jobid"; exit 1; }
cd $D
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
DEVSHARE=0
summary() {  # <log> <ns>
  local L=$1 ns=$2
  DEVSHARE=$(grep -m1 -h '\[DiffusionLB\] measured loads explain' $L | sed -E 's/.*device ([0-9]+)%.*/\1/'); [ -n "$DEVSHARE" ] || DEVSHARE=0
  echo "$(date +%T) $(basename $L .log): $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$ns '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | wall@25 $(grep -E '^  step +25:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') @50 $(grep -E '^  step +50:' $L | sed -E 's/.*wall ([0-9.]+) s.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | device-share ${DEVSHARE}% (metis: $(grep -m1 -h '\[MetisLB\] measured loads explain' $L | sed -E 's/.*device ([0-9]+)%.*/\1/')%) | close $(grep '^\[LBCUPTI pe=0\] flush' $L | sed -E 's/.*total=([0-9.]+)s.*/\1/' | awk '{t+=$1; n++} END{if(n) printf "%.1f ms x%d", 1000*t/n, n}')"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|too small" $L | head -2 | cut -c1-160
}
run_arm() {  # <mode> <pfx> <PPN> <GPP> <suffix> <ns> <env...>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4 sfx=$5 ns=$6; shift 6
  local pes=$((4*gpp)) np=$((2*ppn))
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx $mode $sfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}${sfx}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 "$@" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF > /dev/null 2>&1
  summary $D/logs/$TAG.log $ns
  LASTLOG=$D/logs/$TAG.log
}
echo "=== spanlog start $(date +%T), job $JID, left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
echo "--- 1. attribution check + launch-path tally (s14 async)"
run_arm async s14 1 4 Sl 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=1
echo "    launch path (PE 0, closes 2-4): $(grep -h '^\[LBCUPTI pe=0\]   records' $LASTLOG | sed -n 2,4p | sed -E 's/.*records //' | tr '\n' ';')"
echo "    per-shard attribution (first 2 lines): $(grep -h 'hapiProcessCuptiBuffers kernels=' $LASTLOG | head -2 | sed -E 's/.*kernels=/kernels=/' | cut -c1-150 | tr '\n' ';')"
if [ "$DEVSHARE" -lt 30 ]; then echo "!!! ATTRIBUTION BROKEN (device share ${DEVSHARE}%): stopping; node KEPT"; exit 1; fi
echo "--- 2. alternating plain async ctl / s14 / s22"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 5 ]; do i=$((i+1))
  run_arm async ctl 4 1 Sl 1000; run_arm async s14 1 4 Sl 1000; run_arm async s22 2 2 Sl 1000
done
echo "--- 3. sync, one each, if time remains"
run_arm sync ctl 4 1 Sl 1000; run_arm sync s14 1 4 Sl 1000
echo "=== spanlog done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
