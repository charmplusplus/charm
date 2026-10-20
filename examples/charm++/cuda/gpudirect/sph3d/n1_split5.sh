#!/bin/bash
# n1_split5.sh <jobid>: control vs (1,4,4) under sync LB with the host profile on (SPH_HOST_PROFILE=1), to see which host phase or transfer tier carries the (1,4,4) residual.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
NS=${NS:-500}
export NODES=1 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=${TMO:-400}
export LB=sync LBF=25 LBB=25 LBLAG=10 LBDEBUG=1
export DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="SPH_HOST_PROFILE=1 CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <PPN> <GPP>
  local pfx=$1 ppn=$2 gpp=$3; local pes=$((4*gpp))
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_64_${NS}_syncH_$(date +%H%M%S)
  env $ENVC PPN=$ppn NP=$ppn GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 1.25 -w 1.0 -t 2 -s 0.0022 -V 10 -x 4 -y 4 -z 4 -r 2 -e 0.05 -i $NS -S 25 -k 1 $XF > /dev/null 2>&1
  local L=$D/logs/$TAG.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION') | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}') | cross-device moves: $(grep -ciE 'staged|peer' $L) lines | $(grep -m1 'HAPI> Config' $L | sed 's/HAPI> Config: //' | cut -c1-70)"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|cuMemSetAccess for" $L | head -3 | cut -c1-200
  grep -q "rc=0" $L
}
echo "=== split5 start $(date +%T), left $(squeue -j $JID -h -o %L), node $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm ctl 4 1

run_arm s14 1 4
echo "--- whole-run ms/step: $(for p in ctl s22 s14; do L=$(ls -t $D/logs/${p}_P*_64_${NS}_syncH_*.log 2>/dev/null | head -1); [ -n "$L" ] && printf '%s=%s ' $p $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/'); done)"
echo "=== split5 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
