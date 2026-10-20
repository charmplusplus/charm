#!/bin/bash
# n2_split.sh <jobid> (2026-10-12, Phase 2): the process/GPU split on TWO nodes at the real configuration
# (256 patches "8 4 8", 98M particles, sort 100): (4,1,4) = 8 processes (the control), (2,2,4), (1,4,4),
# each noLB, sync and async (Metis once then Diffusion, period 25 lag 10). One LCI device per GPU group
# (run3d.sh). Kill criterion: fault, checks mismatch, or a split above 1.10 x the control. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d; NS=${NS:-1000}
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=${TMO:-600}
export LBF=25 LBB=25 LBLAG=10 LBDEBUG=1 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <mode> <pfx> <PPN> <GPP>
  local mode=$1 pfx=$2 ppn=$3 gpp=$4; local pes=$((4*gpp)); local np=$((2*ppn))
  [ "$(left_s)" -gt 240 ] || { echo "skip $pfx $mode: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${NS}_${mode}_$(date +%H%M%S)
  env $ENVC LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $NS -S 25 -k 1 $XF > /dev/null 2>&1
  local L=$D/logs/$TAG.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) bad=$(grep -ci 'fatal\|abort\|did not consume\|Segmentation\|CORRUPTION') | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$NS '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | $(grep -m1 'HAPI> Config' $L | sed 's/HAPI> Config: //' | cut -c1-60) | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | nodes $(grep -m1 -oE 'over [0-9]+ nodes' $L)"
  grep -iE "fatal|abort|Segmentation|did not consume|CORRUPTION|cuMemSetAccess for" $L | head -3 | cut -c1-200
  grep -q "rc=0" $L
}
echo "=== split2n start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
for mode in none sync async; do run_arm $mode ctl 4 1; run_arm $mode s22 2 2; run_arm $mode s14 1 4; done
echo "--- whole-run ms/step:"; for mode in none sync async; do echo "  $mode: $(for p in ctl s22 s14; do L=$(ls -t $D/logs/${p}_P*_256_${NS}_${mode}_*.log 2>/dev/null | head -1); [ -n "$L" ] && printf '%s=%s ' $p $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/'); done)"; done
echo "=== split2n done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
