#!/bin/bash
# n2_spanlog6.sh: last minutes of job 22853463 -- one timed, LB-debug arm each (ctl, s14 async, 1000 steps) for the
# per-event LB timeline with the span-log binary. Never scancel; node KEPT.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d; JID=22853463; cd $D
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=300
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() { local mode=$1 pfx=$2 ppn=$3 gpp=$4 sfx=$5 ns=$6; shift 6; local pes=$((4*gpp)) np=$((2*ppn))
  [ "$(left_s)" -gt 120 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  local TAG=${pfx}_P${ppn}G${gpp}K4_256_${ns}_${mode}${sfx}_$(date +%H%M%S)
  env CHARM_LB_HELPER=near LCIDEV=1 "$@" LB=$mode PPN=$ppn NP=$np GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $ns -S 25 -k 1 $XF > /dev/null 2>&1
  local L=$D/logs/$TAG.log; echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') | metis $(grep -m1 -oE 'MetisLB.*#Objects migrating: [0-9]+' $L | grep -oE '[0-9]+$') | close $(grep '^\[LBCUPTI pe=0\] flush' $L | sed -E 's/.*total=([0-9.]+)s.*/\1/' | awk '{t+=$1; n++} END{if(n) printf "%.1f ms x%d", 1000*t/n, n}')"; }
echo "=== spanlog6 start $(date +%T), left $(squeue -j $JID -h -o %L)"
run_arm async ctl 4 1 Dbg6 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=1
run_arm async s14 1 4 Dbg6 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=1
run_arm sync s14 1 4 Dbg6 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=1
run_arm sync ctl 4 1 Dbg6 1000 CHARM_LB_CUPTI_TIME=1 LBDEBUG=1
echo "=== spanlog6 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
