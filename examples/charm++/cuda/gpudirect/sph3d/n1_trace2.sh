#!/bin/bash
# n1_trace2.sh <jobid>: the same two traces WITHOUT load balancing (CUPTI active through +gputrace alone): does the one-process split pay the same tax when only the tracing subscriber is on?
# split (1,4,4) under sync LB (Metis once then Diffusion, 25/10), 200 steps, one node; then busy % and largest gaps of
# the last second, to see whether the (1,4,4) residual is GPU idle time or longer kernels. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d; NS=${NS:-200}
export NODES=1 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=${TMO:-150}
export LB=none LBDEBUG=1
export DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
declare -A OUTS
run_arm() {  # <pfx> <PPN> <GPP>
  local pfx=$1 ppn=$2 gpp=$3; local pes=$((4*gpp))
  [ "$(left_s)" -gt 120 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  local TAG=gk${NS}_noLB_${pfx}_P${ppn}G${gpp}K4_$(date +%H%M%S); local OUT=/scratch/mzu/bhosale/gputrace/$TAG; mkdir -p $OUT; OUTS[$pfx]=$OUT
  env $ENVC PPN=$ppn NP=$ppn GPP=$gpp PES=$pes $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 1.25 -w 1.0 -t 2 -s 0.0022 -V 10 -x 4 -y 4 -z 4 -r 2 -e 0.05 -i $NS -S 25 -k 1 $XF +gputrace $OUT +gputracenocopy > /dev/null 2>&1
  local L=$D/logs/$TAG.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | files: $(ls $OUT | grep -c '\.proc$') proc"
  python3 $D/gputrace_merge.py --out $OUT/trace_last1s.json --last 1.0 $OUT 2>&1 | tail -1 | cut -c1-160
}
echo "=== trace start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm ctl 4 1
run_arm s14 1 4
echo "--- busy % (all GPUs, last 1 s):"
python3 $D/trace_compare.py ctl=${OUTS[ctl]}/trace_last1s.json s14=${OUTS[s14]}/trace_last1s.json 2>&1 | grep -E "busy|idle|kernel" | cut -c1-200
for p in ctl s14; do echo "--- $p gaps: $(python3 $D/gap_report.py ${OUTS[$p]}/trace_last1s.json 5 2>&1 | grep 'largest gap' | cut -c1-160)"; done
echo "dirs: ${OUTS[ctl]} ${OUTS[s14]}"
echo "=== trace done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
