#!/bin/bash
# n1_split6.sh <jobid> (2026-10-12): validation of the CUPTI drain thread + per-device driver share (items 1 and 3):
# sync then async at the three splits (n1_split2.sh / n1_split3.sh, Metis once then Diffusion 25/10, sort 100),
# then the per-event CUPTI timing of the control and the one-process split. Expectation: (1,4,4) and (2,2,4)
# within the draw band of (4,1,4); the one-process CUPTI close no longer 3x the control's. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
echo "=== split6 start $(date +%T), left $(squeue -j $JID -h -o %L), node $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
bash $D/n1_split2.sh $JID
bash $D/n1_split3.sh $JID
echo "--- per-event CUPTI timing (CHARM_LB_CUPTI_TIME=1, 200 steps, sync):"
export NODES=1 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=120 LB=sync LBF=25 LBB=25 LBLAG=10 LBDEBUG=1 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
for cfg in "ctl 4 1" "s14 1 4"; do set -- $cfg; TAG=${1}_P${2}G${3}K4_64_200_cuptiT_$(date +%H%M%S); env CHARM_LB_HELPER=near CHARM_LB_CUPTI_TIME=1 PPN=$2 NP=$2 GPP=$3 PES=$((4*$3)) $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 1.25 -w 1.0 -t 2 -s 0.0022 -V 10 -x 4 -y 4 -z 4 -r 2 -e 0.05 -i 200 -S 25 -k 1 +LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1 > /dev/null 2>&1; L=$D/logs/$TAG.log; echo "$1: $(grep -o 'rc=[0-9]*' $L | tail -1) whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | $(grep '^\[LBCUPTI pe=0\]' $L | sed -E 's/.*flush=([0-9.]+)s process=([0-9.]+)s normalize=([0-9.]+)s total=([0-9.]+)s.*/\1 \2 \3 \4/' | awk '{f+=$1; p+=$2; nn+=$3; t+=$4; n++} END{if(n) printf "CUPTI per event (pe0, mean of %d): flush %.1f process %.1f normalize %.1f total %.1f ms", n, 1000*f/n, 1000*p/n, 1000*nn/n, 1000*t/n}') | launch share: $(grep -oE 'launch [0-9]+%' $L | sort | uniq -c | sort -rn | head -1 | awk '{print $3}')"; grep -iE "fatal|abort|Segmentation" $L | head -2 | cut -c1-160; done
echo "=== split6 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
