#!/bin/bash
# n2_zcstats.sh <jobid> [steps]: step 0 of PERSISTENT_COMM_PLAN.md, no new code. One sph3d async control arm
# (4,1,4) with CHARM_ZC_STATS=1, which makes every process print at exit its per-message host costs on the
# device zero-copy path ([zc-host]: send prepare and receive issue, mean us per message; [zc-time]: per mode;
# [zc-stats]: mode counts). Multiplied by the messages per PE per step, that is the upper bound of what a
# persistent channel can save on the host. Runs on an allocation already held; never sallocs, never scancels.
JID=$1; NS=${2:-1000}
[ -n "$JID" ] || { echo "usage: n2_zcstats.sh <jobid> [steps]"; exit 1; }
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
cd $D
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600
export LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
TAG=zcstats_ctl_P4G1K4_256_${NS}_async_$(date +%H%M%S)
echo "=== zcstats $(date +%T): sph3d async control, $NS steps, CHARM_ZC_STATS=1"
env CHARM_ZC_STATS=1 CHARM_LB_HELPER=near LCIDEV=1 LBDEBUG=1 CHARM_LB_CUPTI_PERIODIC=0 LB=async PPN=4 NP=8 GPP=1 PES=4 \
  $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $NS -S 25 -k 1 $XF > /dev/null 2>&1
L=$D/logs/$TAG.log
echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) | whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | 2nd-half $(grep -E '^  step +[0-9]+:' $L | awk -v ns=$NS '{for(i=1;i<=NF;i++) if($i=="step") st=$(i+1)+0} /ms\/step/ {match($0,/ ([0-9.]+) ms\/step/,x); if (st>ns/2){s+=x[1]; n++}} END{if(n) printf "%.2f", s/n}') | last: $(tail -1 $L | cut -c1-80)"
echo "--- [zc-host] per process (send: n, mean us | recv-issue: n, mean us | done):"
grep -a '^\[zc-host\]' $L | cut -c1-220 | sed 's/^/    /'
echo "--- [zc-time] per process and mode:"
grep -a '^\[zc-time\]' $L | cut -c1-200 | sed 's/^/    /' | head -24
echo "--- [zc-stats] mode counts per process:"
grep -a '^\[zc-stats\]' $L | cut -c1-200 | sed 's/^/    /'
echo "--- [zc-dest] unresolved destinations per process:"
grep -a '^\[zc-dest\]' $L | cut -c1-160 | sed 's/^/    /'
echo "=== zcstats done $(date +%T)"
