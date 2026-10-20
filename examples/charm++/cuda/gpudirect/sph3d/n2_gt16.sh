#!/bin/bash
# one gputrace arm: e2 (gate-free close) (1,4,4) async, in the tail of job 22958581
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
JID=22958581
XF="+LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
cp -f $D/sph3d.e2 $D/sph3d; echo "--- binary now sph3d.e2 $(date +%T)"
OUT=/scratch/mzu/bhosale/gputrace/gk16b_s14_e2_$(date +%H%M%S); mkdir -p $OUT
TAG=s14_P1G4K4_256_1000_asyncCfGTe2_$(date +%H%M%S)
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768 TMO=600 LBF=25 LBB=25 LBLAG=10 DIFF_FLAGS="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
env CHARM_LB_HELPER=near LCIDEV=1 LBDEBUG=1 CHARM_LB_CUPTI_PERIODIC=0 LB=async PPN=1 NP=2 GPP=4 PES=16 $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i 1000 -S 25 -k 1 $XF +gputracenocopy +gputrace $OUT > /dev/null 2>&1
echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $D/logs/$TAG.log | tail -1) checks=$(grep -c 'check ok' $D/logs/$TAG.log) bad=$(grep -ciE 'fatal|abort|Segmentation|CORRUPTION' $D/logs/$TAG.log) | $(grep -o 'whole-run [0-9.]* ms' $D/logs/$TAG.log | head -1) | trace files $(ls $OUT | wc -l)"
cp -f $D/sph3d.v17 $D/sph3d; echo "--- binary now sph3d.v17"
echo "OUT=$OUT"
( cd $D && timeout 900 python3 gputrace_merge.py --out $OUT/trace_last2.5s.json --last 2.5 $OUT 2>&1 | tail -n 6 | sed 's/^/    merge: /' )
( cd $D && timeout 300 python3 gap_report.py $OUT/trace_last2.5s.json 3 2>&1 | head -n 60 | sed 's/^/    gap: /' )
echo "=== gt16 done $(date +%T); node KEPT"
