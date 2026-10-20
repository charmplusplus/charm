#!/bin/bash
# after_imm2.sh <jobid> (18:16): the immediate-data path (put with imm, LCI registering regions with FI_RMA_EVENT)
# on the imm-ON binaries (chantest.immon, leanmd.immon, jacobi3d.immon). chantest rdma first; the leanmd/jacobi3d
# arms run with whichever setting delivered the immediate completions (default, else FI_CXI_OPTIMIZED_MRS=false).
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; C=$G/chantest; cd $L; ulimit -c 0
ctest() {  # <label> [env...]: chantest rdma on chantest.immon, then the immediate-path evidence from its newest logs
  local label=$1; shift
  env CTBIN=chantest.immon "$@" bash $C/test_chantest_bin.sh $JID 24 3 2>&1 | grep -E 'rdma' | sed "s/^/$label /" | cut -c1-160
  for f in $(ls -t $C/logs/rdma_*.log | head -2); do
    echo "  $(basename $f): $(grep -a -m1 'granted caps' $f | cut -c1-90) | lci-cq=$(grep -ac 'lci-cq' $f) rcomp10=$(grep -a 'lci-cq' $f | grep -c 'rcomp=10\|REMOTE_WRITE=1') | $(grep -a 'CHAN-INIT' $f | grep -ai 'immediate\|signal' | head -2 | cut -c1-120 | tr '\n' ';')"
  done
}
ctest default
PASSED=$(ls -t $C/logs/rdma_plain_*.log | head -1 | xargs grep -ac 'PASS:')
IMM=$(ls -t $C/logs/rdma_plain_*.log | head -1 | xargs grep -ai 'immediate' | grep -aci 'received\|arriv\|first')
EXTRA=()
if [ "$PASSED" = 1 ] && [ "$IMM" -ge 1 ]; then echo "immediate path delivered with optimized MRs"
else
  ctest optmrs_off FI_CXI_OPTIMIZED_MRS=false
  P2=$(ls -t $C/logs/rdma_plain_*.log | head -1 | xargs grep -ac 'PASS:'); I2=$(ls -t $C/logs/rdma_plain_*.log | head -1 | xargs grep -ai 'immediate' | grep -aci 'received\|arriv\|first')
  if [ "$P2" = 1 ] && [ "$I2" -ge 1 ]; then echo "immediate path delivered only with FI_CXI_OPTIMIZED_MRS=false"; EXTRA=(FI_CXI_OPTIMIZED_MRS=false)
  else echo "immediate path NOT delivered in either setting; leanmd/jacobi3d imm arms run anyway (2 s spin abort expected)"; fi
fi
export NODES=2 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib PERIOD=20 LAG=16 POOL=16384 LCIDEV=1
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
larm() {  # <channels> <grid> <tag> <tmo> <lb>
  local ch=$1 grid=$2 tag=$3 tmo=$4 lb=$5 TAG=${3}_$(date +%H%M%S); local R=$L/ranklogs/$TAG
  env "${EXTRA[@]}" CHANNELS=$ch LB=$lb BIN=leanmd.immon GRID="$grid" STEPS=100 PPN=4 GPP=1 PES=8 TMO=$tmo $L/run_lmd.sh $JID $TAG > /dev/null 2>&1
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L/logs/$TAG.log | tail -1) steps=$(grep -c '^Step [0-9]* Benchmark Time' $R/rank_0.log) | mean42-100 $(mean_steps $R/rank_0.log 42 100) ms | migr=[$(grep -ah 'cross node migrations' $R/rank_0.log | awk '{printf "%s ",$NF}')] | imm=$(cat $R/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'immediate') watch=$(cat $R/rank_*.log | grep -ac 'CHAN-WATCH') | bad=$(cat $R/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed')"
  cat $R/rank_*.log | grep -aiE 'fatal|abort|Segmentation|signal_bridge|CHAN-STALL' | head -3 | cut -c1-200 | sed 's/^/    /'
}
larm 1 "16 8 8" i_ch_async_16x8x8 150 async
larm 1 "16 8 8" i_ch_none_16x8x8 200 none
larm 1 "8 4 4" i_ch_async_8x4x4 60 async
cd $J; export PPN=4 GPP=1 PES=4 ITERS=200 WARM=10 TMO=120
for g in 512 1024 2048; do for mode in ch zc; do
  TAG=j3di_${mode}_g${g}_$(date +%H%M%S); env "${EXTRA[@]}" BIN=jacobi3d.immon MODE=$mode GRID=$g $J/run_j3d_bin.sh $JID $TAG > /dev/null 2>&1
  R=$J/ranklogs/$TAG/rank_0.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $J/logs/$TAG.log | tail -1) | $(grep -a 'Average iteration time' $R | head -1) | $(grep -a 'Checksum' $R | head -1) | imm=$(cat $J/ranklogs/$TAG/rank_*.log | grep -a 'CHAN-INIT' | grep -aci 'immediate') bad=$(cat $J/ranklogs/$TAG/rank_*.log | grep -aciE 'fatal|abort|Segmentation|CORRUPTION|exceed|CHAN-WATCH')"
done; done
echo "=== imm2 done $(date +%T); node KEPT"
