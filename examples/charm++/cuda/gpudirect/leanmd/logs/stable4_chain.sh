#!/bin/bash
# stable4_chain.sh <jobid>: (1) jacobi3d on the rebuilt runtime, one node, zero-copy and channels at 512^3 (the
# channel protocol's steady-state path, checksums); (2) two more runs each of the three channel arms whose n=2
# ablation was not decisive: previous build, one migration stream, senders not told.
JID=$1; [ -n "$JID" ] || exit 1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d
echo "=== stable4, job $JID, $(date +%T)"
for m in zc ch; do
  tag=s4_j3d_${m}_$(date +%H%M%S)
  NODES=1 PPN=4 GPP=1 PES=4 MODE=$m GRID=512 ITERS=200 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=240 bash $J/run_j3d.sh $JID $tag > /dev/null 2>&1
  echo "$(date +%T) $tag: $(grep -o 'rc=[0-9]*' $J/logs/$tag.log | tail -1) | $(grep -aiE 'us/iter|ms/iter|Average' $J/logs/$tag.log | tail -1 | cut -c1-100) | $(grep -ai 'checksum' $J/logs/$tag.log | tail -1 | cut -c1-80)"
done
mean_steps() { awk -v lo=$2 -v hi=$3 '/^Step [0-9]+ Benchmark Time/{n=$2+0;t=$5+0;if(n>=lo&&n<=hi){s+=t;c++}} END{if(c)printf "%.1f",s/c; else printf "NA"}' "$1" 2>/dev/null; }
lmd() {  # <tag> <bin> <channels 0|1>
  local tag=${1}_$(date +%H%M%S) bin=$2 ch=$3
  CHARM_LB_MIGSTATS=1 NODES=1 PPN=4 GPP=1 PES=8 GRID="8 4 4" STEPS=100 LB=async CHANNELS=$ch BIN=$bin POOL_ALLOC=vmm POOL=16384 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=300 \
    bash $L/run_lmd.sh $JID $tag > /dev/null 2>&1
  local rc=$?; local R=$L/ranklogs/$tag/rank_0.log
  echo "$(date +%T) $tag: rc=$rc steps=$(grep -c '^Step [0-9]* Benchmark Time' $R 2>/dev/null) | mean42-100 $(mean_steps $R 42 100) ms | migr=[$(grep -aoE '[0-9]+ migration\(s\)' $R 2>/dev/null | grep -oE '^[0-9]+' | head -4 | tr '\n' ' ')]"
}
for r in 4 5; do
  lmd s4_prev_ch_r$r leanmd.prev 1
  lmd s4_onestream_ch_r$r leanmd.onestream 1
  lmd s4_noinform_ch_r$r leanmd.noinform 1
done
echo "=== stable4 done $(date +%T)"
