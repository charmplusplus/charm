#!/bin/bash
# conn1_j3d.sh <jobid>: jacobi3d 512^3, one node, 16 blocks, zero-copy and channels at the runtime default (32 queues),
# two runs each, then one pair with CONNECTIONS=8 in the environment as the same-session reference for this morning's
# 8-queue numbers (zc 1633.7 us, ch 1412.6 us). Waits for the conn1 chain and the placement check to finish first so
# nothing overlaps on the GPUs. Never sallocs, never scancels.
JID=$1; [ -n "$JID" ] || exit 1
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd
J=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi3d
until grep -q 'conn1 chain done\|conn1 chain stopped' $L/logs/conn1_chain2.out 2>/dev/null && grep -q 'RESULT' $L/logs/conn1_preset2_context.log 2>/dev/null; do sleep 10; done
echo "=== conn1 jacobi3d, job $JID, $(date +%T)"
j3d() {  # <tag> <mode> [CONNECTIONS]
  local tag=${1}_$(date +%H%M%S) m=$2 conn=$3
  local envc=(); [ -n "$conn" ] && envc=(CUDA_DEVICE_MAX_CONNECTIONS=$conn)
  env "${envc[@]}" NODES=1 PPN=4 GPP=1 PES=4 MODE=$m GRID=512 ITERS=200 LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=240 bash $J/run_j3d.sh $JID $tag > /dev/null 2>&1
  local banner=$(grep -ah 'HAPI> Config:' $J/ranklogs/$tag/rank_0.log $J/logs/$tag.log 2>/dev/null | head -1 | sed -E 's/.*map (block|roundrobin), //')
  echo "$(date +%T) $tag: $(grep -o 'rc=[0-9]*' $J/logs/$tag.log | tail -1) | $(grep -aiE 'us/iter|ms/iter|Average' $J/logs/$tag.log | tail -1 | cut -c1-100) | $(grep -ai 'checksum' $J/logs/$tag.log | tail -1 | cut -c1-80) | queues: ${banner:-NO BANNER}"
}
for r in 1 2; do j3d c1_j3d_zc32_r$r zc ""; j3d c1_j3d_ch32_r$r ch ""; done
j3d c1_j3d_zc8 zc 8; j3d c1_j3d_ch8 ch 8
echo "=== conn1 jacobi3d done $(date +%T)"
