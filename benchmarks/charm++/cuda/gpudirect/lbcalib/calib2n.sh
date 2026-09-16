#!/bin/bash
# calib2n.sh <JOBID> -- 2-node lbcalib: the ONLY launch that exposes inter_node.
# Two runs: (a) default, whose migrate_* is the same-host move and whose
# inter_node_* are finally measured rather than copied; (b) -M 3, whose
# migrate_* is the OFF-NODE move. Comparing (b) against (a)+inter_node says
# whether pricing a network migration as "local move + transfer tier" holds.
# 8 PEs per process, 1 GPU per process -- the ratio leanmd and sph2d run at.
JID=$1; [ -z "$JID" ] && { echo "usage: calib2n.sh <JOBID>"; exit 1; }
D=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/lbcalib
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192; ulimit -c 0
SRUN="srun --jobid=$JID --mpi=cray_shasta -N 2 -n 8 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=16 --cpu-bind=none --exact --kill-on-bad-exit=1"
cd $D || exit 1
run() { # <tag> <extra args>
  local tag=$1; shift
  echo "--- $tag $(date +%T)"
  RANKLOG_DIR=$D/rl_$tag; rm -rf $RANKLOG_DIR; mkdir -p $RANKLOG_DIR
  env RANKLOG_DIR=$RANKLOG_DIR timeout ${TMO:-600} $SRUN stdbuf -oL -eL $D/calibwrap.sh \
      $D/lbcalib -i ${ITERS:-200} -w 20 -o $D/lbcost.2node.$tag.conf "$@" \
      +pe 64 +setcpuaffinity +gpushm +gpupool +gpupoolsize 2048 +gpuipceventpool 256 \
      > $D/calib_$tag.log 2>&1
  echo "    rc=$? $(grep -ah '=== migration' $RANKLOG_DIR/rank_0.log $D/calib_$tag.log 2>/dev/null | head -1)"
  grep -ah "tier\|R2\|not exercised" $RANKLOG_DIR/rank_0.log 2>/dev/null | head -6
}
run same "$@"
run inter -M 3
echo "=== DONE $(date +%T)"
