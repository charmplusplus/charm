#!/bin/bash
# verify_stash.sh <jobid> [tags...]  -- runs the unverified-stash checks in order, stops at first failure
J=$1; shift
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs
cd $L || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_LB_MIGSTATS=1 CHARM_MD_ALLOCSTATS=1 CHARM_ZC_STATS=1
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
NOPOOL="+gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 512"
POOL="+gpupool"
ASYNC8="-lbasync -lblag 8 +LBAsync"
E=4.9319818165E-14
run() {
  local tag=$1; local ipc=$2; local lb=$3; shift 3
  export RANKLOG_DIR=$RL/vs_$tag; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  local t0=$(date +%s)
  env "$@" timeout 90 srun --jobid=$J --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $PWD/leanmd 6 6 6 100 20 20 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipceventpool 256 $ipc $lb $D >/dev/null 2>&1
  local rc=$?; sleep 1
  local e=$(grep -ho -P 'final \K[0-9.E+-]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)
  local tot=$(grep -ho -P 'Total application time \K[0-9.]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)
  local verdict=PASS; { [ "$rc" != "0" ] || [ "$e" != "$E" ]; } && verdict=FAIL
  echo "##### $tag ($*): $verdict rc=$rc wall=$(( $(date +%s)-t0 ))s total=${tot:-NONE} energy=${e:-NONE} #####"
  echo "  arenas: $(grep -ahc 'device pool: arena' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)  published: $(grep -ahc 'published for pre-open' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)  pre-opened: $(grep -ahc 'pre-opened peer' $RANKLOG_DIR/rank_*.log | paste -sd+ | bc)"
  grep -ah 'HAPI> Device pool on\|HAPI> No GPU communication\|HAPI> Cross-process device sends\|ignored' $RANKLOG_DIR/rank_0.log | head -4 | sed 's/^/  /'
  grep -ah 'ipc-stats' $RANKLOG_DIR/rank_*.log | head -4 | sed 's/^/  /'
  echo "  max emigrate: $(grep -ho -P 'emigrate n=[0-9]+ time=\K[0-9.]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|sort -g|tail -1)   allocstats: $(grep -aho 'device_alloc_free_time=[0-9.]*s calls=[0-9]*' $RANKLOG_DIR/rank_*.log | tr '\n' ' ')"
  if [ $verdict = FAIL ]; then
    echo "  --- FAILURE DETAIL ---"
    grep -ahnE 'Reason:|Fatal|Abort|abort|outstanding for|Segmentation|corrupt|mismatch|hapiDevPool|CkDevice(Free|Malloc)|Err [0-9]' $RANKLOG_DIR/rank_*.log | head -12 | sed 's/^/  /'
    for f in $RANKLOG_DIR/rank_*.log; do echo "  == $(basename $f) last 3:"; tail -3 $f | cut -c1-200 | sed 's/^/     /'; done
    echo "STOPPED at $tag"; exit 1
  fi
}
for tag in "$@"; do
  case $tag in
    nopool)  run nopool  "$NOPOOL" "$ASYNC8" CHARM_MIGRATE_ARENA=1 ;;
    pool)    run pool    "$POOL" "$ASYNC8" ;;
    pool_dbg) run pool_dbg "$POOL" "$ASYNC8" CHARM_ZC_RESTAGE_DEBUG=1 ;;
    pool2)   run pool2   "$POOL" "$ASYNC8" ;;
    pool3)   run pool3   "$POOL" "$ASYNC8" ;;
    pool_lag2) run pool_lag2 "$POOL" "-lbasync -lblag 2 +LBAsync" ;;
    pool_sync) run pool_sync "$POOL" "" ;;
    *) echo "unknown tag $tag"; exit 2 ;;
  esac
done
echo "ALL PASSED: $*"
