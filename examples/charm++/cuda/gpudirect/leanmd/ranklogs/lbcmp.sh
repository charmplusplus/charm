#!/bin/bash
# lbcmp.sh <jobid> <reps> [tags...]   tags: <lb>_<mode>, lb in noLB|sync|async, mode in nopool|pool
J=$1; REPS=$2; shift 2
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs; cd $L || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_LB_MIGSTATS=1 PMI_MAX_KVS_ENTRIES=1024
NODES=${NODES:-1}; NT=$((NODES * 4)); PES=$((NT * 8)); CELLS=${CELLS:-6}
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
C="$CELLS $CELLS $CELLS 100 20 20 -computemap local -density gradient +pe $PES +setcpuaffinity +gpushm +gpuipceventpool 256"
REF=4.9319818165E-14; [ "$CELLS" = 8 ] && REF=1.2357385748E-13
OUT=$RL/lbcmp_results.txt; : > $OUT
for r in $(seq 1 $REPS); do for tag in "$@"; do
  lb=${tag%_*}; mode=${tag#*_}
  case $lb in noLB) LB="";; sync) LB="$D";; async) LB="-lbasync -lblag ${LAG:-2} +LBAsync $D";; esac
  if [ $mode = pool ]; then IPC="+gpupool"; ENV="X=1"; else IPC="+gpuipcdirect +gpucommbuffer 256 +gpulbbuffer 512"; ENV="CHARM_MIGRATE_ARENA=1"; fi
  export RANKLOG_DIR=$RL/lbc_${tag}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  env $ENV timeout 90 srun --jobid=$J --mpi=cray_shasta -N $NODES -n $NT --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd $C $IPC $LB >/dev/null 2>&1; rc=$?; sleep 1
  e=$(grep -ho -P 'final \K[0-9.E+-]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1); tot=$(grep -ho -P 'Total application time \K[0-9.]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)
  emig=$(grep -aho -P 'emigrate n=[0-9]+ time=\K[0-9.]+' $RANKLOG_DIR/rank_*.log | sort -g | tail -1)
  pseudo=$(grep -aho -P 'Pseudo LB: \K[0-9.]+' $RANKLOG_DIR/rank_0.log); nbr=$(grep -aho -P 'Neighbor Selection: \K[0-9.]+' $RANKLOG_DIR/rank_0.log)
  spikes=$(grep -a 'Benchmark Time' $RANKLOG_DIR/rank_0.log | awk '{s+=$5; n++; a[n]=$5; st[n]=$2} END {m=s/n; for(i=1;i<=n;i++) if (a[i]>1.35*m) printf "%s:%.0f ", st[i], a[i]}')
  ok=1; [ "$rc" = 0 ] && [ "$e" = "$REF" ] || ok=0
  echo "$tag rep$r ok=$ok rc=$rc total=${tot:-NONE} energy=${e:-NONE} max_emig=${emig:-?} pseudo=${pseudo:-?} nbr=${nbr:-?} spikes=[${spikes}]" | tee -a $OUT
  if [ $ok = 0 ]; then echo "STOP at $tag rep$r; last step: $(grep -a 'Step [0-9]* Benchmark' $RANKLOG_DIR/rank_0.log | tail -1 | cut -c1-30)"; grep -ahnE 'Reason:|Fatal|Abort|outstanding for|WARNING' $RANKLOG_DIR/rank_*.log | head -5 | cut -c1-200; exit 1; fi
done; done
echo "ALL PASSED"
