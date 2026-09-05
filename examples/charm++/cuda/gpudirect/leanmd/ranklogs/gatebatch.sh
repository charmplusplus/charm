#!/bin/bash
# gatebatch.sh <jobid> [reps]  -- paired host-wait vs CHARM_LB_POOL_EVENT_GATE runs; a RATE measurement, does not stop at failures
J=$1; REPS=${2:-32}
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; RL=$L/ranklogs; OUT=$RL/gatebatch_results.txt; : > $OUT
cd $L || exit 1; ulimit -c 0
cp $L/leanmd $L/leanmd-gatebatch
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH" FI_MR_CACHE_MONITOR=disabled CHARM_MIGRATE_ARENA=1
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim +gpulbbuffer 512"
C="6 6 6 100 20 20 -computemap local -density gradient +pe 32 +setcpuaffinity +gpushm +gpuipcdirect +gpuipceventpool 256 +gpucommbuffer 256"
for r in $(seq 1 $REPS); do for mode in sync gate; do
  if [ $mode = gate ]; then export CHARM_LB_POOL_EVENT_GATE=1; else unset CHARM_LB_POOL_EVENT_GATE; fi
  export RANKLOG_DIR=$RL/gb_${mode}_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  timeout 90 srun --jobid=$J --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL $RL/rankwrap.sh $L/leanmd-gatebatch $C -lbasync -lblag 8 +LBAsync $D >/dev/null 2>&1; rc=$?; sleep 1
  e=$(grep -ho -P 'final \K[0-9.E+-]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1); c=$(grep -ahc 'SIMULATION SUCCESSFULL' $RANKLOG_DIR/rank_*.log 2>/dev/null|paste -sd+|bc)
  tot=$(grep -ho -P 'Total application time \K[0-9.]+' $RANKLOG_DIR/rank_*.log 2>/dev/null|head -1)
  ok=0; [ "$rc" = "0" ] && [ "${c:-0}" -ge 1 ] && [ "$e" = "4.9319818165E-14" ] && ok=1
  why=""; [ $ok = 0 ] && why=$(grep -ahoiE 'Reason: [^,]*|Fatal CUDA Error[^,]*|outstanding for [0-9]*s' $RANKLOG_DIR/rank_*.log|head -1|cut -c1-90)
  echo "$mode rep$r ok=$ok rc=$rc total=${tot:-NONE} energy=${e:-NONE} $why" >> $OUT
done; done
echo "DONE sync_ok=$(grep -c '^sync .*ok=1' $OUT) sync_bad=$(grep -c '^sync .*ok=0' $OUT) gate_ok=$(grep -c '^gate .*ok=1' $OUT) gate_bad=$(grep -c '^gate .*ok=0' $OUT)" >> $OUT
