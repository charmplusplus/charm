#!/bin/bash
# One-node version: run pic2d with LB until it stalls, then dump every stack.
J=$1; N=${2:-4}; ARM=${3:-sync}
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; RL=$P/ranklogs; cd $P || exit 1; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024 CHARM_ZC_STALL_SECS=10
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
[ "$ARM" = async ] && LB="-f 10 -b 20 -a -l 2 +LBAsync $D" || LB="-f 10 -b 20 $D"
for r in $(seq 1 $N); do
  export RANKLOG_DIR=$RL/h1_$r; mkdir -p $RANKLOG_DIR; rm -f $RANKLOG_DIR/*
  env X=1 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact \
      stdbuf -oL -eL $P/pic2d -W 1024 -H 1024 -w 128 -h 128 -i 60 -u 10 -c 5 $LB \
      +gpushm +gpupool +gpuipceventpool 256 +ppn 8 > $RANKLOG_DIR/out.log 2>&1 &
  pid=$!; t=0
  while kill -0 $pid 2>/dev/null && [ $t -lt 40 ]; do sleep 1; t=$((t+1)); done
  if kill -0 $pid 2>/dev/null; then
    echo "run $r ($ARM): STALLED after ${t}s; last line: $(grep -a Iter $RANKLOG_DIR/out.log | tail -1)"
    srun --jobid=$J --overlap -N1 -n1 bash -c 'for p in $(pgrep -x pic2d); do echo "=== pid $p"; gdb -p $p -batch -ex "thread apply all bt 12" 2>&1 | grep -E "^Thread|^#"; done' > $RANKLOG_DIR/stacks.txt 2>&1
    echo "stack lines: $(wc -l < $RANKLOG_DIR/stacks.txt)"
    pkill -P $pid; kill $pid 2>/dev/null; sleep 3
    echo "STALLDIR=$RANKLOG_DIR"; exit 2
  fi
  wait $pid; echo "run $r ($ARM): completed rc=$? avg=$(grep -aoP 'Average iteration time: \K[0-9.]+' $RANKLOG_DIR/out.log)"
done
echo "no stall in $N runs"
