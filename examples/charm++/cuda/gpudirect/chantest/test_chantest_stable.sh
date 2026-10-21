#!/bin/bash
# test_chantest_stable.sh <jobid> [iters] [lb_period]: the channel migration test on one node in the memcpy
# (one process, 2 PEs) and ipc (two processes) transports: plain, migrateMe moves (-l), the balancer's path
# under +LBAsync (-l -a: the Stable device buffer is snapshotted at the decision, the Mutable one packed;
# both verified after every arrival), and the broken promise (-l -a -x: the Stable buffer is rewritten after
# the decision; the runtime must abort naming it). Runs on an allocation already held; never sallocs, never
# scancels. CHARM_LB_MIGSTATS=1 on the -a arms prints the move-window tally ([mig-window] lines).
JID=$1; IT=${2:-24}; LBP=${3:-3}
[ -n "$JID" ] || { echo "usage: test_chantest_stable.sh <jobid> [iters] [lb_period]"; exit 1; }
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
LCI=/u/bhosale/lci/build-cuda/lib
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ulimit -c 0
mkdir -p $D/logs
POOL="+gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm"
BIN=${CTBIN:-$D/chantest}
arm() {  # <name> <np> <per-node> <preload 0|1> <expect: pass|abort> <args...>
  local name=$1 np=$2 ppn=$3 pre=$4 expect=$5; shift 5
  local LOG=$D/logs/${name}_$(date +%H%M%S).log
  local -a PRE=(); [ "$pre" = 1 ] && PRE=(LD_PRELOAD=$LCI/liblci.so LD_LIBRARY_PATH=$LCI:$LD_LIBRARY_PATH)
  # The Stable promise is checked (hashed at the decision and the pack) only under the migration debug switch.
  [ "$expect" = abort ] && PRE+=(CHARM_DEBUG_MIGRATE=1)
  env "${PRE[@]}" CHARM_LB_MIGSTATS=1 timeout 180 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n $np --ntasks-per-node=$ppn \
      --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$D \
      stdbuf -oL -eL $BIN "$@" > $LOG 2>&1
  local rc=$?
  local pass=$(grep -c 'PASS' $LOG) bad=$(grep -ciE 'wrong at index|abort|Fatal|Segmentation|error' $LOG)
  local verdict=FAIL
  if [ "$expect" = pass ]; then [ "$rc" = 0 ] && [ "$pass" = 1 ] && [ "$bad" = 0 ] && verdict=PASS
  else grep -q 'changed between the balancer.s decision and the pack' $LOG && verdict=PASS; fi
  echo "$(date +%T) $verdict $name (expect $expect): rc=$rc bad=$bad | $(grep -o 'PASS: .*' $LOG | cut -c1-110) | last: $(tail -1 $LOG | cut -c1-100)"
  grep -h '\[mig-window\]\|\[STABLE' $LOG | head -14 | cut -c1-160
  [ "$verdict" = PASS ] || grep -iE 'wrong at index|abort|Fatal|Segmentation|error|CkDeviceChannel|Stable' $LOG | head -6 | cut -c1-200
}
echo "=== chantest stable, job $JID, $(date +%T), binary $(ls -la --time-style=+%m-%d_%H:%M $BIN | awk '{print $6}'); $IT iterations, LB every $LBP"
for lay in memcpy ipc; do
  case $lay in memcpy) NP=1 PPN=1 PRE=0 PPNARG="+ppn 2";; ipc) NP=2 PPN=2 PRE=1 PPNARG="+ppn 1";; esac
  arm ${lay}_plain $NP $PPN $PRE pass  -i $IT -s 4096 $PPNARG $POOL
  arm ${lay}_lb    $NP $PPN $PRE pass  -i $IT -s 4096 -l -p $LBP $PPNARG $POOL
  arm ${lay}_lba   $NP $PPN $PRE pass  -i $IT -s 4096 -l -p $LBP -a $PPNARG $POOL +LBAsync
  arm ${lay}_lbx   $NP $PPN $PRE abort -i $IT -s 4096 -l -p $LBP -a -x $PPNARG $POOL +LBAsync
done
echo "=== chantest stable done $(date +%T)"
