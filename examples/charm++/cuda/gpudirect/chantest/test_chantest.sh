#!/bin/bash
# test_chantest.sh <jobid> [iters] [lb_period]: the channel migration test (PERSISTENT_COMM_PLAN.md step 3) in
# the three transports, each first without and then with migration (the two elements swap PEs with migrateMe every
# lb_period iterations; the channels are never re-exchanged by the application). PASS = the example's own
# PASS line (every iteration's buffer verified) with the expected number of migrations. Runs on an allocation
# already held; never sallocs, never scancels.
JID=$1; IT=${2:-24}; LBP=${3:-3}
[ -n "$JID" ] || { echo "usage: test_chantest.sh <jobid> [iters] [lb_period]"; exit 1; }
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/chantest
LCI=/u/bhosale/lci/build-cuda/lib
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ulimit -c 0
mkdir -p $D/logs
NODES_HELD=$(squeue -j $JID -h -o %D 2>/dev/null | head -1)
POOL="+gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm"
arm() {  # <name> <nodes> <np> <per-node> <preload 0|1> <args...>
  local name=$1 nodes=$2 np=$3 ppn=$4 pre=$5; shift 5
  [ "${NODES_HELD:-1}" -ge "$nodes" ] || { echo "skip $name: the allocation holds $NODES_HELD node(s)"; return; }
  local LOG=$D/logs/${name}_$(date +%H%M%S).log
  local -a PRE=(); [ "$pre" = 1 ] && PRE=(LD_PRELOAD=$LCI/liblci.so LD_LIBRARY_PATH=$LCI:$LD_LIBRARY_PATH)
  env "${PRE[@]}" timeout 180 srun --jobid=$JID --overlap --mpi=cray_shasta -N $nodes -n $np --ntasks-per-node=$ppn \
      --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$D \
      stdbuf -oL -eL ./chantest "$@" > $LOG 2>&1
  local rc=$?
  local pass=$(grep -c 'PASS' $LOG) bad=$(grep -ciE 'wrong at index|abort|Fatal|Segmentation|error' $LOG)
  local verdict=FAIL; [ "$rc" = 0 ] && [ "$pass" = 1 ] && [ "$bad" = 0 ] && verdict=PASS
  echo "$(date +%T) $verdict $name: rc=$rc bad=$bad | $(grep -o 'PASS: .*' $LOG | cut -c1-120) | last: $(tail -1 $LOG | cut -c1-100)"
  [ "$verdict" = PASS ] || grep -iE 'wrong at index|abort|Fatal|Segmentation|error|CkDeviceChannel|CkDevicePersistent|LBManager' $LOG | head -5 | cut -c1-200 | sed 's/^/    /'
}
echo "=== chantest, job $JID ($NODES_HELD node(s)), $(date +%T), binary $(ls -la --time-style=+%m-%d_%H:%M $D/chantest | awk '{print $6}'); $IT iterations, LB every $LBP"
for lay in memcpy ipc rdma; do
  case $lay in memcpy) N=1 NP=1 PPN=1 PRE=0 PPNARG="+ppn 2";; ipc) N=1 NP=2 PPN=2 PRE=1 PPNARG="+ppn 1";; rdma) N=2 NP=2 PPN=1 PRE=1 PPNARG="+ppn 1";; esac
  arm ${lay}_plain $N $NP $PPN $PRE -i $IT -s 4096 $PPNARG $POOL
  arm ${lay}_lb    $N $NP $PPN $PRE -i $IT -s 4096 -l -p $LBP $PPNARG $POOL
done
echo "=== chantest done $(date +%T)"
