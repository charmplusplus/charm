#!/bin/bash
# test_persist.sh <jobid>: step-1 correctness of the persistent endpoints (CkDevicePersistent) in the three
# transfer modes, with and without the device pool. Arms: (a) one process with two PEs (MEMCPY), plain and vmm
# pool; (b) two processes on one node (IPC through the import cache), plain, buddy pool, vmm pool; (c) two
# processes on two nodes (one-sided RDMA), plain and vmm pool. The example fills a buffer with the iteration
# number, gets or puts it through the endpoint and verifies every element in the completion callback, so a PASS
# is also the data-visible-at-notification check of the plan (section 2.4). An arm PASSes when the example
# prints PASS for every chare kind it tests (array and group; nodegroup too with two processes) and the log has
# no abort or validation line. Runs on an allocation already held; never sallocs, never scancels.
JID=$1
[ -n "$JID" ] || { echo "usage: test_persist.sh <jobid>"; exit 1; }
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/persistent
LCI=/u/bhosale/lci/build-cuda/lib
export LD_LIBRARY_PATH=/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:${LD_LIBRARY_PATH:-}
export FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ulimit -c 0
mkdir -p $P/logs
NODES_HELD=$(squeue -j $JID -h -o %D 2>/dev/null | head -1)
arm() {  # <name> <nodes> <np> <per-node> <preload the vmm-fixed LCI: 0|1> <program args...>
  local name=$1 nodes=$2 np=$3 ppn=$4 pre=$5; shift 5
  [ "${NODES_HELD:-1}" -ge "$nodes" ] || { echo "skip $name: the allocation holds $NODES_HELD node(s)"; return; }
  local LOG=$P/logs/${name}_$(date +%H%M%S).log
  local -a PRE=(); [ "$pre" = 1 ] && PRE=(LD_PRELOAD=$LCI/liblci.so LD_LIBRARY_PATH=$LCI:$LD_LIBRARY_PATH)
  env "${PRE[@]}" timeout 150 srun --jobid=$JID --overlap --mpi=cray_shasta -N $nodes -n $np --ntasks-per-node=$ppn \
      --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$P \
      stdbuf -oL -eL ./persistent -i 10 -s 4096 "$@" > $LOG 2>&1
  local rc=$?
  local want=2; [ "$np" -gt 1 ] && want=3
  local pass=$(grep -c '^PASS' $LOG) bad=$(grep -ciE 'Validation failure|abort|Fatal|Segmentation|error' $LOG)
  local verdict=FAIL; [ "$rc" = 0 ] && [ "$pass" = "$want" ] && [ "$bad" = 0 ] && verdict=PASS
  echo "$(date +%T) $verdict $name: rc=$rc pass=$pass/$want bad=$bad elapsed=$(grep -oP 'Elapsed: \K[0-9.]+' $LOG)s log=$(basename $LOG)"
  [ "$verdict" = PASS ] || grep -iE 'Validation failure|abort|Fatal|Segmentation|error|CkDevicePersistent|registered' $LOG | head -4 | cut -c1-200 | sed 's/^/    /'
}
echo "=== persistent step-1 tests, job $JID ($NODES_HELD node(s)), $(date +%T), binary $(ls -la --time-style=+%m-%d_%H:%M $P/persistent | awk '{print $6}')"
echo "--- (a) one process, two PEs: MEMCPY"
arm memcpy_plain 1 1 1 0 +ppn 2 +gpushm
arm memcpy_vmm   1 1 1 0 +ppn 2 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm
echo "--- (b) two processes, one node: IPC through the import cache"
arm ipc_plain 1 2 2 0 +ppn 1 +gpushm
arm ipc_buddy 1 2 2 0 +ppn 1 +gpushm +gpupool +gpupoolsize 256
arm ipc_vmm   1 2 2 0 +ppn 1 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm
echo "--- (c) two processes, two nodes: one-sided RDMA"
arm rdma_plain 2 2 1 1 +ppn 1 +gpushm
arm rdma_vmm   2 2 1 1 +ppn 1 +gpushm +gpupool +gpupoolsize 256 +gpupoolalloc vmm
echo "=== done $(date +%T)"
