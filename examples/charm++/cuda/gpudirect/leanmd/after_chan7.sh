#!/bin/bash
# after_chan7.sh <jobid>: after chan7's six leanmd arms: stop the chain's loop (my own runner), then on the leftover
# minutes: (1) bare converse send with ONE LCI device (+lci_ndevices 1) against the default, two nodes and one node;
# (2) a burst shorter than the backend's send queue (-b 32) to separate the per-send cost from queue back-pressure;
# (3) the one-node two-process bare send with a stack capture of both ranks after 15 s. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; SC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/sendcost; LCI=/u/bhosale/lci/build-cuda/lib
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
until grep -q '^[0-9:]* ch_none_8x4x4' $L/logs/chan7_chain.out; do sleep 5; [ "$(left_s)" -gt 400 ] || break; done
P=$(pgrep -f '^bash /u/bhosale/.*n2_chan7.sh' | head -1); [ -n "$P" ] && kill $P && echo "chan7 runner stopped at $(date +%T) (allocation kept)"
while pgrep -f "run_lmd.sh $JID" >/dev/null; do sleep 3; done
N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID
ENVL="LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so"
echo "=== after_chan7 $(date +%T), left $(left_s) s"
sc() {  # <layout> <timeout> <args...>
  local lay=$1 tmo=$2; shift 2; local N NP PPN PPNARG
  case $lay in ipc) N=1 NP=2 PPN=2 PPNARG="+ppn 1";; rdma) N=2 NP=2 PPN=1 PPNARG="+ppn 1";; esac
  [ "$(left_s)" -gt $((tmo+20)) ] || { echo "skip sendcost $lay $*: $(left_s) s left"; return 1; }
  local LOG=$SC/logs/sendcost_${lay}_$(date +%H%M%S).log
  env $ENVL timeout $tmo srun --jobid=$JID --overlap --mpi=cray_shasta -N $N -n $NP --ntasks-per-node=$PPN -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$SC stdbuf -oL -eL ./sendcost $PPNARG "$@" > $LOG 2>&1 &
  local RP=$!
  if [ "$lay" = ipc ]; then
    sleep 15
    if ! grep -q 'entry    burst' $LOG; then
      echo "  ipc bare send: no result after 15 s; stacks:"
      srun --jobid=$JID --overlap -N 1 -n 1 -w $N0 --cpu-bind=none bash -c 'for p in $(pgrep -x sendcost); do echo "  == pid $p"; timeout 25 gdb -p $p -batch -ex "set pagination off" -ex "thread apply all bt" 2>/dev/null | grep -E "^Thread|^#[0-9]+ " | grep -vE "libc\.so|libpthread|clock_gettime|sched_yield|__GI_" | head -30 | cut -c1-200 | sed "s/^/    /"; done' 2>&1
    fi
  fi
  wait $RP
  echo "$(date +%T) sendcost $lay $*: rc=$? | $(grep -E '^(converse|entry) ' $LOG | tr '\n' '|' | cut -c1-330)"
}
sc rdma 60 +lci_ndevices 1
sc rdma 60 -b 32 -p 200
sc rdma 60 -b 32 -p 200 +lci_ndevices 1
sc ipc 60
sc ipc 60 +lci_ndevices 1
echo "=== after_chan7 done $(date +%T), left $(left_s) s; node KEPT"
