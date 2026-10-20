#!/bin/bash
# after_chan8.sh <jobid>: after chan8's planned arms (through the channel benchmark), stop the chain's loop and
# re-run the transfer-cost calibration on the uffd monitor (lbcalib/calib2n.sh): the built-in cost table was
# measured on the disabled monitor, where every message carried ~9 us that is now gone. Never sallocs/scancels.
JID=$1; L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; LBC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/lbcalib
until [ "$(squeue -j $JID -h -o %T 2>/dev/null)" = RUNNING ]; do sleep 15; done
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
until grep -q '^--- channel ipc' $L/logs/chan8_chain.out && grep -A3 '^--- channel ipc' $L/logs/chan8_chain.out | grep -q 'rc='; do sleep 5; [ "$(left_s)" -gt 420 ] || break; done
P=$(pgrep -f '^bash /u/bhosale/.*chan8b.sh' | head -1); [ -n "$P" ] && kill $P && echo "chan8 runner stopped at $(date +%T) (allocation kept)"
while pgrep -f "run_lmd.sh $JID\|run_channel.sh $JID" >/dev/null; do sleep 3; done
echo "=== after_chan8 $(date +%T), left $(left_s) s: bare send between two processes on one node (hung twice on the disabled monitor), then lbcalib on uffd"
SC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/sendcost; LCI=/u/bhosale/lci/build-cuda/lib; N0=$(scontrol show hostnames $(squeue -j $JID -h -o %N) | head -1)
SLOG=$SC/logs/sendcost_ipc_uffd_$(date +%H%M%S).log
env LD_LIBRARY_PATH=$LCI:/u/bhosale/charm-reconverse/lib:/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64 LD_PRELOAD=$LCI/liblci.so FI_MR_CACHE_MONITOR=uffd PMI_MAX_KVS_ENTRIES=8192 CUDA_DEVICE_ORDER=PCI_BUS_ID timeout 40 srun --jobid=$JID --overlap --mpi=cray_shasta -N 1 -n 2 --ntasks-per-node=2 -w $N0 --gpus-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact --kill-on-bad-exit=1 --chdir=$SC stdbuf -oL -eL ./sendcost +ppn 1 > $SLOG 2>&1
echo "$(date +%T) sendcost ipc (uffd): rc=$? | $(grep -E '^(converse|entry) ' $SLOG | tr '\n' '|' | cut -c1-300)"
cd $LBC && timeout $(( $(left_s) - 30 )) bash $LBC/calib2n.sh $JID 2>&1 | tail -30 | cut -c1-200
ls -t $LBC/*.conf | head -3
echo "=== after_chan8 done $(date +%T), left $(left_s) s; node KEPT"
