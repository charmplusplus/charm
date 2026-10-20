#!/bin/bash
JID=$1; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "=== after4_chan7 $(date +%T), left $(left_s) s: channel benchmark with the uffd monitor"
for lay in rdma ipc; do
  [ "$(left_s)" -gt 45 ] || { echo "skip $lay"; break; }
  echo "--- channel $lay (uffd)"
  bash $C/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20 2>&1 | grep -E 'channel .*rc=|one-way|credit' | cut -c1-230 | sed 's/^/  /'
done
echo "=== after4_chan7 done $(date +%T), left $(left_s) s; node KEPT"
