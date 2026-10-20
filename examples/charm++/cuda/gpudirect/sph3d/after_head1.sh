#!/bin/bash
# after_head1.sh <jobid>: the leftover minutes of the head1 allocation (node KEPT by the chain), in priority
# order, each arm gated on the time left: (1) the step-1 persistent endpoint tests in all three transports,
# (2) the channel-vs-zero-copy microbenchmark in the three layouts (step 2), (3) the step-0 zero-copy host-cost
# arm on sph3d. Never sallocs, never scancels; the node is left to expire.
JID=$1
[ -n "$JID" ] || { echo "usage: after_head1.sh <jobid>"; exit 1; }
G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null)
[ -n "$END_EPOCH" ] || { echo "cannot read the job end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
echo "=== after_head1 start $(date +%T), job $JID, left $(left_s) s"
if [ "$(left_s)" -gt 150 ]; then bash $G/persistent/test_persist.sh $JID; else echo "skip persistent tests: $(left_s) s left"; fi
if [ "$(left_s)" -gt 150 ]; then bash $G/chantest/test_chantest.sh $JID 24 3; else echo "skip chantest: $(left_s) s left"; fi
for lay in memcpy ipc rdma; do
  if [ "$(left_s)" -gt 120 ]; then bash /u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20; else echo "skip channel $lay: $(left_s) s left"; fi
done
if [ "$(left_s)" -gt 200 ]; then bash $G/sph3d/n2_zcstats.sh $JID 1000; else echo "skip zcstats: $(left_s) s left"; fi
echo "=== after_head1 done $(date +%T), left $(left_s) s"
