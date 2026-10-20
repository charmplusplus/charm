#!/bin/bash
# after2_chan9.sh: chantest and the channel benchmark again on chan9's node with binaries rebuilt from clean
# (the chain's copies had stale objects after the channel class grew). Waits for after_chan9 (lbcalib) to finish.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; G=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
JID=""; until [ -n "$JID" ]; do sleep 20; JID=$(squeue -u bhosale -n lmd_chan9 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
while pgrep -f "^bash /u/bhosale/.*after_chan9.sh" >/dev/null || pgrep -f "^bash /u/bhosale/.*chan9b.sh" >/dev/null; do sleep 5; [ "$(left_s)" -gt 150 ] || break; done
while pgrep -f "run_lmd.sh $JID\|lbcalib" >/dev/null; do sleep 3; done
echo "=== after2_chan9 $(date +%T), left $(left_s) s: chantest + benchmark, clean builds ($(ls -la --time-style=+%H:%M $G/chantest/chantest | awk '{print $6}'), $(ls -la --time-style=+%H:%M $C/channel | awk '{print $6}'))"
[ "$(left_s)" -gt 120 ] && bash $G/chantest/test_chantest.sh $JID 24 3
for a in "rdma" "rdma -g" "ipc"; do [ "$(left_s)" -gt 60 ] || break; echo "--- channel $a"; bash $C/run_channel.sh $JID $a vmm -i 100 -l 20 -r 200 -R 20 2>&1 | grep -E 'channel .*rc=|one-way' | cut -c1-230 | sed 's/^/  /'; done
echo "=== after2_chan9 done $(date +%T), left $(left_s) s; node KEPT"
