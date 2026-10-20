#!/bin/bash
# after2_chan6.sh <jobid>: the three channel-benchmark arms that after_chan6.sh lost to a knob-name collision
# (the chain exported BIN=leanmd.ch, which run_channel.sh's new binary knob picked up; the knob is CHBIN now).
# Waits for chan6b's last arm, then: ipc (array version), rdma (credit cost split), rdma with one LCI device.
JID=$1; C=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/channel
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); [ -n "$END_EPOCH" ] || { echo "no end time"; exit 1; }
left_s() { echo $(( END_EPOCH - $(date +%s) )); }
while pgrep -f "run_lmd.sh $JID" >/dev/null || pgrep -f "^bash /u/bhosale/.*chan6b.sh" >/dev/null; do sleep 3; done
echo "=== after2_chan6 $(date +%T), left $(left_s) s"
ch_arm() { local lay=$1; shift; [ "$(left_s)" -gt 60 ] || { echo "skip channel $lay: $(left_s) s left"; return 1; }; echo "--- channel $lay $*"; bash $C/run_channel.sh $JID $lay vmm -i 100 -l 20 -r 200 -R 20 "$@" 2>&1 | grep -E 'channel .*rc=|one-way|zc-host|Elapsed' | cut -c1-230 | sed 's/^/  /'; }
ch_arm ipc
ch_arm rdma
ch_arm rdma +lci_ndevices 1
echo "=== after2_chan6 done $(date +%T), left $(left_s) s; node KEPT"
