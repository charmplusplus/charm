#!/bin/bash
# after_chan9.sh <jobname lmd_chan9>: after chan9's planned arms (through ch_none_8x4x4), stop the chain's loop and
# run the transfer-cost calibration on uffd with the rebuilt lbcalib (the 09-15 binary aborted on stale objects).
# Waits for the job to be RUNNING before reading its end time. Never sallocs/scancels.
L=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/leanmd; LBC=/u/bhosale/charm-reconverse/benchmarks/charm++/cuda/gpudirect/lbcalib
JID=""; until [ -n "$JID" ]; do sleep 20; JID=$(squeue -u bhosale -n lmd_chan9 -h -t RUNNING -o %i 2>/dev/null | head -1); done
END_EPOCH=$(date -d "$(squeue -j $JID -h -o %e | head -1)" +%s 2>/dev/null); left_s() { echo $(( END_EPOCH - $(date +%s) )); }
until grep -q '^[0-9:]* ch_none_8x4x4' $L/logs/chan9_chain.out; do sleep 5; [ "$(left_s)" -gt 300 ] || break; done
P=$(pgrep -f '^bash /u/bhosale/.*n2_chan9.sh' | head -1); [ -n "$P" ] && kill $P && echo "chan9 runner stopped at $(date +%T) (allocation kept)"
while pgrep -f "run_lmd.sh $JID" >/dev/null; do sleep 3; done
echo "=== after_chan9 $(date +%T), job $JID, left $(left_s) s: lbcalib on uffd (binary $(ls -la --time-style=+%H:%M $LBC/lbcalib | awk '{print $6}'))"
cd $LBC && timeout $(( $(left_s) - 30 )) bash $LBC/calib2n.sh $JID 2>&1 | tail -24 | cut -c1-200
ls -t $LBC/*.conf | head -2
echo "=== after_chan9 done $(date +%T), left $(left_s) s; node KEPT"
