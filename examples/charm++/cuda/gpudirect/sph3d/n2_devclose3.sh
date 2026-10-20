#!/bin/bash
# n2_devclose3.sh (2026-10-12): after n2_devclose2.sh, on the first node of the same allocation, the driver-call cost
# microbenchmark (tests/util/launchcost) in the shapes of the process/GPU split, CUPTI off and on:
#   A. one process, 4 devices: "4D threads x D devices (4 each)" = the (1,4,4) layout; "4 threads x 1 device" and
#      "4D threads x 1 device" alongside;
#   B. four processes at once, one device each (CUDA_VISIBLE_DEVICES), 4 threads each = the (4,1,4) control.
# If a driver call in A costs more than in B, the CUDA runtime (or CUPTI) serialises a process's calls across
# its devices, and that is the one-process layout's steady-state penalty under tracing. Keeps the node.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d; LC=/u/bhosale/charm-reconverse/tests/util/launchcost/launchcost
until grep -q "^=== devclose2 done" $D/logs/devclose2_chain.out 2>/dev/null; do sleep 10; done
JID=$(grep -o 'jobid=[0-9]*' $D/logs/devclose_chain.out | head -1 | cut -d= -f2)
[ -n "$JID" ] || { echo "no jobid"; exit 1; }
NODE0=$(scontrol show hostnames "$(squeue -j $JID -h -o %N)" | head -1)
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
SR="srun --jobid=$JID --overlap -N 1 -n 1 --ntasks-per-node=1 -w $NODE0 --gpus-per-node=4 --cpus-per-task=64 --cpu-bind=none --exact"
echo "=== devclose3 start $(date +%T), job $JID node $NODE0, left $(squeue -j $JID -h -o %L)"
[ "$(left_s)" -gt 240 ] || { echo "skip: $(left_s) s left"; exit 0; }
echo "--- A. one process, four devices (threads pinned from core 0)"
$SR env LAUNCHCOST_BASE_CORE=0 $LC 0 2000 20 4 2>&1 | cut -c1-160
echo "--- B. four processes at once, one device each, 4 threads each (cores 16d..)"
$SR bash -c 'for d in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$d LAUNCHCOST_BASE_CORE=$((16*d)) '$LC' 0 2000 20 1 > /tmp/lc_$d.out 2>&1 & done; wait; for d in 0 1 2 3; do echo "# process $d"; grep -E "^(cupti|off|on) +(op|launch|memcpy|flag|event) +(shape|4 threads x 1 stream)" /tmp/lc_$d.out; done; rm -f /tmp/lc_[0-3].out' 2>&1 | cut -c1-160
echo "--- A again (order check)"
$SR env LAUNCHCOST_BASE_CORE=0 $LC 0 2000 20 4 2>&1 | grep -E "4D threads x D devices|4 threads x 1 device" | cut -c1-160
echo "=== devclose3 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
