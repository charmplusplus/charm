#!/bin/bash
# n2_leanmd_pool.sh <jobid>: LeanMD weak N=2 with a pool the 30% landing reserve leaves room in (vs 1515.7/909.4/904.4 of 2026-09-17).
JID=$1
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect && POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib LMD_POOL_WEAK=${LMD_POOL_WEAK:-8192} STEPS=100 PERIOD=20 LAG=16 bash scale.sh $JID 2 leanmd weak "nolb sync async" 2>&1 | grep -E "leanmd|rc=|FAIL|Fatal|abort|Cannot" | cut -c1-200
for a in nolb sync async; do L=leanmd/ranklogs/weak_N2_$a/rank_0.log; echo "  $a: $(grep -a -m1 -o 'arenas of [0-9]* MB' $L) | $(grep -a -m1 -o 'node 0 memory: pool [0-9]* MB, free [0-9]* MB, reserve [0-9]* MB, advertises [0-9]* MB' $L) | migrations: $(grep -a -o 'Number of cross node migrations AFTER LB: [0-9]*' $L | sed 's/.*: //' | tr '\n' ' ') | Metis: $(grep -a -o '[0-9]* of [0-9]* migratable objects move' $L | head -1)"; done
echo "leanmd-pool done $(date +%T) node time $(squeue -j $JID -h -o %M)"
