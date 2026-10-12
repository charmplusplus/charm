#!/bin/bash
# n2_diff3.sh <jobid> <floor>: after n2_diff2.sh, DiffusionLB alone with a LOWER imbalance floor (+LBDiffusionMinImbalance
# <floor>): period 25 lag 10, then period 10 lag 4. The default floor left Diffusion at "max ratio 0" (no moves) at every
# event while the step drifted 38.8 -> 48.0 ms (D10) and 35.6 -> 47.4 (Metis-then-Diffusion, 10-08). Keeps the node.
JID=$1; FLOOR=${2:-0.02}; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="+LBDiffusionCommOn +LBDiffusionGpuDim +LBDiffusionMinImbalance $FLOOR"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <period> <lag>
  local pfx=$1 P=$2 LAG=$3
  [ "$(left_s)" -gt 150 ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env $ENVC TMO=300 LB_BAL=DiffusionLB DIFF_FLAGS="$DF" LBF=$P LBB=$P LBLAG=$LAG XFLAGS="" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 async 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  local L=$(ls -t $D/logs/${pfx}_256_2000_p${P}_async_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation" $L | head -2 | cut -c1-200
  echo "  physics @2000: $(grep -E '^  step +2000:' $L | sed -E 's/.*(rho [0-9.]+ +KE [0-9.e+]+ +\|v\|max [0-9.]+).*/\1/'); series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%8==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}'); max ratio: $(grep -oE 'max ratio [0-9.]+' $L | awk 'NR%8==1{printf "%s ", $3}'); moves: $(grep -ciE 'migrat(e|ing) [1-9]' $L)"
}
until grep -q "diff2 done" $D/logs/diff2_chain.out 2>/dev/null; do sleep 5; done
echo "=== diff3 start $(date +%T), left $(squeue -j $JID -h -o %L), floor $FLOOR, runners $(ps -u bhosale -o cmd | grep -E 'run3d\.sh' | grep -v grep | wc -l)"
run_arm Df25 25 10
run_arm Df10 10 4
echo "=== diff3 done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
