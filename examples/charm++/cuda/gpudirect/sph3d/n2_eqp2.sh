#!/bin/bash
# n2_eqp2.sh (2026-10-11 15:10): the traces that matter: sync@25 vs async@25 with the configuration that makes sense --
# Metis ONCE then Diffusion (+LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02), sort every 100,
# 256 patches, CHARM_LB_HELPER=near. Finds its own job (name sph3d_eqp2). 1) kernels-only traces, 200 steps, syncH then asyncH;
# 2) 2000-step sync and async pairs with the same configuration (the sorted sync counterpart was never run). Keeps the node.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"; ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
until JID=$(squeue -u bhosale -h -n sph3d_eqp2 -t RUNNING -o %i | head -1) && [ -n "$JID" ]; do sleep 5; done
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
echo "=== eqp2 start $(date +%T), job $JID, left $(left_s) s, nodes $(squeue -j $JID -h -o %N), runners $(ps -u bhosale -o cmd | grep -E 'run3d\.sh' | grep -v grep | wc -l)"
for ARM in syncH asyncH; do
  [ "$(left_s)" -gt 150 ] || { echo "skip trace $ARM"; continue; }
  PER=25 LAGV=10 GT_NOCOPY=1 TMO=120 DIFF_FLAGS="$DF" bash $D/n2_gputrace.sh $JID 200 $ARM 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_${ARM}_256_* | head -1); python3 $D/gputrace_merge.py --out $T/trace_last4s.json --last 4.0 $T 2>&1 | tail -1 | cut -c1-120
  L=$(ls -t $D/logs/gk200_${ARM}_256_*.log | head -1); echo "  $ARM: $(grep -m1 -oE '\+balancer [A-Za-z]+ \+balancer [A-Za-z]+' $L); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" Metis acted, "m" moves"}'); Diffusion cross-node moves: $(grep -oE 'cross node migrations AFTER LB: [0-9]+' $L | awk '{s+=$NF} END{print s+0}'); $(grep -c 'SPH FAULT' $L) faults"
done
S=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_syncH_256_* | head -1); A=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* | head -1)
echo "--- Metis once + Diffusion, sorted, period 25: last 1 s: sync=$S async=$A"; python3 $D/trace_compare.py sync25=$S/trace_last1s.json async25=$A/trace_last1s.json 2>&1 | grep -vE "^\s*$" | cut -c1-200 | head -30
echo "--- last 4 s:"; python3 $D/trace_compare.py sync25=$S/trace_last4s.json async25=$A/trace_last4s.json 2>&1 | grep -E "busy|gap time by size" | cut -c1-200
for m in sync async; do
  [ "$(left_s)" -gt 160 ] || { echo "skip 2000-step $m"; continue; }
  env $ENVC TMO=200 DIFF_FLAGS="$DF" LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=eqp2_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 ${m}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/eqp2_256_2000_p25_${m}T_*.log | head -1); echo "  $m@25 sorted, Metis once + Diffusion: whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; $(grep -c 'SPH FAULT' $L) faults; series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%10==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}')"
done
echo "=== eqp2 done $(date +%T), left $(left_s) s; node KEPT"
