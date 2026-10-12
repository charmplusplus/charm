#!/bin/bash
# n2_eqp3.sh (2026-10-11 15:45): on the next hold (job name sph3d_eqp2): sync@25 vs async@25 traces with the configuration
# that makes sense (Metis ONCE then Diffusion: +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02,
# sort every 100, CHARM_LB_HELPER=near, 256 patches), then the user's PE-count sweep: +ppn 4 (today's), 8, 16 per process
# (numa_wrap_sph.sh pins PES cores of the process's NUMA domain; 16 = every core of the domain, the helper thread shares).
# Order: traces PES=4 (sync, async) | 2000-step async PES=4, 8, 12 (never 16: that oversubscribes the domain with the helper thread) | trace async PES=8 | 2000-step sync PES=4, 8. Keeps the node.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"; ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
DF="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
until JID=$(squeue -u bhosale -h -n sph3d_eqp2 -t RUNNING -o %i | head -1) && [ -n "$JID" ]; do sleep 5; done
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
echo "=== eqp3 start $(date +%T), job $JID, left $(left_s) s, nodes $(squeue -j $JID -h -o %N), runners $(ps -u bhosale -o cmd | grep -E 'run3d\.sh' | grep -v grep | wc -l)"
trace() {  # <arm> <pes>
  [ "$(left_s)" -gt 150 ] || { echo "skip trace $1 pe$2"; return; }
  PES=$2 PER=25 LAGV=10 GT_NOCOPY=1 TMO=120 DIFF_FLAGS="$DF" bash $D/n2_gputrace.sh $JID 200 $1 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_$1_256_pe$2_* | head -1); python3 $D/gputrace_merge.py --out $T/trace_last4s.json --last 4.0 $T 2>&1 | tail -1 | cut -c1-120
  L=$D/logs/$(basename $T).log; echo "  $1 pe$2: $(grep -m1 -oE '\+balancer [A-Za-z]+ \+balancer [A-Za-z]+' $L); Metis events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); Diffusion cross-node: $(grep -oE 'cross node migrations AFTER LB: [0-9]+' $L | awk '{s+=$NF} END{print s+0}'); $(grep -c 'SPH FAULT' $L) faults; busy: $(grep -oE 'busy [0-9.]+%' $L | head -1)"
}
arm() {  # <sync|async> <pes>
  [ "$(left_s)" -gt 150 ] || { echo "skip 2000-step $1 pe$2"; return; }
  env $ENVC PES=$2 TMO=200 DIFF_FLAGS="$DF" LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=eqp3pe$2_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 $1T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/eqp3pe$2_256_2000_p25_$1T_*.log | head -1); echo "  $1@25 pe$2 sorted, Metis once + Diffusion: whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; $(grep -c 'SPH FAULT' $L) faults; ppn $(grep -m1 -oE '\+ppn [0-9]+' $L); series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%10==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}')"
}
trace syncH 4; trace asyncH 4
S=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_syncH_256_pe4_* 2>/dev/null | head -1); A=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_pe4_* 2>/dev/null | head -1)
[ -n "$S" ] && [ -n "$A" ] && { echo "--- Metis once + Diffusion, sorted, period 25, pe4: last 1 s: sync=$S async=$A"; python3 $D/trace_compare.py sync25=$S/trace_last1s.json async25=$A/trace_last1s.json 2>&1 | grep -vE "^\s*$" | cut -c1-200 | head -30; echo "--- last 4 s:"; python3 $D/trace_compare.py sync25=$S/trace_last4s.json async25=$A/trace_last4s.json 2>&1 | grep -E "busy|gap time by size" | cut -c1-200; }
arm async 4; arm async 8; arm async 12
trace asyncH 8
A8=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_pe8_* 2>/dev/null | head -1); [ -n "$A8" ] && [ -n "$A" ] && { echo "--- async pe4 vs pe8, last 4 s:"; python3 $D/trace_compare.py pe4=$A/trace_last4s.json pe8=$A8/trace_last4s.json 2>&1 | grep -E "busy|gap time by size|by the kernel" | cut -c1-200; }
arm sync 4; arm sync 8
echo "=== eqp3 done $(date +%T), left $(left_s) s; node KEPT"
