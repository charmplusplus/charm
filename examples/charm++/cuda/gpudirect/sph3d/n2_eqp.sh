#!/bin/bash
# n2_eqp.sh <jobid> (2026-10-11): the EQUAL-PERIOD sync vs async comparison (the user: "never compare async 25 with sync 100").
# Both arms period 25 lag 10, MetisLB alone, floor 3%, cut 10%, HostByMsgs, CHARM_LB_HELPER=near, 256 patches, 32 GB vmm pool.
# 1) kernels-only +gputrace of 200 steps, syncH then asyncH -> busy %, gaps, last-1s and last-4s windows for Perfetto;
# 2) if >= 6 min remain, one untraced 2000-step pair (sync@25, async@25) for step time on this pair. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
until [ "$(squeue -j $JID -h -o %T)" = RUNNING ]; do sleep 10; done
echo "=== eqp start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), runners $(ps -u bhosale -o cmd | grep -E 'run3d\.sh' | grep -v grep | wc -l)"
for ARM in syncH asyncH; do
  [ "$(left_s)" -gt 150 ] || { echo "skip trace $ARM: $(left_s) s left"; continue; }
  PER=25 LAGV=10 GT_NOCOPY=1 TMO=120 bash $D/n2_gputrace.sh $JID 200 $ARM 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  T=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_${ARM}_256_* | head -1)
  python3 $D/gputrace_merge.py --out $T/trace_last4s.json --last 4.0 $T 2>&1 | tail -1 | cut -c1-160
  L=$(ls -t $D/logs/gk200_${ARM}_256_*.log | head -1)
  echo "  $ARM log $L: events $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); $(grep -c 'SPH FAULT' $L) faults"
done
S=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_syncH_256_* | head -1); A=$(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_256_* | head -1)
echo "--- EQUAL PERIOD 25, last 1 s: sync=$S async=$A"
python3 $D/trace_compare.py sync25=$S/trace_last1s.json async25=$A/trace_last1s.json 2>&1 | grep -vE "^\s*$" | cut -c1-200 | head -40
echo "--- EQUAL PERIOD 25, last 4 s:"
python3 $D/trace_compare.py sync25=$S/trace_last4s.json async25=$A/trace_last4s.json 2>&1 | grep -E "busy|idle %|gap time by size|LoadBalance" | cut -c1-200 | head -20
for m in sync async; do
  [ "$(left_s)" -gt 360 ] || { echo "skip 2000-step $m: $(left_s) s left"; continue; }
  env $ENVC TMO=300 DIFF_FLAGS= LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=eqp_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 ${m}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/eqp_256_2000_p25_${m}T_*.log | head -1)
  echo "  $m@25 whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); $(grep -c 'SPH FAULT' $L) faults; series: $(grep -E '^  step +[0-9]+:' $L | awk 'NR%10==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}')"
done
echo "=== eqp done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
