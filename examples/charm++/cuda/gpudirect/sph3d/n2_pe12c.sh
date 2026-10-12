#!/bin/bash
# n2_pe12c.sh <jobid>: last minutes of the hold: a second 12-PE async draw (pe12c) and a 4-PE arm with the same smaller
# interprocess event pool (pe4ev128) to separate the event-pool variable from the PE count. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
DF="+balancer DiffusionLB +LBDiffusionCommOn +LBLoadVectorAbove 0.15 +LBDiffusionMinImbalance 0.02"
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"; E="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
for spec in "12 128 pe12c" "4 128 pe4ev128"; do
  set -- $spec
  [ "$(left_s)" -gt 130 ] || { echo "skip $3: $(left_s) s left"; continue; }
  echo "=== $3 start $(date +%T), $(left_s) s left"
  env $E PES=$1 IPCEV=$2 TMO=200 DIFF_FLAGS="$DF" LBF=25 LBB=25 LBLAG=10 XFLAGS="$XF" TAGPFX=eqp3${3}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" 2000 asyncT 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  G=$(ls -t $D/logs/eqp3${3}_256_2000_p25_asyncT_*.log | head -1)
  echo "  async@25 $3: whole-run $(grep 'Average iteration' $G | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms/step; $(grep -c 'SPH FAULT\|Fatal' $G) faults; PEs: $(grep -m1 -oE '[0-9]+ PEs? per' $G); series: $(grep -E '^  step +[0-9]+:' $G | awk 'NR%10==0{match($0,/, ([0-9.]+) ms\/step/,m); printf "%s:%s ", $2, m[1]}')"
done
echo "=== extra draws done $(date +%T)"
