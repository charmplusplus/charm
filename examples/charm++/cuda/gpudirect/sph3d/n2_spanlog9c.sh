#!/bin/bash
# n2_spanlog9c.sh: one timed (1,4,4) async arm on the tail of job 22856533 with the registration-outside-the-lock
# build (03:4x): read the Metis window (LB at step 25), the registration times, the close breakdown.
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d; cd $D; JID=22856533
source <(sed -n '/^export NODES=2/,/^}$/p; /^run_arm() {/,/^}$/p' n2_spanlog9.sh)
echo "=== spanlog9c start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm async s14 1 4 CfR 1000 CHARM_LB_CUPTI_TIME=1
echo "    metis window: $(grep -m1 'LB at step 25: resumed' $LASTLOG | cut -c1-80)"
echo "    registrations: $(grep -c 'registered once for RDMA' $LASTLOG) total; ms: $(grep -o 'RDMA ([0-9.]* ms' $LASTLOG | sed -E 's/RDMA \(//; s/ ms//' | sort -n | awk '{a[NR]=$1; s+=$1} END{printf "n=%d min=%.1f med=%.1f max=%.1f sum=%.0f", NR, a[1], a[int((NR+1)/2)], a[NR], s}')"
echo "    window 25-50 registrations: $(awk '/^  step +25:/{p=1} /^  step +50:/{p=0} p' $LASTLOG | grep -c 'registered once for RDMA')"
echo "    closes 2-9: $(grep -h '^\[LBCUPTI pe=0\] flush' $LASTLOG | sed -n 2,9p | sed -E 's/.*flush=/flush=/' | tr '\n' ';')"
grep -h '^\[LBCUPTI pe=[0-9]*\] flush' $LASTLOG | sed -E 's/.*process=([0-9.]+)s.*wait=([0-9.]+)s retire=([0-9.]+)s pass1=([0-9.]+)s pass2=([0-9.]+)s merge=([0-9.]+)s.*/\1 \2 \3 \4 \5 \6/' | awk '{n++; p+=$1; w+=$2; r+=$3; a+=$4; b+=$5; m+=$6; if($2>wm)wm=$2; if($4>am)am=$4} END{if(n) printf "    close means (ms): process=%.1f wait=%.1f retire=%.1f pass1=%.1f pass2=%.1f merge=%.1f | max wait=%.0f pass1=%.0f\n", 1000*p/n, 1000*w/n, 1000*r/n, 1000*a/n, 1000*b/n, 1000*m/n, 1000*wm, 1000*am}'
echo "=== spanlog9c done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
