#!/bin/bash
# n2_flush.sh <jobid> (2026-10-11): the CUPTI window-close fix (default non-forced flush, 256 KB activity buffers,
# correlations and parked kernels carried over one drain) on the async configuration: 256-patch async@25 arms, 2000
# steps each, until the hold runs out. Under the old forced flush 2 of the first 4 replicates of this hold segfaulted
# inside CUPTI's launch intercept at a window close (f1 step 1600, c64 step 1425); each arm here is a crash chance for
# the fixed build and a step-time / attribution check (CHARM_LB_CUPTI_TIME prints the flush and parse times). Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1 CHARM_LB_CUPTI_TIME=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <sync|async> <extra app args> <env...>
  local pfx=$1 g="$2" ns=$3 mode=$4 extra=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local P=25 LAG=10; [ $mode = sync ] && { P=100; LAG=50; }
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300
  grep -A24 "SPH FAULT" $L | head -26 | cut -c1-200
  echo "  send stalls over 5 ms: $(grep -c SENDSTALL $L); events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); flush/process/total ms (mean over windows, PE 0): $(grep '^\[LBCUPTI pe=0\]' $L | sed -E 's/.*flush=([0-9.]+)s process=([0-9.]+)s.*total=([0-9.]+)s/\1 \2 \3/' | awk '{f+=$1; p+=$2; t+=$3; n++} END{if(n) printf "%.1f %.1f %.1f over %d", 1000*f/n, 1000*p/n, 1000*t/n, n}')"
  grep -q "rc=0" $L
}
echo "=== flush chain start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 20 ]; do i=$((i+1)); run_arm x$i "8 4 8" 2000 async "" $ENVC; done
echo "=== flush chain done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
