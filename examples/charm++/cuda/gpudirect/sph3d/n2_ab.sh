#!/bin/bash
# n2_ab.sh <jobid> (2026-10-11): A/B of the CUPTI flush gate on one pair: 256-patch async controls alternating gate ON
# (default) and OFF (CHARM_LB_CUPTI_NOGATE=1, the pre-gate behaviour, which can fault) until the hold runs out.
# Question: does the gate cost step time? (one pre-gate control on gpub[067,074] ran 39.99; two gated 41.21 / 41.14.)
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
ENVC="CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1 CHARM_LB_CUPTI_TIME=1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
run_arm() {  # <pfx> <"gx gy gz"> <steps> <sync|async> <extra app args> <env...>
  local pfx=$1 g="$2" ns=$3 mode=$4 extra=$5; shift 5
  set -- $g "$@"; local np=$(($1*$2*$3)); shift 3
  local P=25 LAG=10   # same period and lag for sync and async (2026-10-11 rule: never async@25 vs sync@100)
  local need=$(( ns > 500 ? 150 : 70 )); [ "$(left_s)" -gt $need ] || { echo "skip $pfx: $(left_s) s left"; return 1; }
  env "$@" TMO=300 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "$g" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_${np}_${ns}_p${P}_${mode}T_*.log | head -1)
  grep -E "SPH FAULT|CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300
  echo "  events: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}'); send stalls: $(grep -c SENDSTALL $L); flush ms (mean, PE 0): $(grep '^\[LBCUPTI pe=0\]' $L | sed -E 's/.*flush=([0-9.]+)s.*/\1/' | awk '{f+=$1; n++} END{if(n) printf "%.1f over %d", 1000*f/n, n}')"
  grep -q "rc=0" $L
}
echo "=== A/B chain start $(date +%T), left $(squeue -j $JID -h -o %L), nodes $(squeue -j $JID -h -o %N), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
i=0; while [ "$(left_s)" -gt 150 ] && [ $i -lt 20 ]; do i=$((i+1)); if [ $((i % 2)) = 1 ]; then run_arm on$i "8 4 8" 2000 async "" $ENVC; else run_arm off$i "8 4 8" 2000 async "" $ENVC CHARM_LB_CUPTI_NOGATE=1; fi; done
echo "--- A/B summary (whole-run ms/step) ---"
for k in on off; do echo "$k: $(for L in $(ls -t $D/logs/${k}[0-9]*_256_2000_p25_asyncT_*.log 2>/dev/null); do grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/'; done | tr '\n' ' ')"; done
echo "=== A/B chain done $(date +%T), left $(squeue -j $JID -h -o %L); node KEPT"
