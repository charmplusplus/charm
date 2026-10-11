#!/bin/bash
# n2_interior.sh <jobid> (2026-10-10 night): the interior-first force pass (SPH_INTERIOR_FIRST=1: the survivors' interior
# cells compute at the end of the step while the exchange is in flight; the next step runs the boundary layers only)
# against the control, same nodes, helper on, plus whatever flush setting $D/.flush_env holds; async then sync; 2000
# steps each; then kernels-only traces of both async arms. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
FL=$(cat $D/.flush_env 2>/dev/null)
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=${PER:-25} LAG=${LAGV:-10}; [ $mode = sync ] && { P=100; LAG=50; }
  env "$@" $FL TMO=400 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="$XF $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300; grep -c SENDSTALL $L | sed 's/^/  send stalls over 5 ms: /'; grep -iE "MIGSTATS|migration (path|time)" $L | head -4 | cut -c1-200; grep -q "rc=0" $L; }
n=0; until grep -q "build done" $D/logs/build_interior.out 2>/dev/null || [ $n -ge 120 ]; do sleep 5; n=$((n+1)); done
echo "=== interior chain start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}'), flush env: ${FL:-default}"
run_arm cA 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1; okc=$?
run_arm iA 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 SPH_INTERIOR_FIRST=1 CHARM_LB_MIGSTATS=1; oki=$?
run_arm iS 2000 sync "" SPH_INTERIOR_FIRST=1 SPH_STUCK_SECONDS=30; okiS=$?
# the LB window every 25 steps carries the forced-flush drain (~135 ms with a deep queue): at period 100 it is 4x rarer
PER=100 LAGV=50 run_arm cA100 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5; okc100=$?
PER=100 LAGV=50 run_arm iA100 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_INTERIOR_FIRST=1; oki100=$?
GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
SPH_INTERIOR_FIRST=1 GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
python3 $D/trace_compare.py $(for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_* | head -2); do echo "$(basename $T)=$T/trace_last1s.json"; done) 2>&1 | grep -E "=====|busy|by the kernel|idle %" | cut -c1-200
# the forced flush replaced by a soft flush over 256 KB buffers (the open buffer then holds <1% of a window's records)
run_arm cB 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 CHARM_LB_CUPTI_SOFT_FLUSH=1 CHARM_LB_CUPTI_BUFKB=256; okcB=$?
run_arm iB 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 CHARM_LB_CUPTI_SOFT_FLUSH=1 CHARM_LB_CUPTI_BUFKB=256 SPH_INTERIOR_FIRST=1; okiB=$?
echo "=== interior chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: control=$okc interior=$oki interior-sync=$okiS control@100=$okc100 interior@100=$oki100 softflush256=$okcB interior-softflush256=$okiB; node KEPT"
