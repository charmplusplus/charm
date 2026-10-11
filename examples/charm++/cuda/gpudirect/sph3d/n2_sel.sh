#!/bin/bash
# n2_sel.sh <jobid> (2026-10-11): interior-first with SELECTED targets (each pass compacts its targets in cell order
# with CUB DeviceSelect::Flagged and runs one thread per target; the v4 passes exited most warps at once) against the
# control, same nodes, helper on, async@25 then sync, 2000 steps; then kernels-only traces of both async arms and an
# ncu profile of the two selected kernels in a short noLB run. Keeps the node.
JID=$1; D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
run_arm() { local pfx=$1 ns=$2 mode=$3 extra=$4; shift 4; local P=${PER:-25} LAG=${LAGV:-10}; [ $mode = sync ] && { P=100; LAG=50; }
  env "$@" TMO=400 DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG XFLAGS="${XFO:-$XF} $extra" TAGPFX=${pfx}_ POOL=32768 bash $D/n2_final2.sh $JID "8 4 8" $ns ${mode}T 2>&1 | grep -v "^HAPI\[" | cut -c1-200
  L=$(ls -t $D/logs/${pfx}_256_${ns}_p${P}_${mode}T_*.log | head -1); grep -E "CORRUPTION|Reason:|Fatal|Segmentation|expected|STUCK\] no" $L | head -3 | cut -c1-300; grep -c SENDSTALL $L | sed "s/^/  send stalls over 5 ms: /"; grep -iE "MIGSTATS|migration (path|time)" $L | head -3 | cut -c1-200; grep -q "rc=0" $L; }
echo "=== sel chain start $(date +%T), left $(squeue -j $JID -h -o %L), binary $(ls -la --time-style=+%H:%M $D/sph3d | awk '{print $6}')"
run_arm sC 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1; okc=$?
run_arm sI 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 SPH_INTERIOR_FIRST=1 CHARM_LB_MIGSTATS=1; oki=$?
run_arm sR 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 CHARM_LB_MIGSTATS=1; okr=$?
run_arm sIS 2000 sync "" SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 SPH_STUCK_SECONDS=30; okiS=$?
# the balancer repartitions from scratch whenever it acts and moves 150-240 of the 256 patches for priced gains of
# 3-6% that never show in the step time (control log of job 22801886): act only on a 10% priced gain
XFO="+LBMetisMinGain 0.10 +LBMetisMaxCutRise 0.1" run_arm sC10 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 CHARM_LB_MIGSTATS=1; okc10=$?
XFO="+LBMetisMinGain 0.10 +LBMetisMaxCutRise 0.1" run_arm sR10 2000 async "" CHARM_LB_HELPER=near SPH_SEND_WARN_MS=5 SPH_STUCK_SECONDS=30 SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 CHARM_LB_MIGSTATS=1; okr10=$?
for pfx in sC sI sR sC10 sR10; do L=$(ls -t $D/logs/${pfx}_256_2000_p25_asyncT_*.log 2>/dev/null | head -1); [ -n "$L" ] && echo "$pfx moves per event: $(grep -h '#Objects migrating' $L | awk '{print $7}' | tr -d ',' | awk '{n++; if($1>0){a++; m+=$1}} END{print a" of "n" acted, "m" moves"}')"; done
GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
SPH_INTERIOR_FIRST=1 GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 GT_NOCOPY=1 TMO=150 bash $D/n2_gputrace.sh $JID 200 asyncH 2>&1 | grep -v "^HAPI\[" | cut -c1-200
python3 $D/trace_compare.py $(for T in $(ls -td /scratch/mzu/bhosale/gputrace/gk200_asyncH_* | head -3 | tac); do echo "$(basename $T)=$T/trace_last1s.json"; done) 2>&1 | grep -E "=====|busy|by the kernel|idle %" | cut -c1-200
# ncu on the selected kernels (noLB, 40 steps, the flag/select/forces launches 8..12 of every process)
NCU_DIR=/scratch/mzu/bhosale/ncu/sel1; mkdir -p $NCU_DIR
NODES=2 NP=8 POOL=32768 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib TMO=240 SPH_WRAP=$D/numa_wrap_sph_ncu.sh NCU_DIR=$NCU_DIR NCU_SKIP=8 NCU_COUNT=6 NCU_REGEX="forces|rowTable" SPH_INTERIOR_FIRST=1 SPH_INTERIOR_ROWS=1 \
  bash $D/run3d.sh $JID ncu_sel1 -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i 40 -S 20 -k 1 2>&1 | grep -E "^\[run3d\]" | cut -c1-120
ls $NCU_DIR | head -3
echo "=== sel chain done $(date +%T), left $(squeue -j $JID -h -o %L); rc ok: control=$okc selected=$oki selected-sync=$okiS rows=$okr control@gain10=$okc10 rows@gain10=$okr10; node KEPT"
