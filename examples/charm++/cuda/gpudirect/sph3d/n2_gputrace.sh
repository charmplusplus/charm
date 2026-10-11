#!/bin/bash
# n2_gputrace.sh <jobid> <steps> <arm: none|sync|async[H|O|S]>... (H: CHARM_LB_HELPER=near = both; O: CHARM_LB_CUPTI_OFFTHREAD=near, S: CHARM_LB_STRATEGY_THREAD=near;
# env GT_NOCOPY=1 adds +gputracenocopy = kernels only, tag prefix gk; XTRA_APP="-c" adds app args): 8-GPU runs of the final configuration (256 patches,
# sort/100, Metis alone, floor 3%, cut 10%, 32 GB pool; async at period 25 window 10, sync at 100) with the
# runtime's own GPU timeline (+gputrace: CUPTI kernels + entry methods + idle, LB unaffected), then one merged
# Perfetto trace of the last second per arm. Files: /scratch/mzu/bhosale/gputrace/<tag>/.
JID=$1; NS=$2; shift 2
set -- ${GDIMS:-8 4 8} "$@"; GX=$1; GY=$2; GZ=$3; shift 3   # env GDIMS="4 4 8" for other patch grids (default 256 patches)
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768
for ARM in "$@"; do
  PFX=gt; [ -n "$GT_NOCOPY" ] && PFX=gk   # gk = kernels-only trace
  TAG=${PFX}${NS}_${ARM}_$((GX*GY*GZ))_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/gputrace/$TAG; mkdir -p $OUT
  NOCOPY=${GT_NOCOPY:+ +gputracenocopy}
  BASE=$(echo $ARM | sed 's/[OSH]*$//'); ENVX=""   # arm suffix: O = CUPTI load build off-thread, S = strategy off-thread
  [[ $ARM == *O* ]] && ENVX="$ENVX CHARM_LB_CUPTI_OFFTHREAD=near"
  [[ $ARM == *S* ]] && ENVX="$ENVX CHARM_LB_STRATEGY_THREAD=near"
  [[ $ARM == *H* ]] && ENVX="$ENVX CHARM_LB_HELPER=near"
  case $BASE in async) P=25; LAG=10;; *) P=100; LAG=50;; esac
  env $ENVX DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG LB=$BASE LBDEBUG=1 TMO=${TMO:-600} $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x $GX -y $GY -z $GZ -r 2 -e 0.05 -i $NS -S 25 -k 1 \
    $XTRA_APP +LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1 +gputrace $OUT$NOCOPY > /dev/null 2>&1
  L=$D/logs/$TAG.log
  echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms | files: $(ls $OUT | grep -c '\.proc$') proc, $(ls $OUT | grep -c '\.pe[0-9]*$') pe; $(du -sh $OUT | cut -f1)"
  grep -i "GPU timeline trace on\|+gputrace\|fatal\|abort\|segmentation" $L | head -3 | cut -c1-160
  for f in $OUT/*.proc; do echo "  $(basename $f): $(grep -c '^K' $f) K, $(grep -c '^M' $f) M, $(grep -c '^T' $f) T"; done | head -2
  ( python3 $D/gputrace_merge.py --out $OUT/trace_last1s.json --last 1.0 $OUT 2>&1 | tail -1 | cut -c1-200
    ls -la $OUT/trace_last1s.json | awk '{print "  trace:", $5/1e6, "MB", $9}'
    python3 $D/gap_report.py $OUT/trace_last1s.json 5 | grep "largest gap per rank" | sed "s/^/  $TAG (gaps>5ms) /" ) &   # merge on the login node while the next arm runs
done
wait
echo "done $(date +%T); left $(squeue -j $JID -h -o %L)"
