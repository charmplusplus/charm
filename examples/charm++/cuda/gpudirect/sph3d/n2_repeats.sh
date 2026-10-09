#!/bin/bash
# n2_repeats.sh <jobid> "<async env, e.g. CHARM_LB_MIGRATE_WINDOW=1>" [lag]  -- alternate asyncT/syncT in the final configuration
# (256 patches, sort/100, Metis alone, floor 3%, cut 10%, period 100) until the allocation has < 3.5 min left; prints each arm.
JID=$1; AENV=$2; LAG=${3:-10}; [ "$LAG" -ge "${APERIOD:-25}" ] && LAG=10   # the window must be shorter than the period   # async at its best period (APERIOD, default 25) and window (LAG, default 10); sync at its best, period 100
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768
# POOL 32 GB: at 16 GB the pool is 75% full at 98M and an async first re-partition deadlocked 123 s (landings waiting on departures).
XF="+LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1"
left_s() { squeue -j $JID -h -o %L | awk -F: '{if(NF==3) print $1*3600+$2*60+$3; else if(NF==2) print $1*60+$2; else print $1}'; }
i=0
while [ "$(left_s)" -gt 210 ]; do
  i=$((i+1))
  env $AENV DIFF_FLAGS= LBF=${APERIOD:-25} LBB=${APERIOD:-25} LBLAG=$LAG XFLAGS="$XF" TAGPFX=rep${i} $D/n2_final2.sh $JID "8 4 8" 2000 asyncT
  [ "$(left_s)" -gt 210 ] || break
  DIFF_FLAGS= LBF=100 LBB=100 LBLAG=50 XFLAGS="$XF" TAGPFX=rep${i} $D/n2_final2.sh $JID "8 4 8" 2000 syncT
done
echo "repeats done $(date +%T), left $(squeue -j $JID -h -o %L)"
