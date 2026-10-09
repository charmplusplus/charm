#!/bin/bash
# n2_nsys.sh <jobid> <steps> <arm: none|sync|async> [sph3d flags...]: one 8-GPU run of the final configuration
# (256 patches, sort/100, Metis alone, floor 3%, cut 10%, 32 GB pool; async at period 25 window 10, sync at 100)
# under nsys with +nvtx, then a per-rank sqlite export inside the allocation. Reports: /scratch/mzu/bhosale/nsys/<tag>/.
# Merge afterwards: python3 nsys_merge.py --out trace.json [--window T0:T1] /scratch/mzu/bhosale/nsys/<tag>/rank_*.sqlite
JID=$1; NS=$2; ARM=$3; shift 3
D=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/sph3d
export NODES=2 NP=8 POOL_ALLOC=vmm LCI_LIB=/u/bhosale/lci/build-cuda/lib SPH_SORT_EVERY=100 POOL=32768
TAG=nsys${NS}_${ARM}_$(date +%H%M%S); OUT=/scratch/mzu/bhosale/nsys/$TAG; mkdir -p $OUT
export SPH_WRAP=$D/numa_wrap_sph_nsys.sh SPH_NSYS_DIR=$OUT
case $ARM in async) P=25; LAG=10;; *) P=100; LAG=50;; esac
DIFF_FLAGS= LBF=$P LBB=$P LBLAG=$LAG LB=$ARM LBDEBUG=1 TMO=${TMO:-900} $D/run3d.sh $JID $TAG -X 1.5 -Y 0.5 -Z 2.5 -w 1.0 -t 2 -s 0.0022 -V 10 -x 8 -y 4 -z 8 -r 2 -e 0.05 -i $NS -S 25 -k 1 \
  +LBMetisHostByMsgs +LBMetisMinGain 0.03 +LBMetisMaxCutRise 0.1 +nvtx "$@" > /dev/null 2>&1
L=$D/logs/$TAG.log
echo "$(date +%T) $TAG: $(grep -o 'rc=[0-9]*' $L | tail -1) checks=$(grep -c 'check ok' $L) whole-run $(grep 'Average iteration' $L | sed -E 's/.*: ([0-9.]+) ms.*/\1/') ms; reports: $(ls $OUT/*.nsys-rep 2>/dev/null | wc -l)"
grep -i "NVTX ranges on\|fatal\|abort\|segmentation" $L | head -3
srun --jobid=$JID --overlap -N 2 -n 8 --ntasks-per-node=4 bash -c "cd $OUT && nsys export -t sqlite -f true -o rank_\$SLURM_PROCID.sqlite rank_\$SLURM_PROCID.nsys-rep" > $OUT/export.log 2>&1
echo "$(date +%T) exports: $(ls $OUT/*.sqlite 2>/dev/null | wc -l) sqlite files, $(du -sh $OUT | cut -f1) in $OUT"
