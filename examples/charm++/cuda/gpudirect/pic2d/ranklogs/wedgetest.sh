#!/bin/bash
# wedgetest.sh <jobid> <runs> <tag>
# The pic2d LB wedge repro. Before the admission-gate fix this stalled in
# roughly a third to a half of runs, sync or async.
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; cd $P || exit 1
J=$1; N=${2:-8}; T=${3:-fix}; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
stall=0; ok=0
for r in $(seq 1 $N); do
  [ $((r % 2)) -eq 0 ] && A="-a -l 2 +LBAsync" || A=""
  env X=1 timeout 120 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 \
      --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL \
      $P/pic2d -W 1024 -H 1024 -w 128 -h 128 -p 4 -i 30 -u 3 -d 1 -c 10 -f 5 -b 5 $A $D \
      +gpushm +gpupool +gpuipceventpool 256 +ppn 8 > wt_${T}_$r.log 2>&1
  rc=$?
  tot=$(grep -aoP 'Total time: \K[0-9.]+' wt_${T}_$r.log)
  if [ "$rc" = 124 ]; then stall=$((stall+1)); else [ -n "$tot" ] && ok=$((ok+1)); fi
  echo "run $r ($([ -n "$A" ] && echo async || echo sync)) rc=$rc total=${tot:-STALL}"
done
echo "$T: $ok completed, $stall stalled, of $N"
