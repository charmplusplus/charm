#!/bin/bash
# pictimings.sh <jobid> <reps>
# noLB / sync / async on the imbalanced pic2d configuration, pool flags always.
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; cd $P || exit 1
J=$1; N=${2:-3}; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
POOL="+gpushm +gpupool +gpuipceventpool 256"
B="-W 1024 -H 1024 -w ${PW:-128} -h ${PW:-128} -p 4 -i 60 -u 10 -d 1 -c 20"
for r in $(seq 1 $N); do for tag in noLB sync async; do
  case $tag in
    noLB)  A="-f 99999";;
    sync)  A="-f 10 -b 10 $D";;
    async) A="-f 10 -b 10 -a -l 2 +LBAsync $D";;
  esac
  env X=1 timeout 150 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 --ntasks-per-node=4 \
      --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL \
      $P/pic2d $B $A $POOL +ppn 8 > pt${PW:-128}_${tag}_$r.log 2>&1
  rc=$?
  echo "$tag rep$r rc=$rc avg=$(grep -aoP 'Average iteration time: \K[0-9.]+' pt${PW:-128}_${tag}_$r.log) imb=$(grep -aoP 'imbalance \(max/avg\): \K[0-9.]+' pt${PW:-128}_${tag}_$r.log | tail -1)"
done; done
echo "PICTIMINGS DONE"
