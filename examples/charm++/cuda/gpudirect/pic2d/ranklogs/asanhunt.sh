#!/bin/bash
# asanhunt.sh <jobid> <attempts>
# Loop pic2d with LB under AddressSanitizer until the heap corruption or the
# wedge fires. The three things that make ASan work here at all:
#   - LD_PRELOAD the REAL libasan (the toolset's libasan.so is a linker script)
#   - protect_shadow_gap=0, or cuInit(0) fails: ASan's shadow gap collides with
#     CUDA's address-space reservations
#   - detect_leaks=0, so a clean exit is not reported as a failure
P=/u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/pic2d; cd $P || exit 1
J=$1; N=${2:-6}; ulimit -c 0
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/asan-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
export FI_MR_CACHE_MONITOR=disabled PMI_MAX_KVS_ENTRIES=1024
export ASAN_OPTIONS=detect_leaks=0:protect_shadow_gap=0:print_stacktrace=1:halt_on_error=1:log_path=$P/asanlog
D="+balancer DiffusionLB +LBDiffusionCommOn +LBDiffusionGpuDim"
for r in $(seq 1 $N); do
  rm -f asanlog.* 
  env X=1 LD_PRELOAD=/lib64/libasan.so.8 timeout 420 srun --jobid=$J --mpi=cray_shasta -N 1 -n 4 \
      --ntasks-per-node=4 --cpus-per-task=8 --cpu-bind=none --exact stdbuf -oL -eL \
      $P/pic2d -W 1024 -H 1024 -w 128 -h 128 -p 4 -i 30 -u 3 -d 1 -c 5 -f 5 -b 5 $D \
      +gpushm +gpupool +gpuipceventpool 256 +ppn 8 > ah_$r.log 2>&1
  rc=$?
  asan=$(grep -lE "ERROR: AddressSanitizer" ah_$r.log asanlog.* 2>/dev/null | head -1)
  echo "run $r rc=$rc last=$(grep -a 'Iter ' ah_$r.log | tail -1 | cut -c1-40) asan=${asan:-none}"
  if [ -n "$asan" ]; then
    echo "=== ASAN REPORT in $asan ==="; grep -aA 25 "ERROR: AddressSanitizer" $asan | head -35; exit 2
  fi
  [ "$rc" != 0 ] && { echo "=== non-zero exit, tail ==="; tail -6 ah_$r.log; }
done
echo "ASANHUNT DONE (no report in $N runs)"
