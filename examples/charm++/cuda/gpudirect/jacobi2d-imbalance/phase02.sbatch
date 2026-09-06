#!/bin/bash
#SBATCH --job-name=j2d-asynclb
#SBATCH --account=mzu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:40:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -x
cd /u/bhosale/charm-reconverse/examples/charm++/cuda/gpudirect/jacobi2d-imbalance

# libreconverse.so is not rpath'd into the binary; liblci.so needs liblct.so.
export LD_LIBRARY_PATH="/u/bhosale/charm-reconverse/multicore-linux-x86_64-cuda/lib:/u/bhosale/lci_install/lib64:$LD_LIBRARY_PATH"
# Stop libinfinipath's SIGSEGV handler from swallowing genuine crash reports.
export IPATH_NO_BACKTRACE=1
# Deliberately NOT set: CHARM_METALB_GPU_FAKE_DEVICES. On real hardware the
# device count must come from hapiMyDevice().

RUN="srun --mpi=cray_shasta -n 4 --cpus-per-task=8 --cpu-bind=cores --exact"
PEFLAGS="+pe 32 +setcpuaffinity"
GPUFLAGS="+gpushm +gpucommbuffer 256 +gpulbbuffer 256 +gpuipceventpool 256"
GEOM="-W 8192 -H 8192 -w 1024 -h 1024 -i 200 -u 10 -m 8"

echo "############ STEP 0: device identity gate ############"
echo "# PASS if: 'HAPI> Config' shows 4 device(s) per process, AND the"
echo "#          '[MetaLB] sample= gpu-dev n=' line reads n=4, not n=1."
echo "# If n=1, stop -- metalbMyDeviceId()/hapiMyDevice() is collapsing the"
echo "# per-device identity and no other result below means anything."
$RUN ./jacobi2d -W 8192 -H 8192 -w 1024 -h 1024 -i 40 -u 5 -m 8 -M -s 5 \
  $PEFLAGS $GPUFLAGS +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
echo "=== STEP 0 exit: $? ==="

echo "############ STEP 1: no-LB reference hash (x3) ############"
echo "# All three must print the SAME 'Final checksum:'. That value is the"
echo "# oracle for every run below. Do not reuse the laptop hashes."
for rep in 1 2 3; do
  $RUN ./jacobi2d $GEOM -b 100000 $PEFLAGS $GPUFLAGS +balancer MetisLB
  echo "=== ref rep $rep exit: $? ==="
done

echo "############ STEP 2: GPU-triggered, synchronous ############"
$RUN ./jacobi2d $GEOM -M -s 5 $PEFLAGS $GPUFLAGS \
  +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
echo "=== STEP 2 exit: $? ==="

echo "############ STEP 3: asynchronous, overlap lags 0 / 3 / 9 ############"
for lag in 0 3 9; do
  echo "---- lag $lag ----"
  $RUN ./jacobi2d $GEOM -A -s 5 -l $lag $PEFLAGS $GPUFLAGS \
    +LBAsync +balancer MetisLB +MetaLB +MetaLBGpuTrigger +LBDebug 1
  echo "=== lag $lag exit: $? ==="
done

echo "############ STEP 4: Metis first, Diffusion thereafter ############"
$RUN ./jacobi2d $GEOM -A -s 5 -l 3 $PEFLAGS $GPUFLAGS +LBAsync \
  +balancer MetisLB +balancer DiffusionLB +LBDiffusionGpuDim \
  +MetaLB +MetaLBGpuTrigger +LBDebug 1
echo "=== STEP 4 exit: $? ==="

echo "############ ALL DONE ############"
echo "--- checksums (all must match the STEP 1 reference) ---"
grep -h "Final checksum" ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out 2>/dev/null

# Run hygiene: these runs leave core dumps and shm segments behind.
rm -f core.* 2>/dev/null
find /dev/shm -maxdepth 1 -user "$USER" -delete 2>/dev/null
