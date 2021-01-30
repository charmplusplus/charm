#!/bin/bash
#BSUB -W 30
#BSUB -P csc357
#BSUB -nnodes 256
#BSUB -J jacobi3d-openmpi-strong-n256

# These need to be changed between submissions
file=jacobi3d_mpi-bench
n_nodes=256
n_procs=$((n_nodes * 6))
grid_width=3072
grid_height=3072
grid_depth=3072

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm-inter/examples/ampi/cuda/gpudirect/jacobi3d

module unload spectrum-mpi
export PATH=$HOME/work/openmpi-4.1.0/install/bin:$HOME/work/ucx/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/work/openmpi-4.1.0/install/lib:$HOME/work/ucx/install/lib:/sw/summit/gdrcopy/2.0/lib64:$LD_LIBRARY_PATH

echo 'LSB_MCPU_HOSTS:'
echo $LSB_MCPU_HOSTS

# Create rankfile for OpenMPI
python3 scripts/create_rankfile.py

n_iters=100
warmup_iters=10

echo "# OpenMPI Jacobi3D Performance Benchmarking (GPUDirect off)"

for iter in 1 2 3
do
  date
  echo "# Run $iter"
  exe mpirun -rf rankfile-$LSB_JOBID -x PATH -x LD_LIBRARY_PATH ./$file -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters
done

echo "# OpenMPI Jacobi3D Performance Benchmarking (GPUDirect on)"

for iter in 1 2 3
do
  date
  echo "# Run $iter"
  exe mpirun -rf rankfile-$LSB_JOBID -x PATH -x LD_LIBRARY_PATH ./$file -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters -d
done

rm rankfile-$LSB_JOBID
