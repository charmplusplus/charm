#!/bin/bash
#BSUB -W 30
#BSUB -P csc357
#BSUB -nnodes 16
#BSUB -J jacobi3d-openmpi-weak-n16

# These need to be changed between submissions
file=jacobi3d_mpi-bench
n_nodes=16
n_procs=$((n_nodes * 6))
grid_width=6144
grid_height=3072
grid_depth=3072

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm-inter/examples/ampi/cuda/gpudirect/jacobi3d

module unload spectrum-mpi
export PATH=$HOME/work/openmpi-4.1.0/install/bin:$HOME/work/ucx-1.9.0/install/bin:$PATH
export LD_LIBRARY_PATH=$HOME/work/openmpi-4.1.0/install/lib:$HOME/work/ucx-1.9.0/install:/sw/summit/gdrcopy/2.0/lib64:$LD_LIBRARY_PATH

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
