#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

ppn=1
n_iters=100
warmup_iters=10
grid_size=32768
sync=""
pemap="L0,4,84,88"

echo "# Multi-process Jacobi2D"

for div in 2 4 8 16 32
do
  block_size=$((grid_size / div))
  echo "# Block size $block_size"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    exe jsrun -n4 -a1 -c$ppn -g1 -K2 -r4 ./jacobi2d -s $grid_size -b $block_size -i $n_iters -w $warmup_iters $sync +ppn $ppn +pemap $pemap +gpumap block +gpunoshm
  done
done
