#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

ppn=4
n_iters=100
warmup_iters=10
grid_width=32768
grid_height=32768
zerocopy=""
sync=""
pemap="L0,84,88,92"

echo "# Single-process Jacobi2D"

for div in 2 4 8 16 32
do
  block_width=$((grid_width / div))
  block_height=$((grid_height / div))
  echo "# Block size $block_width x $block_height"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    exe jsrun -n1 -a1 -c$ppn -g4 ./jacobi2d -W $grid_width -H $grid_height -w $block_width -h $block_height -i $n_iters -w $warmup_iters $zerocopy $sync +ppn $ppn +pemap $pemap +gpumap block
  done
done
