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
grid_width=32768
grid_height=32768
zerocopy=""
#zerocopy_options="+gpushm +gpucommbuffer 1024 +gpuipceventpool 2048"
zerocopy_options=""
sync=""
pemap="L0,4,84,88"

echo "# Multi-process Jacobi2D"

for div in 2 4 8 16 32
do
  block_width=$((grid_width / div))
  block_height=$((grid_height / div))
  echo "# Block size $block_width x $block_height"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    exe jsrun -n4 -a1 -c$ppn -g1 -K2 -r4 ./jacobi2d -W $grid_width -H $grid_height -w $block_width -h $block_height -i $n_iters -w $warmup_iters $zerocopy $sync +ppn $ppn +pemap $pemap +gpumap block $zerocopy_options
  done
done
