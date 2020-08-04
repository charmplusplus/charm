#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

ppn=1
n_iters=100
warmup_iters=10
grid_size=32768
sync=""
pemap="L0,4,84,88"
pool_size=2048
buffer_size=1024

echo "# Multi-process Jacobi2D"

for block_size in 16384 8192 4096 2048 1024
do
  echo "# Block size $block_size"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    jsrun -n4 -a1 -c$ppn -g1 -K2 -r4 ./jacobi2d -s $grid_size -b $block_size -i $n_iters -w $warmup_iters $sync +ppn $ppn +pemap $pemap +gpumap block +gpuipceventpool $pool_size +gpucommbuffer $buffer_size
  done
done
