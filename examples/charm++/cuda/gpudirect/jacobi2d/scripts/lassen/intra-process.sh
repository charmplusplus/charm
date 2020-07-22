#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -J jacobi2d-sync

cd $HOME/charm/examples/charm++/cuda/gpudirect/jacobi2d

n_iters=100
grid_size=32768
sync="-y"

echo "# Intra-process Jacobi2D"

echo "#Regular"
for block_size in 16384 8192 4096 2048 1024
do
  echo "# Block size $block_size"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    jsrun -n1 -a1 -c4 -g4 ./jacobi2d -s $grid_size -b $block_size -i $n_iters $sync +ppn 4 +pemap L0,4,80,84 +gpumap block
  done
done

echo "# Zerocopy"
for block_size in 16384 8192 4096 2048 1024
do
  echo "# Block size $block_size"
  for iter in 1 2 3
  do
    echo "# Iteration $iter"
    jsrun -n1 -a1 -c4 -g4 ./jacobi2d -s $grid_size -b $block_size -i $n_iters $sync -z +ppn 4 +pemap L0,4,80,84 +gpumap block
  done
done
