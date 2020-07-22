#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 1
#BSUB -J jacobi2d-sync-nsys

cd $HOME/charm/examples/charm++/cuda/gpudirect/jacobi2d

n_iters=100
grid_size=32768
sync="-y"

echo "# Jacobi2D"

for block_size in 16384
do
  echo "# Block size $block_size"
  jsrun -n4 -a1 -c1 -g1 nsys profile -f true -o jacobi2d-g$grid_size-b$block_size-p%q{OMPI_COMM_WORLD_RANK} ./jacobi2d -s $grid_size -b $block_size -i $n_iters $sync +ppn 1 +pemap L0,4,80,84 +gpumap block
done
