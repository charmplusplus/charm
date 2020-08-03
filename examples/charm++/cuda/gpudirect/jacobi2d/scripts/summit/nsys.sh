#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d-nsys

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

n_iters=10
warmup_iters=0
grid_size=32768
sync=""

echo "# Jacobi2D"

for block_size in 16384
do
  echo "# Block size $block_size"
  jsrun -n4 -a1 -c1 -g1 -K2 -r4 nsys profile -f true -o jacobi2d-g$grid_size-b$block_size-p%q{OMPI_COMM_WORLD_RANK} ./jacobi2d -s $grid_size -b $block_size -i $n_iters -w $warmup_iters $sync +ppn 1 +pemap L0,4,84,88 +gpumap block
done
