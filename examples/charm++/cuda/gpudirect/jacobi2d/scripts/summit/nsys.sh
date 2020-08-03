#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d-nsys

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

ppn=1
n_iters=10
warmup_iters=0
grid_size=32768
sync=""
#pemap="L0-28:4,84-112:4"
pemap="L0,4,84,88"

echo "# Jacobi2D"

for block_size in 8192
do
  echo "# Block size $block_size"
  jsrun -n4 -a1 -c$ppn -g1 -K2 -r4 nsys profile -f true -o jacobi2d-g$grid_size-b$block_size-ppn$ppn-p%q{OMPI_COMM_WORLD_RANK} ./jacobi2d -s $grid_size -b $block_size -i $n_iters -w $warmup_iters $sync +ppn $ppn +pemap $pemap +gpumap block
done
