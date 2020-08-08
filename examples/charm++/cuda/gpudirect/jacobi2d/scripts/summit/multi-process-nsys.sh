#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi2d-nsys

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi2d

ppn=1
n_iters=10
warmup_iters=0
grid_size=32768
zerocopy=""
#zerocopy_options="+gpucommbuffer 1024 +gpuipceventpool 2048"
zerocopy_options="+gpunoshm"
sync=""
pemap="L0,4,84,88"

echo "# Multi-process Jacobi2D NVIDIA Nsight Systems"

for div in 2 4 8 16 32
do
  block_size=$((grid_size / div))
  echo "# Block size $block_size"
  exe jsrun -n4 -a1 -c$ppn -g1 -K2 -r4 nsys profile -f true -o jacobi2d-g$grid_size-b$block_size-ppn$ppn-p%q{OMPI_COMM_WORLD_RANK} ./jacobi2d -s $grid_size -b $block_size -i $n_iters -w $warmup_iters $zerocopy $sync +ppn $ppn +pemap $pemap +gpumap block $zerocopy_options
done
