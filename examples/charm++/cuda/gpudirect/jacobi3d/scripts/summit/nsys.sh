#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi3d-nsys

# These need to be changed between submissions
file=jacobi3d-e
n_nodes=1
n_procs=$((n_nodes * 6))
grid_width=3072
grid_height=2048
grid_depth=512

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi3d

ppn=1
pemap="L0,4,8,84,88,92"
n_iters=100
warmup_iters=10
sync=""

echo "# Jacobi3D Performance Benchmarking"

for overdecomp in 1 2 3 4 5
do
  if [ $overdecomp -eq 1 ]
  then
    block_width=1024
    block_height=1024
    block_depth=512
  elif [ $overdecomp -eq 2 ]
  then
    block_width=1024
    block_height=512
    block_depth=512
  elif [ $overdecomp -eq 3 ]
  then
    block_width=512
    block_height=512
    block_depth=512
  elif [ $overdecomp -eq 4 ]
  then
    block_width=512
    block_height=512
    block_depth=256
  else
    block_width=512
    block_height=256
    block_depth=256
  fi

  echo "# Block size $block_width x $block_height x $block_depth"
  exe jsrun -n$n_procs -a1 -c$ppn -g1 -K3 -r6 nsys profile -f true -o jacobi3d-n$n_nodes-x$block_width-y$block_height-z$block_depth-p%q{OMPI_COMM_WORLD_RANK} ./$file -X $grid_width -Y $grid_height -Z $grid_depth -x $block_width -y $block_height -z $block_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap
done
