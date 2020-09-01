#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 192
#BSUB -J jacobi3d-e-n192
#BSUB -o jacobi3d-e-n192.%J

# These need to be changed between submissions
file=jacobi3d-e
n_nodes=192
n_procs=$((n_nodes * 4))
grid_width=12288
grid_height=8192
grid_depth=4096

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/charm/examples/charm++/cuda/gpudirect/jacobi3d

ppn=1
pemap="L0,4,80,84"
n_iters=100
warmup_iters=10
sync=""

echo "# Jacobi3D Performance Benchmarking"

for overdecomp in 1 2 4 8 16
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
  elif [ $overdecomp -eq 4 ]
  then
    block_width=512
    block_height=512
    block_depth=512
  elif [ $overdecomp -eq 8 ]
  then
    block_width=512
    block_height=512
    block_depth=256
  else
    block_width=512
    block_height=256
    block_depth=256
  fi

  echo -e "# ODF-$overdecomp (Block size $block_width x $block_heigh x $block_depth)\n"
  for iter in 1 2 3
  do
    date
    echo -e "# Run $iter\n"
    exe jsrun -n$n_procs -a1 -c$ppn -g1 -K2 -r4 ./$file -X $grid_width -Y $grid_height -Z $grid_depth -x $block_width -y $block_height -z $block_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap
  done
done
