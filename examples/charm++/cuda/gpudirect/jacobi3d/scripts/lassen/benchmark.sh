#!/bin/bash
#BSUB -G asccasc
#BSUB -W 10
#BSUB -core_isolation 2
#BSUB -q pbatch
#BSUB -nnodes 3
#BSUB -J jacobi3d-e-n3
#BSUB -o jacobi3d-e-n3.%J

# These need to be changed between submissions
file=jacobi3d-e
n_nodes=3
n_procs=$((n_nodes * 4))
grid_width=3072
grid_height=1536
grid_depth=1536

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
  num_chares=$((n_procs * overdecomp))

  echo -e "# ODF-$overdecomp\n"
  for iter in 1 2 3
  do
    date
    echo -e "# Run $iter\n"
    exe jsrun -n$n_procs -a1 -c$ppn -g1 -K2 -r4 ./$file -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap
  done
done
