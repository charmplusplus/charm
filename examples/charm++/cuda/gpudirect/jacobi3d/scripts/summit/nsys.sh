#!/bin/bash
#BSUB -W 10
#BSUB -P csc357
#BSUB -nnodes 1
#BSUB -J jacobi3d-nsys-n1

# These need to be changed between submissions
file=jacobi3d-e
n_nodes=1
n_procs=$((n_nodes * 6))
grid_width=1536
grid_height=1536
grid_depth=1536

# Function to display commands
exe() { echo "\$ $@" ; "$@" ; }

cd $HOME/work/charm/examples/charm++/cuda/gpudirect/jacobi3d

ppn=1
pemap="L0,4,8,84,88,92"
n_iters=100
warmup_iters=0
sync=""

echo "# Jacobi3D Performance Benchmarking"

for overdecomp in 1 2 4 8 16
do
  num_chares=$((n_procs * overdecomp))

  echo -e "# ODF-$overdecomp\n"
  exe jsrun -n$n_procs -a1 -c$ppn -g1 -K3 -r6 nsys profile -t cuda -s none -b none -f true -o jacobi3d-n$n_nodes-odf$overdecomp-p%q{OMPI_COMM_WORLD_RANK} ./$file -c $num_chares -x $grid_width -y $grid_height -z $grid_depth -w $warmup_iters -i $n_iters +ppn $ppn +pemap $pemap
done
