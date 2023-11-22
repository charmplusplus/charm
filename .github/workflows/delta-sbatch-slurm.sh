#!/bin/bash -l
#SBATCH -N 2
#SBATCH -n 64
#SBATCH -o %j.output
#SBATCH -e %j.output
#SBATCH -t 1:00:00
#SBATCH -J autobuild
#SBATCH -p cpu
#SBATCH -A mzu-delta-cpu

set -x

module load libfabric; module load cmake

mkdir -p $target

# ./build all-test $target --with-production --enable-error-checking -j64 -g

cd $target
# make -C tests test OPTS="$flags" TESTOPTS="$testopts" $maketestopts
# make -C examples test OPTS="$flags" TESTOPTS="$testopts" $maketestopts
# make -C benchmarks test OPTS="$flags" TESTOPTS="$testopts" $maketestopts

# For debugging:
echo "Just 4 debugging"
true

# Save make exit status
status=$?
echo $status > ../$SLURM_JOBID.result
