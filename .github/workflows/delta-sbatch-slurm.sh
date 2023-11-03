#!/bin/bash -l
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -t 0:30:00
#SBATCH -J autobuild
#SBATCH -p cpu
#SBATCH -A mzu-delta-cpu
#cd $indir
set -x
module load libfabric; module load cmake
./build all-test $target --with-production --enable-error-checking -j -g
make -C $target/tmp test TESTOPTS+="+setcpuaffinity"
#cd $testdir
#$make clean
#$make -C ../tests $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
#$make -C ../examples $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
#$make -C ../benchmarks $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
# Save make exit status
#status=\$?
#echo \$? > $result
