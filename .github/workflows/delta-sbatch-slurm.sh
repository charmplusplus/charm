#!/bin/bash -l
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -t 02:00:00
#SBATCH -J autobuild
#SBATCH -o $output
#SBATCH -e $output
#SBATCH -p regular
#SBATCH --partition=cpu-interactive
#SBATCH --pty bash
#SBATCH -A mzu-delta-cpu
#cd $indir
module load libfabric; module load cmake
./build all-test $target -j8 $flags
#cd $testdir
#$make clean
#$make -C ../tests $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
#$make -C ../examples $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
#$make -C ../benchmarks $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
# Save make exit status
#status=\$?
#echo \$? > $result
