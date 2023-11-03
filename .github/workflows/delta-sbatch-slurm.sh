#!/bin/bash -l
#SBATCH -N 2
#SBATCH -n 32
#SBATCH -t 02:00:00
#SBATCH -J autobuild
#SBATCH -o $output
#SBATCH -e $output
#SBATCH -p regular
#SBATCH --partition=cpu
#SBATCH -A mzu-delta-cpu
cd $indir
cd $testdir
module load libfabric; module load cmake
#$make clean
$make -C ../tests $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
$make -C ../examples $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
$make -C ../benchmarks $target OPTS="$flags" TESTOPTS="$testopts" $maketestopts
# Save make exit status
status=\$?
echo \$? > $result
