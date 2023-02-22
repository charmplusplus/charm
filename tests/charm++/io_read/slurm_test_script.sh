#!/bin/bash
#SBATCH --job-name testing
#SBATCH -p skx-dev
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --output=test_results_execution
#SBATCH -t 00:15:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH  --exclusive
module load gcc/9.1.0
./charmrun +p4 ./iotest # readtest.txt 128 4 1
