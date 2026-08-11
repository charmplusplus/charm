make clean
make verify
# srun --mpi=cray_shasta -n 2 ./verify +gpushm +gpuipceventpool 256 +allgpus +gpucommbuffer 256 +ppn 2
# ./charmrun ++local ++p 2 ./verify +gpushm +gpuipceventpool 512 +allgpus +gpucommbuffer 128
# srun -n 2 ./verify +ppn 2
# srun -n 4 ./verify
