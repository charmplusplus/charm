make  clean
make
srun -n 2 ./jacobi2d -y -z +ppn 2 +gpushm +gpuipceventpool 512 +allgpus +gpucommbuffer 128

