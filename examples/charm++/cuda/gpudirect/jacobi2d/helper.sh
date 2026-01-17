make  clean
make
srun -n 2 ./jacobi2d -y -z +ppn 8 +gpushm

