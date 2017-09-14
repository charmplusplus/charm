This example performs matrix multiplication with a chare array. There is no
decomposition involved, as each chare computes an independent set of matrices.
It can be set to use the CuBLAS library on the GPU.

Usage: ./matmul -c [chares] -s [matrix size] -b: use CuBLAS
