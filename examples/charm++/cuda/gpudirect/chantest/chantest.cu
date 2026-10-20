#include "hapi.h"

#define BLOCK_SIZE 256

__global__ void fillKernel(double* data, int count, double val) {
  int ti = blockDim.x * blockIdx.x + threadIdx.x;

  if (ti < count) {
    data[ti] = val;
  }
}

void invokeFillKernel(double* data, int count, double val, cudaStream_t stream) {
  dim3 block_dim(BLOCK_SIZE);
  dim3 grid_dim((count + block_dim.x - 1) / block_dim.x);

  fillKernel<<<grid_dim, block_dim, 0, stream>>>(data, count, val);

  hapiCheck(cudaPeekAtLastError());
}
