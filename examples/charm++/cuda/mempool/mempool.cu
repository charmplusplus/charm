#include "hapi.h"

#define BLOCK_SIZE 256

__global__ void vecAdd(float* C, float* A, float* B, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    C[id] = A[id] + B[id];
  }
}

void cudaVecAdd(int n_floats, size_t size, float* h_A, float* h_B, float* h_C,
    float* d_A, float* d_B, float* d_C, cudaStream_t stream) {
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((n_floats - 1) / BLOCK_SIZE + 1);

  hapiCheck(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  vecAdd<<<dimGrid, dimBlock, 0, stream>>>(d_C, d_A, d_B, n_floats);
  hapiCheck(cudaPeekAtLastError());

  hapiCheck(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
}
