#include <math.h>
#include <stdio.h>
#include "hapi.h"

#define BLOCK_SIZE 256
#define A_INDEX 0
#define B_INDEX 1
#define C_INDEX 2

__global__ void vecAdd(float* C, float* A, float* B, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    C[id] = A[id] + B[id];
  }
}

#ifdef USE_WR
void run_VECADD_KERNEL(hapiWorkRequest* wr, cudaStream_t kernel_stream,
                       void** devBuffers) {
  vecAdd<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>(
      (float*)devBuffers[wr->getBufferID(C_INDEX)],
      (float*)devBuffers[wr->getBufferID(A_INDEX)],
      (float*)devBuffers[wr->getBufferID(B_INDEX)], *((int*)wr->getUserData()));
}
#endif


void cudaVecAdd(int vectorSize, float* h_A, float* d_A) {
  int size = vectorSize * sizeof(float);
  hapiCheck(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
}
