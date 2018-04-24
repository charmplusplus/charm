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
void run_VECADD_KERNEL(workRequest* wr, cudaStream_t kernel_stream,
                       void** devBuffers) {
  vecAdd<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>(
      (float*)devBuffers[wr->getBufferID(C_INDEX)],
      (float*)devBuffers[wr->getBufferID(A_INDEX)],
      (float*)devBuffers[wr->getBufferID(B_INDEX)], *((int*)wr->getUserData()));
}
#endif

#ifdef USE_WR
void cudaVecAdd(int vectorSize, float* h_A, float* h_B, float* h_C,
                cudaStream_t stream, void* cb) {
#else
void cudaVecAdd(int vectorSize, float* h_A, float* h_B, float* h_C, float* d_A,
                float* d_B, float* d_C, cudaStream_t stream, void* cb) {
#endif
  int size = vectorSize * sizeof(float);
  dim3 dimBlock(BLOCK_SIZE, 1);
  dim3 dimGrid((vectorSize - 1) / dimBlock.x + 1, 1);

#ifdef USE_WR
  // DEPRECATED
  workRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(dimGrid, dimBlock);
  wr->setStream(stream);
  wr->addBuffer(h_A, size, true, false, true);
  wr->addBuffer(h_B, size, true, false, true);
  wr->addBuffer(h_C, size, false, true, true);
  wr->setCallback(cb);
  wr->setTraceName("vecadd");
  wr->setRunKernel(run_VECADD_KERNEL);
  wr->copyUserData(&vectorSize, sizeof(int));

  hapiEnqueue(wr);
#else
  hapiCheck(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  hapiCheck(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  vecAdd<<<dimGrid, dimBlock, 0, stream>>>(d_C, d_A, d_B, vectorSize);
  hapiCheck(cudaPeekAtLastError());

  hapiCheck(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));

  hapiAddCallback(stream, cb);
#endif
}
