#include <math.h>
#include <stdio.h>
#include "cublas_v2.h"
#include "hapi.h"

#define BLOCK_SIZE 8

#ifdef USE_WR
#define A_INDEX 0
#define B_INDEX 1
#define C_INDEX 2
#endif

extern bool useCublas;

// matrix multiplication code adapted from the CUDA SDK
__global__ void matMul(float* C, float* A, float* B, int wA, int wB) {
  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Index of the first sub-matrix of A processed by the block
  int aBegin = wA * BLOCK_SIZE * by;

  // Index of the last sub-matrix of A processed by the block
  int aEnd = aBegin + wA - 1;

  // Step size used to iterate through the sub-matrices of A
  int aStep = BLOCK_SIZE;

  // Index of the first sub-matrix of B processed by the block
  int bBegin = BLOCK_SIZE * bx;

  // Step size used to iterate through the sub-matrices of B
  int bStep = BLOCK_SIZE * wB;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub = 0;

  // Thread ids with respect to grid.
  int row = by * BLOCK_SIZE + ty;
  int col = bx * BLOCK_SIZE + tx;

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix
    // If a thread corresponds to an out-of-bounds index in 'A or B',
    // set the corresponding index in 'As or Bs' to 0
    if (aBegin + wA > a + tx && row < wA) {
      As[ty][tx] = A[a + wA * ty + tx];
    } else {
      As[ty][tx] = 0.0;
    }
    if (bBegin + wB * (wB - 1) >= b + wB * ty && col < wB) {
      Bs[ty][tx] = B[b + wB * ty + tx];
    } else {
      Bs[ty][tx] = 0.0;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
    for (int k = 0; k < BLOCK_SIZE; ++k) Csub += As[ty][k] * Bs[k][tx];

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write to C buffer only if threads lie within the A and B matrix indices
  if (row < wA && col < wB) {
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
  }
}

#ifdef USE_WR
void run_MATMUL_KERNEL(workRequest* wr, cudaStream_t kernel_stream,
                       void** devBuffers) {
  matMul<<<wr->grid_dim, wr->block_dim, wr->shared_mem, kernel_stream>>>(
      (float*)devBuffers[wr->getBufferID(C_INDEX)],
      (float*)devBuffers[wr->getBufferID(A_INDEX)],
      (float*)devBuffers[wr->getBufferID(B_INDEX)], *((int*)wr->getUserData()),
      *((int*)wr->getUserData()));
}

void run_BLAS_KERNEL(workRequest* wr, cudaStream_t kernel_stream,
                     void** devBuffers) {
  int size = *((int*)wr->getUserData());
  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, kernel_stream);

  // need to switch A and B due to how CuBLAS sees arrays in fortran style
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size, &alpha,
              (float*)devBuffers[wr->getBufferID(B_INDEX)], size,
              (float*)devBuffers[wr->getBufferID(A_INDEX)], size, &beta,
              (float*)devBuffers[wr->getBufferID(C_INDEX)], size);

  cublasDestroy(handle);
}
#endif

#ifdef USE_WR
void cudaMatMul(int matrixSize, float* h_A, float* h_B, float* h_C,
                cudaStream_t stream, void* cb) {
#else
void cudaMatMul(int matrixSize, float* h_A, float* h_B, float* h_C, float* d_A,
                float* d_B, float* d_C, cudaStream_t stream,
                cublasHandle_t handle, void* cb) {
#endif
  int size = matrixSize * matrixSize * sizeof(float);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(ceil((float)matrixSize / dimBlock.x),
               ceil((float)matrixSize / dimBlock.y));
#ifdef USE_WR
  // DEPRECATED
  workRequest* wr = hapiCreateWorkRequest();
  wr->setExecParams(dimGrid, dimBlock);
  wr->setStream(stream);
  wr->addBuffer(h_A, size, true, false, true);
  wr->addBuffer(h_B, size, true, false, true);
  wr->addBuffer(h_C, size, false, true, true);
  wr->setCallback(cb);
  if (useCublas) {
    wr->setTraceName("blas");
    wr->setRunKernel(run_BLAS_KERNEL);
  } else {
    wr->setTraceName("matmul");
    wr->setRunKernel(run_MATMUL_KERNEL);
  }
  wr->copyUserData(&matrixSize, sizeof(int));

  hapiEnqueue(wr);
#else
  if (!useCublas) {
    hapiCheck(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
    hapiCheck(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

    matMul<<<dimGrid, dimBlock, 0, stream>>>(d_C, d_A, d_B, matrixSize,
                                                matrixSize);
    hapiCheck(cudaPeekAtLastError());

    hapiCheck(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
  } else {
    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSetMatrixAsync(matrixSize, matrixSize, sizeof(float), h_A, matrixSize,
                         d_A, matrixSize, stream);
    cublasSetMatrixAsync(matrixSize, matrixSize, sizeof(float), h_B, matrixSize,
                         d_B, matrixSize, stream);

    // need to switch A and B due to how CuBLAS sees arrays in fortran style
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixSize, matrixSize,
                matrixSize, &alpha, d_B, matrixSize, d_A, matrixSize, &beta,
                d_C, matrixSize);

    cublasGetMatrixAsync(matrixSize, matrixSize, sizeof(float), d_C, matrixSize,
                         h_C, matrixSize, stream);
  }

  hapiAddCallback(stream, cb);
#endif
}
