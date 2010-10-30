#include "overlapTestConsts.h"

#if CHECK_BANK_CONFLICTS
#define AS(i, j) CUT_BANK_CHECKER(((float*)&As[0][0]), (BLOCK_SIZE * i + j))
#define BS(i, j) CUT_BANK_CHECKER(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif

// matrix multiplication code taken from the CUDA SDK

__global__ void
matrixMul(float* C, float* A, float* B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + wA * ty + tx];
        BS(ty, tx) = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void cudaMatMul(int matrixSize, ElementType *A, ElementType *B, ElementType *C) {
  cudaStream_t stream; 
  cudaStreamCreate(&stream); 
  ElementType *h_A, *h_B, *h_C; 
  ElementType *d_A, *d_B, *d_C;
  int size = matrixSize * matrixSize * sizeof(ElementType);

  cudaMallocHost((void **) &h_A, size); 
  cudaMallocHost((void **) &h_B, size); 
  cudaMallocHost((void **) &h_C, size);  

  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);
  cudaMalloc((void **) &d_C, size);

  memcpy(h_A, A, size);
  memcpy(h_B, B, size); 

  cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream); 
  cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream); 

  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(matrixSize / threads.x, matrixSize / threads.y);
  
  // execute the kernel
  matrixMul<<< grid, threads, 0, stream >>>(d_C, d_A, d_B, matrixSize, matrixSize);  

  cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream); 

  cudaStreamSynchronize(stream); 

  memcpy(C, h_C, size);

  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaStreamDestroy(stream); 
}
