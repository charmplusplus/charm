#include "overlapTestConsts.h"
#include "wr.h"
#include <stdio.h>
#include <math.h>
#include "cublas_v2.h"
// matrix multiplication code adapted from the CUDA SDK

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

    // Thread ids with respect to grid.
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

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
        // If a thread corresponds to an out-of-bounds index in 'A or B',
        // set the corresponding index in 'As or Bs' to 0
        if(aBegin + wA > a + tx && row < wA) {
          As[ty][tx] = A[a + wA * ty + tx];
        }
        else {
          As[ty][tx] = 0.0;
        }
        if(bBegin + wB*(wB-1) >= b+ wB*ty && col<wB) {
          Bs[ty][tx] = B[b + wB * ty + tx];
        }
        else {
          Bs[ty][tx] = 0.0;
        }

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write to C buffer only if threads lie within the A and B matrix indices
    if(row < wA && col < wB) {
      // Write the block sub-matrix to device memory;
      // each thread writes one element
      int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
      C[c + wB * ty + tx] = Csub;
    }
}

void hostMemorySetup(int matrixSize, ElementType **h_A_ptr,
                     ElementType **h_B_ptr, ElementType **h_C_ptr, void *cb) {
  pinnedMemReq reqs;

  int nBuffers = 3;
  int size = matrixSize * matrixSize * sizeof(ElementType);

  size_t *sizes = (size_t *) malloc(nBuffers * sizeof(size_t));
  void ***hostPtrs = (void ***) malloc(nBuffers * sizeof(void **));
  hostPtrs[0] = (void **) h_A_ptr;
  hostPtrs[1] = (void **) h_B_ptr;
  hostPtrs[2] = (void **) h_C_ptr;
  sizes[0] = size;
  sizes[1] = size;
  sizes[2] = size;

  reqs.nBuffers = nBuffers;
  reqs.sizes = sizes;
  reqs.hostPtrs = hostPtrs;
  reqs.callbackFn = cb;

  pinnedMallocHost(&reqs);
}

void hostMemoryCleanup(ElementType *h_A, ElementType *h_B, ElementType *h_C) {

  delayedFree(h_A);
  delayedFree(h_B);
  delayedFree(h_C);

}

void cudaMatMul(int matrixSize, ElementType *h_A, ElementType *h_B,
                ElementType *h_C, int myIndex, void *cb,int useCublas) {
  int size = matrixSize * matrixSize * sizeof(ElementType);
  dataInfo *AInfo, *BInfo, *CInfo;

  workRequest matmul;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  matmul.dimGrid = dim3( ceil((float)matrixSize / threads.x),
                         ceil((float)matrixSize / threads.y) );
  matmul.dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
  matmul.smemSize = 0;
  matmul.nBuffers = 3;
  matmul.bufferInfo = (dataInfo *) malloc(matmul.nBuffers * sizeof(dataInfo));

  AInfo = &(matmul.bufferInfo[0]);
  AInfo->bufferID = BUFFERS_PER_CHARE * myIndex + A_INDEX;
  AInfo->transferToDevice = YES;
  AInfo->transferFromDevice = NO;
  AInfo->freeBuffer = YES;
  AInfo->hostBuffer = h_A;
  AInfo->size = size;

  BInfo = &(matmul.bufferInfo[1]);
  BInfo->bufferID = BUFFERS_PER_CHARE * myIndex + B_INDEX;
  BInfo->transferToDevice = YES;
  BInfo->transferFromDevice = NO;
  BInfo->freeBuffer = YES;
  BInfo->hostBuffer = h_B;
  BInfo->size = size;

  CInfo = &(matmul.bufferInfo[2]);
  CInfo->bufferID = BUFFERS_PER_CHARE * myIndex + C_INDEX;
  CInfo->transferToDevice = NO;
  CInfo->transferFromDevice = YES;
  CInfo->freeBuffer = YES;
  CInfo->hostBuffer = h_C;
  CInfo->size = size;

  matmul.callbackFn = cb;
  if(useCublas)
   matmul.id = BLAS_KERNEL;
  else
   matmul.id = MATMUL_KERNEL;

  matmul.userData = malloc(sizeof(int));
  memcpy(matmul.userData, &matrixSize, sizeof(int));

  enqueue(wrQueue, &matmul);
}

void kernelSelect(workRequest *wr) {

  switch (wr->id) {
  case MATMUL_KERNEL:
    printf("MATMUL KERNEL");
    matrixMul<<< wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream >>>
      ((ElementType *) devBuffers[wr->bufferInfo[C_INDEX].bufferID],
       (ElementType *) devBuffers[wr->bufferInfo[A_INDEX].bufferID],
       (ElementType *) devBuffers[wr->bufferInfo[B_INDEX].bufferID],
       *((int *) wr->userData), *((int *) wr->userData));
    break;
  case BLAS_KERNEL:
    printf("CUBLAS KERNEL");
    int size=*((int *) wr->userData);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle,kernel_stream);
    float alpha=1.0;
    float beta=0.0;
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, size, size, size, &alpha, (ElementType *) devBuffers[wr->bufferInfo[A_INDEX].bufferID], size, (ElementType *) devBuffers[wr->bufferInfo[B_INDEX].bufferID], size, &beta, (ElementType *) devBuffers[wr->bufferInfo[C_INDEX].bufferID], size);
    cublasDestroy(handle);
    break;
  }
}
