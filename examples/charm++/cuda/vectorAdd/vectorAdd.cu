#include "vectorAddConsts.h"
#include "wr.h"
#include <stdio.h>
#include <math.h>

__global__ void vecAdd(float *C, float *A, float *B, int n)
{
    // Get our global thread ID
  int id = blockIdx.x*blockDim.x+threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
  {
    C[id] = A[id] + B[id];
  }
}

void createWorkRequest(int vectorSize, float *h_A, float *h_B,
                float **h_C, int myIndex, void *cb) {
  int size = vectorSize * vectorSize * sizeof(float);
  dataInfo *AInfo, *BInfo, *CInfo;

  workRequest *vectorAddReq = new workRequest;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  vectorAddReq->dimGrid = dim3((vectorSize -1 / threads.x +1),1,1 );
  vectorAddReq->dimBlock = dim3(BLOCK_SIZE,1);
  vectorAddReq->smemSize = 0;
  vectorAddReq->nBuffers = 3;
  vectorAddReq->bufferInfo = (dataInfo *) malloc(vectorAddReq->nBuffers * sizeof(dataInfo));

  AInfo = &(vectorAddReq->bufferInfo[0]);
 // AInfo->bufferID = BUFFERS_PER_CHARE * myIndex + A_INDEX;
  AInfo->bufferID = -1;
  AInfo->transferToDevice = YES;
  AInfo->transferFromDevice = NO;
  AInfo->freeBuffer = YES;
  AInfo->hostBuffer = hapi_poolMalloc(size);
  memcpy(AInfo->hostBuffer, h_A, size);
  AInfo->size = size;

  BInfo = &(vectorAddReq->bufferInfo[1]);
 // BInfo->bufferID = BUFFERS_PER_CHARE * myIndex + B_INDEX;
  BInfo->bufferID = -1;
  BInfo->transferToDevice = YES;
  BInfo->transferFromDevice = YES;
  BInfo->freeBuffer = YES;
  BInfo->hostBuffer = hapi_poolMalloc(size);
  memcpy(BInfo->hostBuffer, h_B, size);
  BInfo->size = size;

  CInfo = &(vectorAddReq->bufferInfo[2]);
  //CInfo->bufferID = BUFFERS_PER_CHARE * myIndex + C_INDEX;
  CInfo->bufferID = -1;
  CInfo->transferToDevice = NO;
  CInfo->transferFromDevice = YES;
  CInfo->freeBuffer = YES;
  CInfo->hostBuffer = hapi_poolMalloc(size);
  *h_C = (float *)CInfo->hostBuffer ;
  CInfo->size = size;

  vectorAddReq->callbackFn = cb;
  vectorAddReq->id = VECTOR_KERNEL;

  vectorAddReq->userData = malloc(sizeof(int));
  memcpy(vectorAddReq->userData, &vectorSize, sizeof(int));

  enqueue(wrQueue, vectorAddReq);
}

void kernelSelect(workRequest *wr) {

  switch (wr->id) {
    case VECTOR_KERNEL:
      printf("VECTOR KERNEL");
      vecAdd<<< wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream >>>
      ((float *) devBuffers[wr->bufferInfo[C_INDEX].bufferID],
       (float *) devBuffers[wr->bufferInfo[A_INDEX].bufferID],
       (float *) devBuffers[wr->bufferInfo[B_INDEX].bufferID],
       *((int *) wr->userData));
    break;
  }
}
