#include <stdlib.h>
#include <stdio.h>
#include "wr.h"

extern workRequestQueue* wrQueue;

__global__ void helloKernel() {

}

void run_hello(workRequest *wr, cudaStream_t kernel_stream, void **devBuffers) {
  printf("calling kernel\n");
  helloKernel<<<wr->dimGrid,wr->dimBlock,wr->smemSize,kernel_stream>>>();
}

extern "C"
void *kernelSetup() {
  workRequest *wr = new workRequest;
  wr = (workRequest*) malloc(sizeof(workRequest));

  wr->dimGrid.x = 1;
  wr->dimBlock.x = 1;
  wr->smemSize = 0;
  wr->traceName = "hello";
  wr->runKernel = run_hello;
  wr->nBuffers = 0;
  wr->bufferInfo = NULL;

  return wr;
}
