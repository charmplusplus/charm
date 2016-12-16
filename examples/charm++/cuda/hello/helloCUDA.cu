#include <stdlib.h>
#include <stdio.h>
#include "wr.h"

extern workRequestQueue* wrQueue; 

__global__ void helloKernel() { 

}

void run_hello(struct workRequest *wr, cudaStream_t kernel_stream, void **deviceBuffers)
{
    printf("calling kernel\n");
    helloKernel<<<wr->dimGrid,wr->dimBlock,wr->smemSize, kernel_stream>>>();
}

void kernelSetup(void *cb) {
  workRequest wr; 
  wr.dimGrid = dim3(1, 1);
  wr.dimBlock = dim3(1,1);
  wr.smemSize = 0;
  wr.nBuffers = 0; 
  wr.bufferInfo = NULL;
  wr.callbackFn = cb; 
  wr.traceName = "hello";
  wr.runKernel = run_hello;

  enqueue(&wr);
}

