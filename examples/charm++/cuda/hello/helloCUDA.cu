#include <stdlib.h>
#include <stdio.h>
#include "wr.h"

extern workRequestQueue* wrQueue; 

__global__ void helloKernel() { 

}

void kernelSetup(void *cb) {
  workRequest wr; 
  wr.dimGrid = dim3(1, 1);
  wr.dimBlock = dim3(1,1);
  wr.smemSize = 0;
  wr.nBuffers = 0; 
  wr.bufferInfo = NULL;
  wr.callbackFn = cb; 
  wr.id = 0; 

  enqueue(wrQueue, &wr); 

}

void kernelSelect(workRequest *wr) {
  printf("inside kernelSelect\n"); 
  switch (wr->id) {
  case 0: 
    printf("calling kernel\n"); 
    helloKernel<<<wr->dimGrid,wr->dimBlock,wr->smemSize, kernel_stream>>>();
    break;
  default:
    printf("error: id %d not valid\n", wr->id); 
    break; 
  }
}
