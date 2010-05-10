#include <stdlib.h>
#include <stdio.h>
#include "wr.h"

extern workRequestQueue* wrQueue;

__global__ void helloKernel() {

}

extern "C"
void *kernelSetup() {
  workRequest *wr = new workRequest;
  wr = (workRequest*) malloc(sizeof(workRequest));

  wr->dimGrid.x = 1;
  wr->dimBlock.x = 1;
  wr->smemSize = 0;
  wr->id = 0;
  wr->nBuffers = 0;
  wr->bufferInfo = NULL;

  return wr;
}

void kernelSelect(workRequest *wr) {
  printf("inside kernelSelect\n");
  switch (wr->id) {
  case 0:
    printf("calling kernel\n");
    helloKernel<<<wr->dimGrid,wr->dimBlock,wr->smemSize>>>();
    break;
  default:
    printf("error: id %d not valid\n", wr->id);
    break;
  }
}
