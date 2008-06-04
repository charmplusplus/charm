#include <stdlib.h>
#include <stdio.h>
#include "wr.h"

extern workRequestQueue* wrQueue; 
extern void kernelReturn(); 

__global__ void helloKernel() { 

}

void kernelSetup() {
  workRequest *wr; 
  wr = (workRequest*) malloc(sizeof(workRequest)); 

  wr->dimGrid.x = 1; 
  wr->dimBlock.x = 1; 
  wr->smemSize = 0;
  
  wr->readWriteDevicePtr = NULL;
  wr->readWriteHostPtr = NULL; 
  wr->readWriteLen = 0; 

  wr->readOnlyDevicePtr = NULL;
  wr->readOnlyHostPtr = NULL;
  wr->readOnlyLen = 0; 

  wr->writeOnlyDevicePtr = NULL;
  wr->writeOnlyHostPtr = NULL;
  wr->writeOnlyLen = 0; 

  wr->callbackFn = kernelReturn; 

  wr->id = 0; 

  wr->executing = 0; 

  enqueue(wrQueue, wr); 

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
