/* 
 * cuda-hybrid-api.cu
 *
 * by Lukasz Wesolowski
 * 04.01.2008
 *
 * an interface for execution on the GPU
 *
 * description: 
 * -user enqueues one or more work requests to the work
 * request queue (wrQueue) to be executed on the GPU
 * - a converse function (gpuProgressFn) executes periodically to
 * offload work requests to the GPU one at a time
 *
 */

#include "wrqueue.h"
#include "cuda-hybrid-api.h"

workRequestQueue *wrQueue; 

/*
  TO DO
  stream 1 - kernel execution
  stream 2 - memory setup
  stream 3 - memory copies
*/

void setupMemory(workRequest *wr) {

  cudaMalloc((void **)&(wr->readWriteDevicePtr), wr->readWriteLen);
  cudaMalloc((void **)&(wr->readOnlyDevicePtr), wr->readOnlyLen); 
  cudaMalloc((void **)&(wr->writeOnlyDevicePtr), wr->writeOnlyLen);

  cudaMemcpy(wr->readWriteDevicePtr, wr->readWriteHostPtr, wr->readWriteLen, 
		  cudaMemcpyHostToDevice); 
  cudaMemcpy(wr->readOnlyDevicePtr, wr->readOnlyHostPtr, wr->readOnlyLen, 
		  cudaMemcpyHostToDevice); 
} 

void cleanupMemory(workRequest *wr) {

  cudaMemcpy(wr->readWriteHostPtr, wr->readWriteDevicePtr, wr->readWriteLen, cudaMemcpyDeviceToHost); 
  cudaMemcpy(wr->writeOnlyHostPtr, wr->writeOnlyDevicePtr, wr->writeOnlyLen, cudaMemcpyHostToDevice); 

  cudaFree(wr->readWriteDevicePtr); 
  cudaFree(wr->readOnlyDevicePtr); 
  cudaFree(wr->writeOnlyDevicePtr);

}

void kernelSelect(workRequest *wr);

void initHybridAPI() {
  init_wrqueue(wrQueue); 
}

void gpuProgressFn() {
  while (!isEmpty(wrQueue)) {

    workRequest *wr = head(wrQueue); 
    
    if (wr->executing == 0) {
      setupMemory(wr); 
      kernelSelect(wr); 
      cudaEventRecord(wr->completionEvent, 0); 
      return; 
    }  
    else if (cudaEventQuery(wr->completionEvent) == cudaSuccess ) {      
      cleanupMemory(wr);
      dequeue(wrQueue);
      wr->callbackFn();
    }
      
  }
}

void exitHybridAPI() {
  delete_wrqueue(wrQueue); 
}
