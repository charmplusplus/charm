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

workRequestQueue *wrQueue = NULL; 

/*
  TO DO
  stream 1 - kernel execution
  stream 2 - memory setup
  stream 3 - memory copies
*/

/* setupMemory
   set up memory on the gpu for this kernel's execution */
void setupMemory(workRequest *wr) {
  cudaMalloc((void **)&(wr->readWriteDevicePtr), wr->readWriteLen);
  cudaMalloc((void **)&(wr->readOnlyDevicePtr), wr->readOnlyLen); 
  cudaMalloc((void **)&(wr->writeOnlyDevicePtr), wr->writeOnlyLen);
  
  cudaMemcpy(wr->readWriteDevicePtr, wr->readWriteHostPtr, wr->readWriteLen, 
		  cudaMemcpyHostToDevice); 
  cudaMemcpy(wr->readOnlyDevicePtr, wr->readOnlyHostPtr, wr->readOnlyLen, 
		  cudaMemcpyHostToDevice); 
  
} 

/* cleanupMemory
   free memory no longer needed on the gpu */ 
void cleanupMemory(workRequest *wr) {
  
  cudaMemcpy(wr->readWriteHostPtr, wr->readWriteDevicePtr, wr->readWriteLen, cudaMemcpyDeviceToHost); 
  cudaMemcpy(wr->writeOnlyHostPtr, wr->writeOnlyDevicePtr, wr->writeOnlyLen, cudaMemcpyDeviceToHost); 
  

  cudaFree(wr->readWriteDevicePtr); 
  cudaFree(wr->readOnlyDevicePtr); 
  cudaFree(wr->writeOnlyDevicePtr);

}

/* kernelSelect
   a switch statement defined by the user to allow the library to execute
   the correct kernel */ 
void kernelSelect(workRequest *wr);

/* initHybridAPI
   initializes the work request queue
*/
void initHybridAPI() {
  initWRqueue(&wrQueue); 
}

/* gpuProgressFn
   called periodically to check if the current kernel has completed,
   and invoke subsequent kernel */
void gpuProgressFn() {
  if (wrQueue == NULL) {
    return; 
  }

  while (!isEmpty(wrQueue)) {
    workRequest *wr = head(wrQueue); 
    
    if (wr->executing == 0) {
      setupMemory(wr); 
      kernelSelect(wr); 
      // cudaEventRecord(wr->completionEvent, 0);
      wr->executing = 1; 
      return; 
    }  
    // else if (cudaEventQuery(wr->completionEvent) == cudaSuccess ) {      
    else if (cudaStreamQuery(0) == cudaSuccess ) {      
      cleanupMemory(wr);
      dequeue(wrQueue);
      wr->callbackFn();
    }
      
  }
}

/* exitHybridAPI
   cleans up and deletes memory allocated for the queue
*/
void exitHybridAPI() {
  deleteWRqueue(wrQueue); 
}
