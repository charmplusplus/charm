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
#include "stdio.h"

extern void CUDACallbackManager(void *); 

/* initial size of host/device buffer arrays - dynamically expanded by
 *  the runtime system if needed
 */ 
#define NUM_BUFFERS 100
#define GPU_DEBUG 0


/* work request queue */
workRequestQueue *wrQueue = NULL; 

/* The runtime system keeps track of all allocated buffers on the GPU.
 * The following arrays contain pointers to host (CPU) data and the
 * corresponding data on the device (GPU). 
 */ 

/* host buffers  */ 
void **hostBuffers = NULL; 

/* device buffers */
void **devBuffers = NULL; 

extern void CUDACallbackManager(void * fn); 

/* setupMemory
   set up memory on the gpu for this kernel's execution */
void setupMemory(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {
    for (int i=0; i<wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      hostBuffers[index] = bufferInfo[i].hostBuffer; 
      
      /* allocate if the buffer for the corresponding index is NULL */
      if (devBuffers[index] == NULL) {
#ifdef GPU_DEBUG
	printf("buffer %d allocated\n", index); 
#endif
	cudaMalloc((void **)&devBuffers[index], size); 
      }
      
      if (bufferInfo[i].transferToDevice) {
#ifdef GPU_DEBUG
	printf("transferToDevice: %d bufId: %d\n", i, index); 
#endif
	cudaMemcpy(devBuffers[index], hostBuffers[index], size, 
		   cudaMemcpyHostToDevice);
      }
    }
  }
} 

/* cleanupMemory
   free memory no longer needed on the gpu */ 
void cleanupMemory(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {
    int nBuffers = wr->nBuffers; 
    
    for (int i=0; i<nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      
      if (bufferInfo[i].transferFromDevice) {
#ifdef GPU_DEBUG
	printf("transferFromDevice: %d\n", i); 
#endif
	cudaMemcpy(hostBuffers[index], devBuffers[index], size, cudaMemcpyDeviceToHost);
      }
      
      if (bufferInfo[i].freeBuffer) {
#ifdef GPU_DEBUG
	printf("buffer %d freed\n", i);
#endif 
	cudaFree(devBuffers[index]); 
	devBuffers[index] = NULL; 
      }
    }
  }
}

/* kernelSelect
 * a switch statement defined by the user to allow the library to execute
 * the correct kernel 
 */ 
void kernelSelect(workRequest *wr);

/* initHybridAPI
 *   initializes the work request queue and host/device buffer pointer arrays
 */
void initHybridAPI() {
  initWRqueue(&wrQueue);

  /* allocate host/device buffers array */
  hostBuffers = (void **) malloc(NUM_BUFFERS * sizeof(void *)); 
  devBuffers = (void **) malloc(NUM_BUFFERS * sizeof(void *)); 

  /* initialize device array to NULL */ 
  for (int i=0; i<NUM_BUFFERS; i++) {
    devBuffers[i] = NULL; 
  }
  
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
      wr->executing = 1; 
      return; 
    }  
    else if (cudaStreamQuery(0) == cudaSuccess ) {      
#ifdef GPU_DEBUG
      printf("completion event success \n");
#endif  
      cleanupMemory(wr);
      dequeue(wrQueue);
      CUDACallbackManager(wr->callbackFn);
    } 
#ifdef GPU_DEBUG
    else {
      printf("GPU is busy, return code %d\n", cudaStreamQuery(0)); 
      return; 
    }
#endif
  }
}

/* exitHybridAPI
   cleans up and deletes memory allocated for the queue
*/
void exitHybridAPI() {
  deleteWRqueue(wrQueue); 

}
