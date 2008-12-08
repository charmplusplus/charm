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


/* A function in ck.C which casts the void * to a CkCallback object
 *  and executes the callback 
 */ 
extern void CUDACallbackManager(void * fn); 

/* initial size of host/device buffer arrays - dynamically expanded by
 *  the runtime system if needed
 */ 
#define NUM_BUFFERS 100

//#define GPU_DEBUG


/* a flag which tells the system to record the time for invocation and
 *  completion of GPU events: memory allocation, transfer and
 *  kernel execution
 */  
// #define GPUTIME

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


#ifdef GPUTIME
#include <time.h>
/* event types */
#define MEMORY_ALLOCATION     1
#define CPU_GPU_TRANSFER      2
#define GPU_CPU_TRANSFER      3
#define MEMORY_FREE           4
#define KERNEL_EXECUTION      5

typedef struct gpuEventTimer {
  clock_t startTime; 
  clock_t endTime; 
  int eventType;
  int ID; 
} gpuEventTimer; 

gpuEventTimer gpuEvents[QUEUE_SIZE_INIT * 3]; 
int timeIndex = 0; 

#endif

/* There are separate CUDA streams for kernel execution, data transfer
 *  into the device, and data transfer out. This allows prefetching of
 *  data for a subsequent kernel while the previous kernel is
 *  executing and transferring data out of the device. 
 */
cudaStream_t kernel_stream; 
cudaStream_t data_in_stream;
cudaStream_t data_out_stream; 

/* setupData
 *  sets up data on the gpu before kernel execution 
 */
void setupData(workRequest *wr) {
  int returnVal;
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
#ifdef GPUTIME
	gpuEvents[timeIndex].startTime = clock(); 
	gpuEvents[timeIndex].eventType = MEMORY_ALLOCATION; 
	gpuEvents[timeIndex].ID = index; 
#endif
	returnVal = cudaMalloc((void **) &devBuffers[index], size); 
#ifdef GPUTIME
	gpuEvents[timeIndex].endTime = clock(); 
	timeIndex++;
#endif
#ifdef GPU_DEBUG
	printf("cudaMalloc returned %d\n", returnVal); 
#endif
      }
      
      if (bufferInfo[i].transferToDevice) {
#ifdef GPU_DEBUG
	printf("transferToDevice bufId: %d\n", index); 
#endif

	cudaMemcpyAsync(devBuffers[index], hostBuffers[index], size, 
			cudaMemcpyHostToDevice, data_in_stream);
      }
    }
  }
} 

/* copybackData
 *  transfer data from the GPU to the CPU after a work request is done 
 */ 
void copybackData(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {
    int nBuffers = wr->nBuffers; 
    
    for (int i=0; i<nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      
      if (bufferInfo[i].transferFromDevice) {
#ifdef GPU_DEBUG
	printf("transferFromDevice: %d\n", index); 
#endif

	cudaMemcpyAsync(hostBuffers[index], devBuffers[index], size,
			cudaMemcpyDeviceToHost, data_out_stream);
      }
    }     
  }
}

/* frees GPU memory for buffers specified by the user; also frees the
 *  work request's bufferInfo array
 */
void freeMemory(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo;   
  int nBuffers = wr->nBuffers; 
  if (bufferInfo != NULL) {
    for (int i=0; i<nBuffers; i++) {    
      int index = bufferInfo[i].bufferID; 
      if (bufferInfo[i].freeBuffer) {
#ifdef GPU_DEBUG
	printf("buffer %d freed\n", index);
#endif 
	cudaFree(devBuffers[index]); 
	devBuffers[index] = NULL; 
      }
    }
    free(bufferInfo); 
  }
}

/* kernelSelect
 * a switch statement defined by the user to allow the library to execute
 * the correct kernel 
 */ 
void kernelSelect(workRequest *wr);

/* initHybridAPI
 *   initializes the work request queue, host/device buffer pointer
 *   arrays, and CUDA streams
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
  
  cudaStreamCreate(&kernel_stream); 
  cudaStreamCreate(&data_in_stream); 
  cudaStreamCreate(&data_out_stream); 

}

/* gpuProgressFn
 *  called periodically to monitor work request progress, and perform
 *  the prefetch of data for a subsequent work request
 */
void gpuProgressFn() {

  if (wrQueue == NULL) {
    return; 
  }

  while (!isEmpty(wrQueue)) {
    int returnVal; 
    workRequest *wr = head(wrQueue); 
    workRequest *second = next(wrQueue); 
    
    if (wr->state == QUEUED) {
      setupData(wr); 
      wr->state = TRANSFERRING_IN; 
      return; 
    }  
    else if (wr->state == TRANSFERRING_IN) {
      if ((returnVal = cudaStreamQuery(data_in_stream)) == cudaSuccess) {

#ifdef GPUTIME
	gpuEvents[timeIndex].startTime = clock(); 
	gpuEvents[timeIndex].eventType = KERNEL_EXECUTION; 
	gpuEvents[timeIndex].ID = wr->id; 
#endif
	kernelSelect(wr); 
	wr->state = EXECUTING; 
      }
#ifdef GPU_DEBUG
      printf("Querying memory stream returned: %d\n", returnVal);
#endif  
    }
    else if (wr->state == EXECUTING) {
      if ((returnVal = cudaStreamQuery(kernel_stream)) == cudaSuccess) {
#ifdef GPUTIME
	gpuEvents[timeIndex].endTime = clock(); 
	timeIndex++; 
#endif
        copybackData(wr);
	wr->state = TRANSFERRING_OUT;
      }
#ifdef GPU_DEBUG
      printf("Querying kernel completion returned: %d \n", returnVal);
#endif  

      /* prefetch data for the subsequent kernel */
      if (second != NULL && second->state == QUEUED) {
	setupData(second); 
	second->state = TRANSFERRING_IN; 
	return; 
      }
    }
    else if (wr->state == TRANSFERRING_OUT) {
      if (cudaStreamQuery(data_out_stream) == cudaSuccess) {
	freeMemory(wr); 
	dequeue(wrQueue);
	CUDACallbackManager(wr->callbackFn);
      }
    }
#ifdef GPU_DEBUG
    else {
      printf("Error: unrecognized state\n"); 
      return; 
    }
#endif
  }
}

/* exitHybridAPI
 *  cleans up and deletes memory allocated for the queue and the CUDA streams
 */
void exitHybridAPI() {
  deleteWRqueue(wrQueue); 
  cudaStreamDestroy(kernel_stream); 
  cudaStreamDestroy(data_in_stream); 
  cudaStreamDestroy(data_out_stream); 

#ifdef GPU_DEBUG
  for (int i=0; i<timeIndex; i++) {
    switch (gpuEvents[timeIndex].eventType) {
    case (MEMORY_ALLOCATION) :
      printf("Buffer %d allocation ", gpuEvents[timeIndex].ID); 
      break; 
    case (KERNEL_EXECUTION) :
      printf("Kernel %d execution ", gpuEvents[timeIndex].ID); 
      break;
    default:
      printf("Error, invalid timer identifier\n"); 
    }
    printf("%d:%d\n", gpuEvents[timeIndex].startTime, gpuEvents[timeIndex].endTime); 
  }
#endif
}






