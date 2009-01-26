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
#include <cutil.h>


/* A function in ck.C which casts the void * to a CkCallback object
 *  and executes the callback 
 */ 
extern void CUDACallbackManager(void * fn); 

/* initial size of host/device buffer arrays - dynamically expanded by
 *  the runtime system if needed
 */ 
#define NUM_BUFFERS 100

// #define GPU_DEBUG


/* a flag which tells the system to record the time for invocation and
 *  completion of GPU events: memory allocation, transfer and
 *  kernel execution
 */  
#define GPU_TIME

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


#ifdef GPU_TIME

unsigned int timerHandle; 

/* event types */
#define DATA_SETUP          1            
#define KERNEL_EXECUTION    2
#define DATA_CLEANUP        3

typedef struct gpuEventTimer {
  float startTime; 
  float endTime; 
  int eventType;
  int ID; 
} gpuEventTimer; 

gpuEventTimer gpuEvents[QUEUE_SIZE_INIT * 10]; 
int timeIndex = 0; 
int runningKernelIndex = 0; 
int dataSetupIndex = 0; 
int dataCleanupIndex = 0; 

#endif

/* There are separate CUDA streams for kernel execution, data transfer
 *  into the device, and data transfer out. This allows prefetching of
 *  data for a subsequent kernel while the previous kernel is
 *  executing and transferring data out of the device. 
 */
cudaStream_t kernel_stream; 
cudaStream_t data_in_stream;
cudaStream_t data_out_stream; 

/* allocateBuffers
 *
 * allocates a work request's data on the GPU
 *
 * used to allocate memory for work request data in advance in order
 * to allow overlapping the work request's data transfer to the GPU
 * with the execution of the previous kernel; the allocation needs to
 * take place before the kernel starts executing in order to allow overlap
 *
 */

/*
void allocateBuffers(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {

    for (int i=0; i<wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      
      // allocate if the buffer for the corresponding index is NULL 
      if (devBuffers[index] == NULL) {
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &devBuffers[index], size));
#ifdef GPU_DEBUG
	printf("buffer %d allocated %.2f\n", index, 
	       cutGetTimerValue(timerHandle)); 
#endif
      }
    }
  }
}
*/


/* setupData
 *  sets up data on the GPU before kernel execution 
 */
void setupData(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {
    for (int i=0; i<wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      hostBuffers[index] = bufferInfo[i].hostBuffer; 
      
      /* allocate if the buffer for the corresponding index is NULL */
      if (devBuffers[index] == NULL) {
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &devBuffers[index], size));
#ifdef GPU_DEBUG
	printf("buffer %d allocated %.2f\n", index,
	       cutGetTimerValue(timerHandle)); 
#endif
      }
      
      if (bufferInfo[i].transferToDevice) {
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(devBuffers[index], 
          hostBuffers[index], size, cudaMemcpyHostToDevice, data_in_stream));
#ifdef GPU_DEBUG
	printf("transferToDevice bufId: %d %.2f\n", index,
	       cutGetTimerValue(timerHandle)); 
#endif	
	/*
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(devBuffers[index], 
          hostBuffers[index], size, cudaMemcpyHostToDevice));
	*/

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
	printf("transferFromDevice: %d %.2f\n", index, 
	       cutGetTimerValue(timerHandle)); 
#endif
	
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(hostBuffers[index], 
          devBuffers[index], size, cudaMemcpyDeviceToHost,
          data_out_stream));
	
	/*
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(hostBuffers[index], 
          devBuffers[index], size, cudaMemcpyDeviceToHost));
	*/
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
	printf("buffer %d freed %.2f\n", index, cutGetTimerValue(timerHandle));
#endif 
	CUDA_SAFE_CALL_NO_SYNC(cudaFree(devBuffers[index])); 
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
  
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&kernel_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&data_in_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&data_out_stream)); 

#ifdef GPU_TIME
  CUT_SAFE_CALL(cutCreateTimer(&timerHandle));
  CUT_SAFE_CALL(cutStartTimer(timerHandle));
#endif
}

/* gpuProgressFn
 *  called periodically to monitor work request progress, and perform
 *  the prefetch of data for a subsequent work request
 */
void gpuProgressFn() {

  if (wrQueue == NULL) {
    printf("Error: work request queue not initialized\n"); 
    return; 
  }

  while (!isEmpty(wrQueue)) {
    int returnVal; 
    workRequest *head = firstElement(wrQueue); 
    workRequest *second = secondElement(wrQueue);
    workRequest *third = thirdElement(wrQueue); 
    
    if (head->state == QUEUED) {
#ifdef GPU_TIME
	gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].eventType = DATA_SETUP; 
	gpuEvents[timeIndex].ID = head->id; 
	dataSetupIndex = timeIndex; 
	timeIndex++; 
#endif
      setupData(head); 
      head->state = TRANSFERRING_IN; 
    }  

    if (head->state == TRANSFERRING_IN) {

      if ((returnVal = cudaStreamQuery(data_in_stream)) == cudaSuccess) {
#ifdef GPU_TIME
	gpuEvents[dataSetupIndex].endTime = cutGetTimerValue(timerHandle); 
#endif

#ifdef GPU_TIME
	gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].eventType = KERNEL_EXECUTION; 
	gpuEvents[timeIndex].ID = head->id; 
	runningKernelIndex = timeIndex; 
	timeIndex++; 
#endif
	
	if (second != NULL /*&& (second->state == QUEUED)*/) {
#ifdef GPU_TIME
	  gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	  gpuEvents[timeIndex].eventType = DATA_SETUP; 
	  gpuEvents[timeIndex].ID = second->id; 
	  dataSetupIndex = timeIndex; 
	  timeIndex++; 
#endif
	  setupData(second); 
	  second->state = TRANSFERRING_IN; 	
	}
	
	kernelSelect(head); 
	head->state = EXECUTING; 

      }
#ifdef GPU_DEBUG
      printf("Querying memory stream returned: %d %.2f\n", returnVal, 
	     cutGetTimerValue(timerHandle));
#endif  

    }

    if (head->state == EXECUTING) {
      if ((returnVal = cudaStreamQuery(kernel_stream)) == cudaSuccess) {
#ifdef GPU_TIME
	gpuEvents[runningKernelIndex].endTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].eventType = DATA_CLEANUP; 
	gpuEvents[timeIndex].ID = head->id; 
	dataCleanupIndex = timeIndex; 
	timeIndex++; 
#endif

	if (second != NULL && second->state == QUEUED) {
#ifdef GPU_TIME
	  gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	  gpuEvents[timeIndex].eventType = DATA_SETUP; 
	  gpuEvents[timeIndex].ID = second->id; 
	  dataSetupIndex = timeIndex; 
	  timeIndex++; 
#endif
	  setupData(second); 
	  second->state = TRANSFERRING_IN; 	
	} 

	if (second != NULL && second->state == TRANSFERRING_IN) {
	  if (cudaStreamQuery(data_in_stream) == cudaSuccess) {
#ifdef GPU_TIME
	    gpuEvents[dataSetupIndex].endTime = cutGetTimerValue(timerHandle); 
#endif
	    if (third != NULL /*&& (third->state == QUEUED)*/) {
#ifdef GPU_TIME
	      gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	      gpuEvents[timeIndex].eventType = DATA_SETUP; 
	      gpuEvents[timeIndex].ID = third->id; 
	      dataSetupIndex = timeIndex; 
	      timeIndex++; 
#endif
	      setupData(third); 
	      third->state = TRANSFERRING_IN; 	
	    }
	  }
	}

        copybackData(head);
	head->state = TRANSFERRING_OUT;

      }
#ifdef GPU_DEBUG
      printf("Querying kernel completion returned: %d %.2f\n", returnVal,
	     cutGetTimerValue(timerHandle));
#endif  
      
    }

    if (head->state == TRANSFERRING_OUT) {
      if (cudaStreamQuery(data_out_stream) == cudaSuccess) {
	freeMemory(head); 
#ifdef GPU_TIME
	gpuEvents[dataCleanupIndex].endTime = cutGetTimerValue(timerHandle);
#endif
	dequeue(wrQueue);
	CUDACallbackManager(head->callbackFn);
      }
    }
  }
}

/* exitHybridAPI
 *  cleans up and deletes memory allocated for the queue and the CUDA streams
 */
void exitHybridAPI() {
  deleteWRqueue(wrQueue); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(kernel_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(data_in_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(data_out_stream)); 

#ifdef GPU_TIME
  for (int i=0; i<timeIndex; i++) {
    switch (gpuEvents[i].eventType) {
    case DATA_SETUP:
      printf("Kernel %d data setup", gpuEvents[i].ID); 
      break;
    case DATA_CLEANUP:
      printf("Kernel %d data cleanup", gpuEvents[i].ID); 
      break; 
    case KERNEL_EXECUTION:
      printf("Kernel %d execution", gpuEvents[i].ID); 
      break;
    default:
      printf("Error, invalid timer identifier\n"); 
    }
    printf(" %.2f:%.2f\n", gpuEvents[i].startTime, gpuEvents[i].endTime); 
  }

  CUT_SAFE_CALL(cutStopTimer(timerHandle));
  CUT_SAFE_CALL(cutDeleteTimer(timerHandle));  

#endif
}
