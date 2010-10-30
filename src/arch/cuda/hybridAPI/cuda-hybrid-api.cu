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

#if defined GPU_MEMPOOL || defined GPU_INSTRUMENT_WRS
#include "cklists.h"
#endif

/* A function in ck.C which casts the void * to a CkCallback object
 *  and executes the callback 
 */ 
extern void CUDACallbackManager(void * fn); 
extern int CmiMyPe();

/* initial size of the user-addressed portion of host/device buffer
 * arrays; the system-addressed portion of host/device buffer arrays
 * (used when there is no need to share buffers between work requests)
 * will be equivalant in size.  
 */ 
#define NUM_BUFFERS 256
#define MAX_PINNED_REQ 64  
#define MAX_DELAYED_FREE_REQS 64  

/* a flag which tells the system to record the time for invocation and
 *  completion of GPU events: memory allocation, transfer and
 *  kernel execution
 */  
//#define GPU_PROFILE
//#define GPU_DEBUG
//#define GPU_TRACE
//#define _DEBUG

/* work request queue */
workRequestQueue *wrQueue = NULL; 

/* pending page-locked memory allocation requests */
unsigned int pinnedMemQueueIndex = 0; 
pinnedMemReq pinnedMemQueue[MAX_PINNED_REQ];

unsigned int currentDfr = 0;
void *delayedFreeReqs[MAX_DELAYED_FREE_REQS];

#ifdef GPU_MEMPOOL
#define GPU_MEMPOOL_NUM_SLOTS 15

CkVec<BufferPool> memPoolFreeBufs;
CkVec<int> memPoolBoundaries;
//int memPoolBoundaries[GPU_MEMPOOL_NUM_SLOTS];
#endif

/* The runtime system keeps track of all allocated buffers on the GPU.
 * The following arrays contain pointers to host (CPU) data and the
 * corresponding data on the device (GPU). 
 */ 

/* host buffers  */ 
void **hostBuffers = NULL; 

/* device buffers */
void **devBuffers = NULL; 

/* used to assign bufferIDs automatically by the system if the user 
   specifies an invalid bufferID */
unsigned int nextBuffer; 

unsigned int timerHandle; 

#ifdef GPU_PROFILE

/* event types */
#define DATA_SETUP          1            
#define KERNEL_EXECUTION    2
#define DATA_CLEANUP        3

typedef struct gpuEventTimer {
  float startTime; 
  float endTime; 
  int eventType;
  int ID; 
#ifdef GPU_TRACE
  int stage; 
  double cmistartTime; 
  double cmiendTime; 
#endif
} gpuEventTimer; 

gpuEventTimer gpuEvents[QUEUE_SIZE_INIT * 3]; 
unsigned int timeIndex = 0; 
unsigned int runningKernelIndex = 0; 
unsigned int dataSetupIndex = 0; 
unsigned int dataCleanupIndex = 0; 

#if defined GPU_TRACE || defined GPU_INSTRUMENT_WRS
extern "C" double CmiWallTimer(); 
#endif

#ifdef GPU_TRACE
extern "C" int traceRegisterUserEvent(const char*x, int e);
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);

#define GPU_MEM_SETUP 8800
#define GPU_KERNEL_EXEC 8801
#define GPU_MEM_CLEANUP 8802

#endif

#endif

#ifdef GPU_INSTRUMENT_WRS
CkVec<CkVec<CkVec<RequestTimeInfo> > > avgTimes;
bool initialized_instrument;
bool initializedInstrument();
#endif

/* There are separate CUDA streams for kernel execution, data transfer
 *  into the device, and data transfer out. This allows prefetching of
 *  data for a subsequent kernel while the previous kernel is
 *  executing and transferring data out of the device. 
 */
cudaStream_t kernel_stream; 
cudaStream_t data_in_stream;
cudaStream_t data_out_stream; 

/* pinnedMallocHost
 *
 * schedules a pinned memory allocation so that it does not impede
 * concurrent asynchronous execution 
 *
 */
void pinnedMallocHost(pinnedMemReq *reqs) {

  if ( (cudaStreamQuery(kernel_stream) == cudaSuccess) &&
       (cudaStreamQuery(data_in_stream) == cudaSuccess) &&
       (cudaStreamQuery(data_out_stream) == cudaSuccess) ) {    



    for (int i=0; i<reqs->nBuffers; i++) {
      CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **) reqs->hostPtrs[i], 
					    reqs->sizes[i])); 
    }

    free(reqs->hostPtrs);
    free(reqs->sizes);

    CUDACallbackManager(reqs->callbackFn);
    gpuProgressFn(); 
  }
  else {
    pinnedMemQueue[pinnedMemQueueIndex].hostPtrs = reqs->hostPtrs;
    pinnedMemQueue[pinnedMemQueueIndex].sizes = reqs->sizes; 
    pinnedMemQueue[pinnedMemQueueIndex].nBuffers = reqs->nBuffers; 
    pinnedMemQueue[pinnedMemQueueIndex].callbackFn = reqs->callbackFn;     
    pinnedMemQueueIndex++;
    if (pinnedMemQueueIndex == MAX_PINNED_REQ) {
      printf("Error: pinned memory request buffer is overflowing\n"); 
    }
  }
}

void delayedFree(void *ptr){
  if(currentDfr == MAX_DELAYED_FREE_REQS){
    printf("Ran out of DFR queue space. Increase MAX_DELAYED_FREE_REQS\n");
    exit(-1);
  }
  else{
    delayedFreeReqs[currentDfr] = ptr;
  }
  currentDfr++;
}

void flushDelayedFrees(){
  for(int i = 0; i < currentDfr; i++){
    if(delayedFreeReqs[i] == NULL){
      printf("recorded NULL ptr in delayedFree()");
      exit(-1);
    }
    cudaFreeHost(delayedFreeReqs[i]);
  }
  currentDfr = 0; 
}

/* flushPinnedMemQueue
 *
 * executes pending pinned memory allocation requests
 *
 */
void flushPinnedMemQueue() {

  for (int i=0; i<pinnedMemQueueIndex; i++) {
    pinnedMemReq *req = &pinnedMemQueue[i]; 
    for (int j=0; j<req->nBuffers; j++) {
      CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **) req->hostPtrs[j], 
					    req->sizes[j])); 
    }
    free(req->hostPtrs);
    free(req->sizes);
    CUDACallbackManager(pinnedMemQueue[i].callbackFn);    
  }
  pinnedMemQueueIndex = 0; 

}

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

void allocateBuffers(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {

    for (int i=0; i<wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 

      // if index value is invalid, use an available ID  
      if (index < 0 || index >= NUM_BUFFERS) {
	int found = 0; 
	for (int j=nextBuffer; j<NUM_BUFFERS*2; j++) {
	  if (devBuffers[j] == NULL) {
	    index = j;
	    found = 1; 
	    break;
	  }
	}

	/* if no index was found, try to search for a value at the
	 * beginning of the system addressed space 
	 */
	
	if (!found) {
	  for (int j=NUM_BUFFERS; j<nextBuffer; j++) {
	    if (devBuffers[j] == NULL) {	
	      index = j;
	      found = 1; 
	      break;
	    }
	  }
	}

	/* if no index was found, print an error */
	if (!found) {
	  printf("Error: devBuffers is full \n");
	}

	nextBuffer = index+1; 
	if (nextBuffer == NUM_BUFFERS * 2) {
	  nextBuffer = NUM_BUFFERS; 
	}
	
	bufferInfo[i].bufferID = index; 

      }      
      
      // allocate if the buffer for the corresponding index is NULL 
      if (devBuffers[index] == NULL && size > 0) {
#ifdef GPU_PRINT_BUFFER_ALLOCATE
        double mil = 1e3;
        printf("*** ALLOCATE buffer 0x%x (%d) size %f kb\n", devBuffers[index], index, 1.0*size/mil);

#endif

        CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &devBuffers[index], size));
#ifdef GPU_DEBUG
        printf("buffer %d allocated at time %.2f size: %d error string: %s\n", 
	       index, cutGetTimerValue(timerHandle), size, 
	       cudaGetErrorString( cudaGetLastError() ) );
#endif
      }
    }
  }
}


/* setupData
 *  copy data to the GPU before kernel execution 
 */
void setupData(workRequest *wr) {
  dataInfo *bufferInfo = wr->bufferInfo; 

  if (bufferInfo != NULL) {
    for (int i=0; i<wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID; 
      int size = bufferInfo[i].size; 
      hostBuffers[index] = bufferInfo[i].hostBuffer; 
      
      /* allocate if the buffer for the corresponding index is NULL */
      /*
      if (devBuffers[index] == NULL) {
	CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &devBuffers[index], size));
#ifdef GPU_DEBUG
	printf("buffer %d allocated %.2f\n", index,
	       cutGetTimerValue(timerHandle)); 
#endif
      }
      */
      
      if (bufferInfo[i].transferToDevice && size > 0) {
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(devBuffers[index], 
          hostBuffers[index], size, cudaMemcpyHostToDevice, data_in_stream));
#ifdef GPU_DEBUG
	printf("transferToDevice bufId: %d at time %.2f size: %d " 
	       "error string: %s\n", index, cutGetTimerValue(timerHandle), 
	       size, cudaGetErrorString( cudaGetLastError() )); 
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
      
      if (bufferInfo[i].transferFromDevice && size > 0) {
#ifdef GPU_DEBUG
	printf("transferFromDevice: %d at time %.2f size: %d "
	       "error string: %s\n", index, cutGetTimerValue(timerHandle), 
	       size, cudaGetErrorString( cudaGetLastError() )); 
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
#ifdef GPU_PRINT_BUFFER_ALLOCATE
        printf("*** FREE buffer 0x%x (%d)\n", devBuffers[index], index);
#endif

#ifdef GPU_DEBUG
        printf("buffer %d freed at time %.2f error string: %s\n", 
	       index, cutGetTimerValue(timerHandle),  
	       cudaGetErrorString( cudaGetLastError() ));
#endif 
        CUDA_SAFE_CALL_NO_SYNC(cudaFree(devBuffers[index])); 
        devBuffers[index] = NULL; 
      }
    }
    free(bufferInfo); 
  }
}

/* 
 * a switch statement defined by the user to allow the library to execute
 * the correct kernel 
 */ 
void kernelSelect(workRequest *wr);
#ifdef GPU_MEMPOOL
void createPool(int *nbuffers, int nslots, CkVec<BufferPool> &pools);
#endif

/* initHybridAPI
 *   initializes the work request queue, host/device buffer pointer
 *   arrays, and CUDA streams
 */
void initHybridAPI(int myPe) {

  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  cudaSetDevice(myPe % deviceCount); 

  initWRqueue(&wrQueue);

  /* allocate host/device buffers array (both user and
     system-addressed) */
  hostBuffers = (void **) malloc(NUM_BUFFERS * 2 * sizeof(void *)); 
  devBuffers = (void **) malloc(NUM_BUFFERS * 2 * sizeof(void *)); 

  /* initialize device array to NULL */ 
  for (int i=0; i<NUM_BUFFERS*2; i++) {
    devBuffers[i] = NULL; 
  }
  
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&kernel_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&data_in_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamCreate(&data_out_stream)); 

#ifdef GPU_PROFILE
  CUT_SAFE_CALL(cutCreateTimer(&timerHandle));
  CUT_SAFE_CALL(cutStartTimer(timerHandle));
#endif

  nextBuffer = NUM_BUFFERS;  

#ifdef GPU_TRACE
  traceRegisterUserEvent("GPU Memory Setup", GPU_MEM_SETUP);
  traceRegisterUserEvent("GPU Kernel Execution", GPU_KERNEL_EXEC);
  traceRegisterUserEvent("GPU Memory Cleanup", GPU_MEM_CLEANUP);
#endif

#ifdef GPU_MEMPOOL
  int nslots = GPU_MEMPOOL_NUM_SLOTS;
  int *sizes;
  sizes = (int *)malloc(sizeof(int)*nslots); 

  memPoolBoundaries.reserve(GPU_MEMPOOL_NUM_SLOTS);
  memPoolBoundaries.length() = GPU_MEMPOOL_NUM_SLOTS;

  int bufSize = GPU_MEMPOOL_MIN_BUFFER_SIZE;
  for(int i = 0; i < GPU_MEMPOOL_NUM_SLOTS; i++){
    memPoolBoundaries[i] = bufSize;
    bufSize = bufSize << 1;
  }

  //1K
  sizes[0] = 512; 
  //2K
  sizes[1] = 512;
  //4K
  sizes[2] = 64;
  //8K
  sizes[3] = 64;
  //16K
  sizes[4] = 32;
  //32K
  sizes[5] = 32;
  //64K
  sizes[6] = 32;
  //128K
  sizes[7] = 32;
  //256K
  sizes[8] = 32;
  //512K
  sizes[9] = 32;
  //1M
  sizes[10] = 170;
  //2M
  sizes[11] = 16;
  //4M
  sizes[12] = 4;
  //8M
  sizes[13] = 2;
  //16M
  sizes[14] = 2; 

  createPool(sizes, nslots, memPoolFreeBufs);
  printf("[%d] done creating buffer pool\n", CmiMyPe());

#endif

#ifdef GPU_INSTRUMENT_WRS
  initialized_instrument = false;
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
  if (isEmpty(wrQueue)) {
    flushPinnedMemQueue();    
    flushDelayedFrees();
    return;
  } 
  int returnVal; 
  workRequest *head = firstElement(wrQueue); 
  workRequest *second = secondElement(wrQueue);
  workRequest *third = thirdElement(wrQueue); 

  if (head->state == QUEUED) {
#ifdef GPU_PROFILE
    gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
    gpuEvents[timeIndex].eventType = DATA_SETUP; 
    gpuEvents[timeIndex].ID = head->id; 
    dataSetupIndex = timeIndex; 
#ifdef GPU_TRACE
    gpuEvents[timeIndex].stage = GPU_MEM_SETUP; 
    gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
    timeIndex++; 
#endif

#ifdef GPU_INSTRUMENT_WRS
    head->phaseStartTime = CmiWallTimer(); 
#endif

    allocateBuffers(head); 
    setupData(head); 
    head->state = TRANSFERRING_IN; 
  }  
  if (head->state == TRANSFERRING_IN) {
    if ((returnVal = cudaStreamQuery(data_in_stream)) == cudaSuccess) {
#ifdef GPU_PROFILE
      gpuEvents[dataSetupIndex].endTime = cutGetTimerValue(timerHandle);
#ifdef GPU_TRACE
      gpuEvents[dataSetupIndex].cmiendTime = CmiWallTimer();
      traceUserBracketEvent(gpuEvents[dataSetupIndex].stage, 
			    gpuEvents[dataSetupIndex].cmistartTime, 
			    gpuEvents[dataSetupIndex].cmiendTime); 
#endif 
#endif

#ifdef GPU_INSTRUMENT_WRS
      {
        if(initializedInstrument()){
          double tt = CmiWallTimer()-(head->phaseStartTime);
          int index = head->chareIndex;
          char type = head->compType;
          char phase = head->compPhase;

          CkVec<RequestTimeInfo> &vec = avgTimes[index][type];
          if(vec.length() <= phase){
            vec.growAtLeast(phase);
            vec.length() = phase+1;
          }
          vec[phase].transferTime += tt;
        }
      }
#endif

      if (second != NULL /*&& (second->state == QUEUED)*/) {
	allocateBuffers(second); 
      }
#ifdef GPU_PROFILE
      gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
      gpuEvents[timeIndex].eventType = KERNEL_EXECUTION; 
      gpuEvents[timeIndex].ID = head->id; 
      runningKernelIndex = timeIndex; 
#ifdef GPU_TRACE
      gpuEvents[timeIndex].stage = GPU_KERNEL_EXEC; 
      gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
      timeIndex++; 
#endif
#ifdef GPU_INSTRUMENT_WRS
      head->phaseStartTime = CmiWallTimer(); 
#endif

      //flushPinnedMemQueue();
      flushDelayedFrees();
      kernelSelect(head); 

      head->state = EXECUTING; 
      if (second != NULL) {
#ifdef GPU_PROFILE
	gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].eventType = DATA_SETUP; 
	gpuEvents[timeIndex].ID = second->id; 
	dataSetupIndex = timeIndex; 
#ifdef GPU_TRACE
	gpuEvents[timeIndex].stage = GPU_MEM_SETUP; 
	gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
	timeIndex++; 
#endif

#ifdef GPU_INSTRUMENT_WRS
        second->phaseStartTime = CmiWallTimer();
#endif
	setupData(second); 
	second->state = TRANSFERRING_IN;
      }
    }
      /*
#ifdef GPU_DEBUG
      printf("Querying memory stream returned: %d %.2f\n", returnVal, 
	     cutGetTimerValue(timerHandle));
#endif  
      */
  }
  if (head->state == EXECUTING) {
    if ((returnVal = cudaStreamQuery(kernel_stream)) == cudaSuccess) {
#ifdef GPU_PROFILE
      gpuEvents[runningKernelIndex].endTime = cutGetTimerValue(timerHandle); 
#ifdef GPU_TRACE
      gpuEvents[runningKernelIndex].cmiendTime = CmiWallTimer();
      traceUserBracketEvent(gpuEvents[runningKernelIndex].stage, 
			    gpuEvents[runningKernelIndex].cmistartTime, 
			    gpuEvents[runningKernelIndex].cmiendTime); 
#endif
#endif
#ifdef GPU_INSTRUMENT_WRS
      {
        if(initializedInstrument()){
          double tt = CmiWallTimer()-(head->phaseStartTime);
          int index = head->chareIndex;
          char type = head->compType;
          char phase = head->compPhase;

          CkVec<RequestTimeInfo> &vec = avgTimes[index][type];
          if(vec.length() <= phase){
            vec.growAtLeast(phase);
            vec.length() = phase+1;
          }
          vec[phase].kernelTime += tt;
        }
      }
#endif

      if (second != NULL && second->state == QUEUED) {
#ifdef GPU_PROFILE
	gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	gpuEvents[timeIndex].eventType = DATA_SETUP; 
	gpuEvents[timeIndex].ID = second->id; 
	dataSetupIndex = timeIndex; 
#ifdef GPU_TRACE
	gpuEvents[timeIndex].stage = GPU_MEM_SETUP; 
	gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
	timeIndex++; 
#endif

#ifdef GPU_INSTRUMENT_WRS
        second->phaseStartTime = CmiWallTimer();
#endif
        
	allocateBuffers(second); 
	setupData(second); 
	second->state = TRANSFERRING_IN; 	
      } 
      if (second != NULL && second->state == TRANSFERRING_IN) {
	if (cudaStreamQuery(data_in_stream) == cudaSuccess) {
#ifdef GPU_PROFILE
	  gpuEvents[dataSetupIndex].endTime = cutGetTimerValue(timerHandle); 
#ifdef GPU_TRACE
	  gpuEvents[dataSetupIndex].cmiendTime = CmiWallTimer();
	  traceUserBracketEvent(gpuEvents[dataSetupIndex].stage, 
				gpuEvents[dataSetupIndex].cmistartTime, 
				gpuEvents[dataSetupIndex].cmiendTime); 
#endif
#endif
#ifdef GPU_INSTRUMENT_WRS
          {
            if(initializedInstrument()){
              double tt = CmiWallTimer()-(second->phaseStartTime);
              int index = second->chareIndex;
              char type = second->compType;
              char phase = second->compPhase;

              CkVec<RequestTimeInfo> &vec = avgTimes[index][type];
              if(vec.length() <= phase){
                vec.growAtLeast(phase);
                vec.length() = phase+1;
              }
              vec[phase].transferTime += tt;
            }
          }
#endif

	  if (third != NULL /*&& (third->state == QUEUED)*/) {
	    allocateBuffers(third); 
	  }
#ifdef GPU_PROFILE
	  gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	  gpuEvents[timeIndex].eventType = KERNEL_EXECUTION; 
	  gpuEvents[timeIndex].ID = second->id; 
	  runningKernelIndex = timeIndex; 
#ifdef GPU_TRACE
	  gpuEvents[timeIndex].stage = GPU_KERNEL_EXEC; 
	  gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
	  timeIndex++; 
#endif
#ifdef GPU_INSTRUMENT_WRS
          second->phaseStartTime = CmiWallTimer();
#endif
	  //	    flushPinnedMemQueue();	    
          flushDelayedFrees();
	  kernelSelect(second); 
	  second->state = EXECUTING; 
	  if (third != NULL) {
#ifdef GPU_PROFILE
	    gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
	    gpuEvents[timeIndex].eventType = DATA_SETUP; 
	    gpuEvents[timeIndex].ID = third->id; 
	    dataSetupIndex = timeIndex; 
#ifdef GPU_TRACE
	    gpuEvents[timeIndex].stage = GPU_MEM_SETUP; 
	    gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
	    timeIndex++; 
#endif

#ifdef GPU_INSTRUMENT_WRS
            third->phaseStartTime = CmiWallTimer();
#endif
	    setupData(third); 
	    third->state = TRANSFERRING_IN; 	
	  }
	}
      }
#ifdef GPU_PROFILE
      gpuEvents[timeIndex].startTime = cutGetTimerValue(timerHandle); 
      gpuEvents[timeIndex].eventType = DATA_CLEANUP; 
      gpuEvents[timeIndex].ID = head->id; 
      dataCleanupIndex = timeIndex; 	
#ifdef GPU_TRACE
      gpuEvents[timeIndex].stage = GPU_MEM_CLEANUP; 
      gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
#endif
      timeIndex++; 
#endif
#ifdef GPU_INSTRUMENT_WRS
      head->phaseStartTime = CmiWallTimer(); 
#endif
      copybackData(head);
      head->state = TRANSFERRING_OUT;
    }
      /*
#ifdef GPU_DEBUG
      printf("Querying kernel completion returned: %d %.2f\n", returnVal,
	     cutGetTimerValue(timerHandle));
#endif  
      */
  }
  if (head->state == TRANSFERRING_OUT) {
    if (cudaStreamQuery(data_in_stream) == cudaSuccess &&
	cudaStreamQuery(data_out_stream) == cudaSuccess && 
	cudaStreamQuery(kernel_stream) == cudaSuccess){
      freeMemory(head); 
#ifdef GPU_PROFILE
      gpuEvents[dataCleanupIndex].endTime = cutGetTimerValue(timerHandle);
#ifdef GPU_TRACE
      gpuEvents[dataCleanupIndex].cmiendTime = CmiWallTimer();
      traceUserBracketEvent(gpuEvents[dataCleanupIndex].stage, 
			    gpuEvents[dataCleanupIndex].cmistartTime, 
			    gpuEvents[dataCleanupIndex].cmiendTime); 
#endif
#endif
#ifdef GPU_INSTRUMENT_WRS
      {
        if(initializedInstrument()){
          double tt = CmiWallTimer()-(head->phaseStartTime);
          int index = head->chareIndex;
          char type = head->compType;
          char phase = head->compPhase;

          CkVec<RequestTimeInfo> &vec = avgTimes[index][type];
          if(vec.length() <= phase){
            vec.growAtLeast(phase);
            vec.length() = phase+1;
          }
          vec[phase].cleanupTime += tt;
          vec[phase].n++;
        }
      }
#endif

      dequeue(wrQueue);
      CUDACallbackManager(head->callbackFn);
      gpuProgressFn(); 
    }
  }
}

#ifdef GPU_MEMPOOL
void releasePool(CkVec<BufferPool> &pools);
#endif
/* exitHybridAPI
 *  cleans up and deletes memory allocated for the queue and the CUDA streams
 */
void exitHybridAPI() {
  deleteWRqueue(wrQueue); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(kernel_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(data_in_stream)); 
  CUDA_SAFE_CALL_NO_SYNC(cudaStreamDestroy(data_out_stream)); 

#ifdef GPU_PROFILE
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
    printf(" %.2f:%.2f\n", gpuEvents[i].startTime-gpuEvents[0].startTime, gpuEvents[i].endTime-gpuEvents[0].startTime); 
  }

  CUT_SAFE_CALL(cutStopTimer(timerHandle));
  CUT_SAFE_CALL(cutDeleteTimer(timerHandle));  

#endif

#ifdef GPU_MEMPOOL
  releasePool(memPoolFreeBufs);
#endif

}

#ifdef GPU_MEMPOOL
void releasePool(CkVec<BufferPool> &pools){
  for(int i = 0; i < pools.length(); i++){
    CUDA_SAFE_CALL_NO_SYNC(cudaFreeHost((void *)pools[i].head));
  }
  pools.free();
}

// Create a pool with nslots slots.
// There are nbuffers[i] buffers for each buffer size corresponding to entry i
// FIXME - list the alignment/fragmentation issues with either of two allocation schemes:
// if a single, large buffer is allocated for each subpool
// if multiple smaller buffers are allocated for each subpool
void createPool(int *nbuffers, int nslots, CkVec<BufferPool> &pools){
  //pools  = (BufferPool *)malloc(nslots*sizeof(BufferPool));
  pools.reserve(nslots);
  pools.length() = nslots;

  for(int i = 0; i < nslots; i++){
    int bufSize = memPoolBoundaries[i];
    int numBuffers = nbuffers[i];
    pools[i].size = bufSize;
    
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)(&pools[i].head), 
                                          (sizeof(Header)+bufSize)*numBuffers));
    if(pools[i].head == NULL){
      abort();
    }

    Header *hd = pools[i].head;
    Header *previous = NULL;
    char *memory;

    for(int j = 0; j < numBuffers; j++){
      hd->slot = i;
      hd->next = previous;
      previous = hd;
      hd++; // move ptr past header
      memory = (char *)hd;
      memory += bufSize;
      hd = (Header *)memory;
    }

    pools[i].head = previous;
#ifdef GPU_MEMPOOL_DEBUG
    pools[i].num = numBuffers;
#endif
  }
}

int findPool(int size){
  int boundaryArrayLen = memPoolBoundaries.length();
  if(size <= memPoolBoundaries[0]){
    return (0);
  }
  else if(size > memPoolBoundaries[boundaryArrayLen-1]){
    // create new slot
    memPoolBoundaries.push_back(size);

    BufferPool newpool;
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost((void **)&newpool.head, size+sizeof(Header)));
    newpool.size = size;
#ifdef GPU_MEMPOOL_DEBUG
    newpool.num = 1;
#endif
    memPoolFreeBufs.push_back(newpool);

    Header *hd = newpool.head;
    hd->next = NULL;
    hd->slot = boundaryArrayLen;

    return boundaryArrayLen;
  }
  for(int i = 0; i < GPU_MEMPOOL_NUM_SLOTS-1; i++){
    if(memPoolBoundaries[i] < size && size <= memPoolBoundaries[i+1]){
      return (i+1);
    }
  }
  return -1;
}

void *getBufferFromPool(int pool, int size){
  Header *ret;
  if(pool < 0 || pool >= memPoolFreeBufs.length() || memPoolFreeBufs[pool].head == NULL){
#ifdef GPU_MEMPOOL_DEBUG
    printf("(%d) pool %d size: %d, num: %d\n", CmiMyPe(), pool, size, memPoolFreeBufs[pool].num);
#endif
    abort();
  }
  else{
    ret = memPoolFreeBufs[pool].head;
    memPoolFreeBufs[pool].head = ret->next;
#ifdef GPU_MEMPOOL_DEBUG
    ret->size = size;
    memPoolFreeBufs[pool].num--;
#endif
    return (void *)(ret+1);
  }
  return NULL;
}

void returnBufferToPool(int pool, Header *hd){
  hd->next = memPoolFreeBufs[pool].head;
  memPoolFreeBufs[pool].head = hd;
#ifdef GPU_MEMPOOL_DEBUG
  memPoolFreeBufs[pool].num++;
#endif
}

void *hapi_poolMalloc(int size){
  int pool = findPool(size);
  void *buf = getBufferFromPool(pool, size);
#ifdef GPU_MEMPOOL_DEBUG
  printf("(%d) hapi_malloc size %d pool %d left %d\n", CmiMyPe(), size, pool, memPoolFreeBufs[pool].num);
#endif
  return buf;
}

void hapi_poolFree(void *ptr){
  Header *hd = ((Header *)ptr)-1;
  int pool = hd->slot;
  returnBufferToPool(pool, hd);
#ifdef GPU_MEMPOOL_DEBUG
  int size = hd->size;
  printf("(%d) hapi_free size %d pool %d left %d\n", CmiMyPe(), size, pool, memPoolFreeBufs[pool].num);
#endif
}


#endif

#ifdef GPU_INSTRUMENT_WRS
void hapi_initInstrument(int numChares, char types){
  avgTimes.reserve(numChares);
  avgTimes.length() = numChares;
  for(int i = 0; i < numChares; i++){
    avgTimes[i].reserve(types);
    avgTimes[i].length() = types;
  }
  initialized_instrument = true;
}

bool initializedInstrument(){
  return initialized_instrument;
}

RequestTimeInfo *hapi_queryInstrument(int chare, char type, char phase){
  if(phase < avgTimes[chare][type].length()){
    return &avgTimes[chare][type][phase];
  }
  else{
    return NULL;
  }
}

void hapi_clearInstrument(){
  for(int chare = 0; chare < avgTimes.length(); chare++){
    for(int type = 0; type < avgTimes[chare].length(); type++){
      for(int phase = 0; phase < avgTimes[chare][type].length(); phase++){
        avgTimes[chare][type][phase].transferTime = 0.0;
        avgTimes[chare][type][phase].kernelTime = 0.0;
        avgTimes[chare][type][phase].cleanupTime = 0.0;
        avgTimes[chare][type][phase].n = 0;
      }
      avgTimes[chare][type].length() = 0;
    }
    avgTimes[chare].length() = 0;
  }
  avgTimes.length() = 0;
  initialized_instrument = false;
}

#endif
