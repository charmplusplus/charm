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
#include <stdio.h>
#include <stdlib.h>

#if defined GPU_MEMPOOL || defined GPU_INSTRUMENT_WRS
#include "cklists.h"
#endif

void cudaErrorDie(int err,const char *code,const char *file,int line)
{
  fprintf(stderr,"Fatal CUDA Error at %s:%d.\n"
	  " Return value %d from '%s'.  Exiting.\n",
	  file,line,
	  err,code);
  abort();
}

#define cudaChk(code)							\
  do { int e=(code); if (cudaSuccess!=e) {				\
      cudaErrorDie(e,#code,__FILE__,__LINE__); } } while (0)



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
//#define GPU_TRACE
//#define GPU_DEBUG
//#define _DEBUG

/* work request queue */
workRequestQueue *wrQueue = NULL; 

/* pending page-locked memory allocation requests */
int pinnedMemQueueIndex = 0;
pinnedMemReq pinnedMemQueue[MAX_PINNED_REQ];

int currentDfr = 0;
void *delayedFreeReqs[MAX_DELAYED_FREE_REQS];

#ifdef GPU_MEMPOOL
#define GPU_MEMPOOL_NUM_SLOTS 19
// pre-allocated buffers will be at least this big
#define GPU_MEMPOOL_MIN_BUFFER_SIZE 256

CkVec<BufferPool> memPoolFreeBufs;
CkVec<int> memPoolBoundaries;
//int memPoolBoundaries[GPU_MEMPOOL_NUM_SLOTS];

#ifdef GPU_DUMMY_MEMPOOL
CkVec<int> memPoolMax;
CkVec<int> memPoolSize;
#endif

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
int nextBuffer;

/* event types */
enum WorkRequestStage{
DataSetup        = 1,
KernelExecution  = 2,
DataCleanup      = 3
};


enum ProfilingStage{
GpuMemSetup   = 8800,
GpuKernelExec = 8801,
GpuMemCleanup = 8802
};

int runningKernelIndex = 0;
int dataSetupIndex = 0;
int dataCleanupIndex = 0;

#ifdef GPU_TRACE
typedef struct gpuEventTimer {
  int stage; 
  double cmistartTime; 
  double cmiendTime; 
  int eventType;
  int ID; 
} gpuEventTimer; 

gpuEventTimer gpuEvents[QUEUE_SIZE_INIT * 3]; 
int timeIndex = 0;

#if defined GPU_TRACE || defined GPU_INSTRUMENT_WRS
extern "C" double CmiWallTimer(); 
#endif

extern "C" int traceRegisterUserEvent(const char*x, int e);
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);
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
      cudaChk(cudaMallocHost((void **) reqs->hostPtrs[i], 
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
      cudaChk(cudaMallocHost((void **) req->hostPtrs[j], 
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

        cudaChk(cudaMalloc((void **) &devBuffers[index], size));
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
	cudaChk(cudaMalloc((void **) &devBuffers[index], size));
#ifdef GPU_DEBUG
	printf("buffer %d allocated %.2f\n", index,
	       cutGetTimerValue(timerHandle)); 
#endif
      }
      */
      
      if (bufferInfo[i].transferToDevice && size > 0) {
	cudaChk(cudaMemcpyAsync(devBuffers[index], 
          hostBuffers[index], size, cudaMemcpyHostToDevice, data_in_stream));
#ifdef GPU_DEBUG
	printf("transferToDevice bufId: %d at time %.2f size: %d " 
	       "error string: %s\n", index, cutGetTimerValue(timerHandle), 
	       size, cudaGetErrorString( cudaGetLastError() )); 
#endif	
	/*
	cudaChk(cudaMemcpy(devBuffers[index], 
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
	
	cudaChk(cudaMemcpyAsync(hostBuffers[index], 
          devBuffers[index], size, cudaMemcpyDeviceToHost,
          data_out_stream));
	
	/*
	cudaChk(cudaMemcpy(hostBuffers[index], 
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
        cudaChk(cudaFree(devBuffers[index])); 
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
  
  cudaChk(cudaStreamCreate(&kernel_stream)); 
  cudaChk(cudaStreamCreate(&data_in_stream)); 
  cudaChk(cudaStreamCreate(&data_out_stream)); 

  nextBuffer = NUM_BUFFERS;  

#ifdef GPU_TRACE
  traceRegisterUserEvent("GPU Memory Setup", GpuMemSetup);
  traceRegisterUserEvent("GPU Kernel Execution", GpuKernelExec);
  traceRegisterUserEvent("GPU Memory Cleanup", GpuMemCleanup);
#endif

#ifdef GPU_MEMPOOL

  int nslots = GPU_MEMPOOL_NUM_SLOTS;
  int sizes[GPU_MEMPOOL_NUM_SLOTS];

#ifdef GPU_DUMMY_MEMPOOL
  memPoolMax.reserve(nslots);
  memPoolMax.length() = nslots;
  memPoolSize.reserve(nslots);
  memPoolSize.length() = nslots;
#endif

  memPoolBoundaries.reserve(GPU_MEMPOOL_NUM_SLOTS);
  memPoolBoundaries.length() = GPU_MEMPOOL_NUM_SLOTS;

  int bufSize = GPU_MEMPOOL_MIN_BUFFER_SIZE;
  for(int i = 0; i < GPU_MEMPOOL_NUM_SLOTS; i++){
    memPoolBoundaries[i] = bufSize;
    bufSize = bufSize << 1;
#ifdef GPU_DUMMY_MEMPOOL
    memPoolSize[i] = 0;
    memPoolMax[i] = -1;
#endif
  }


#ifndef GPU_DUMMY_MEMPOOL
/*256*/ sizes[0] = 20;
/*512*/ sizes[1] = 10;
/*1024*/ sizes[2] = 10;
/*2048*/ sizes[3] = 20;
/*4096*/ sizes[4] = 10;
/*8192*/ sizes[5] = 30;
/*16384*/ sizes[6] = 25;
/*32768*/ sizes[7] = 10;
/*65536*/ sizes[8] = 5;
/*131072*/ sizes[9] = 5;
/*262144*/ sizes[10] = 5;
/*524288*/ sizes[11] = 5;
/*1048576*/ sizes[12] = 5;
/*2097152*/ sizes[13] = 10;
/*4194304*/ sizes[14] = 10;
/*8388608*/ sizes[15] = 10;
/*16777216*/ sizes[16] = 8;
/*33554432*/ sizes[17] = 6;
/*67108864*/ sizes[18] = 7;

createPool(sizes, nslots, memPoolFreeBufs);
#endif

  printf("[%d] done creating buffer pool\n", CmiMyPe());

#endif

#ifdef GPU_INSTRUMENT_WRS
  initialized_instrument = false;
#endif
}

inline void gpuEventStart(workRequest *wr, int *index, WorkRequestStage event, ProfilingStage stage){
#ifdef GPU_TRACE
  gpuEvents[timeIndex].cmistartTime = CmiWallTimer();
  gpuEvents[timeIndex].eventType = event;
  gpuEvents[timeIndex].ID = wr->id;
  *index = timeIndex;
  gpuEvents[timeIndex].stage = stage;
  timeIndex++;
#endif
}

inline void gpuEventEnd(int index){
#ifdef GPU_TRACE
  gpuEvents[index].cmiendTime = CmiWallTimer();
  traceUserBracketEvent(gpuEvents[index].stage, gpuEvents[index].cmistartTime, gpuEvents[index].cmiendTime);
#endif
}

inline void workRequestStartTime(workRequest *wr){
#ifdef GPU_INSTRUMENT_WRS
  wr->phaseStartTime = CmiWallTimer();
#endif
}

inline void gpuProfiling(workRequest *wr, WorkRequestStage event){
#ifdef GPU_INSTRUMENT_WRS
  if(initializedInstrument()){
    double tt = CmiWallTimer()-(wr->phaseStartTime);
    int index = wr->chareIndex;
    char type = wr->compType;
    char phase = wr->compPhase;

    CkVec<RequestTimeInfo> &vec = avgTimes[index][type];
    if(vec.length() <= phase){
      vec.growAtLeast(phase);
      vec.length() = phase+1;
    }
    switch(event){
      case DataSetup:
        vec[phase].transferTime += tt;
        break;
      case KernelExecution:
        vec[phase].kernelTime += tt;
        break;
      case DataCleanup:
        vec[phase].cleanupTime += tt;
        vec[phase].n++;
        break;
      default:
        printf("Error: Invalid event during gpuProfiling\n");
    }
  }
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
    gpuEventStart(head,&dataSetupIndex,DataSetup,GpuMemSetup);
    workRequestStartTime(head);
    allocateBuffers(head); 
    setupData(head); 
    head->state = TRANSFERRING_IN; 
  }  
  if (head->state == TRANSFERRING_IN) {
    if ((returnVal = cudaStreamQuery(data_in_stream)) == cudaSuccess) {
      gpuEventEnd(dataSetupIndex);
      gpuProfiling(head,DataSetup);
      if (second != NULL /*&& (second->state == QUEUED)*/) {
	allocateBuffers(second); 
      }
      gpuEventStart(head,&runningKernelIndex,KernelExecution,GpuKernelExec);
      workRequestStartTime(head);
      //flushPinnedMemQueue();
      flushDelayedFrees();
      kernelSelect(head); 

      head->state = EXECUTING; 
      if (second != NULL) {
        gpuEventStart(second,&dataSetupIndex,DataSetup,GpuMemSetup);
        workRequestStartTime(second);
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
      gpuEventEnd(runningKernelIndex);
      gpuProfiling(head,KernelExecution);
      if (second != NULL && second->state == QUEUED) {
        gpuEventStart(second,&dataSetupIndex,DataSetup,GpuMemSetup);
        workRequestStartTime(second);
	allocateBuffers(second); 
	setupData(second); 
	second->state = TRANSFERRING_IN; 	
      } 
      if (second != NULL && second->state == TRANSFERRING_IN) {
	if (cudaStreamQuery(data_in_stream) == cudaSuccess) {
          gpuEventEnd(dataSetupIndex);
          gpuProfiling(second,DataSetup);
	  if (third != NULL /*&& (third->state == QUEUED)*/) {
	    allocateBuffers(third); 
	  }
          gpuEventStart(second,&runningKernelIndex,KernelExecution,GpuKernelExec);
          workRequestStartTime(second);
	  //	    flushPinnedMemQueue();	    
          flushDelayedFrees();
	  kernelSelect(second); 
	  second->state = EXECUTING; 
	  if (third != NULL) {
            gpuEventStart(third,&dataSetupIndex,DataSetup,GpuMemSetup);
            workRequestStartTime(third);
	    setupData(third); 
	    third->state = TRANSFERRING_IN; 	
	  }
	}
      }
      gpuEventStart(head,&dataSetupIndex,DataSetup,GpuMemSetup);
      workRequestStartTime(head);
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
      gpuEventEnd(dataCleanupIndex);
      gpuProfiling(head,DataCleanup);
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
  printf("EXIT HYBRID API\n");

#ifdef GPU_MEMPOOL

#ifndef GPU_DUMMY_MEMPOOL
  releasePool(memPoolFreeBufs);
#else
  for(int i = 0; i < memPoolBoundaries.length(); i++){
    printf("(%d) slot %d size: %d max: %d\n", CmiMyPe(), i, memPoolBoundaries[i], memPoolMax[i]);
  }

  if(memPoolBoundaries.length() != memPoolMax.length()){
    abort();
  }
#endif
  
#endif

  deleteWRqueue(wrQueue); 
  cudaChk(cudaStreamDestroy(kernel_stream)); 
  cudaChk(cudaStreamDestroy(data_in_stream)); 
  cudaChk(cudaStreamDestroy(data_out_stream)); 

#ifdef GPU_TRACE
  for (int i=0; i<timeIndex; i++) {
    switch (gpuEvents[i].eventType) {
    case DataSetup:
      printf("Kernel %d data setup", gpuEvents[i].ID); 
      break;
    case DataCleanup:
      printf("Kernel %d data cleanup", gpuEvents[i].ID); 
      break; 
    case KernelExecution:
      printf("Kernel %d execution", gpuEvents[i].ID); 
      break;
    default:
      printf("Error, invalid timer identifier\n"); 
    }
    printf(" %.2f:%.2f\n", gpuEvents[i].cmistartTime-gpuEvents[0].cmistartTime, gpuEvents[i].cmiendTime-gpuEvents[0].cmistartTime); 
  }

#endif

}

#ifdef GPU_MEMPOOL
void releasePool(CkVec<BufferPool> &pools){
  for(int i = 0; i < pools.length(); i++){
    Header *hdr;
    Header *next;
    for(hdr = pools[i].head; hdr != NULL;){
      next = hdr->next; 
      cudaChk(cudaFreeHost((void *)hdr));
      hdr = next;
    }
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
    pools[i].head = NULL;
    
    /*
    cudaChk(cudaMallocHost((void **)(&pools[i].head), 
                                          (sizeof(Header)+bufSize)*numBuffers));
    */

    Header *hd = pools[i].head;
    Header *previous = NULL;

    for(int j = 0; j < numBuffers; j++){
      cudaChk(cudaMallocHost((void **)&hd,
                                            (sizeof(Header)+bufSize)));
      if(hd == NULL){
        printf("(%d) failed to allocate %dth block of size %d, slot %d\n", CmiMyPe(), j, bufSize, i);
        abort();
      }
      hd->slot = i;
      hd->next = previous;
      previous = hd;
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
#ifdef GPU_DUMMY_MEMPOOL
    memPoolMax.push_back(-1);
    memPoolSize.push_back(0);
#endif

    BufferPool newpool;
    cudaChk(cudaMallocHost((void **)&newpool.head, size+sizeof(Header)));
    if(newpool.head == NULL){
      printf("(%d) findPool: failed to allocate newpool %d head, size %d\n", CmiMyPe(), boundaryArrayLen, size);
      abort();
    }
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
  for(int i = 0; i < memPoolBoundaries.length()-1; i++){
    if(memPoolBoundaries[i] < size && size <= memPoolBoundaries[i+1]){
      return (i+1);
    }
  }
  return -1;
}

void *getBufferFromPool(int pool, int size){
  Header *ret;
  if(pool < 0 || pool >= memPoolFreeBufs.length()){
    printf("(%d) getBufferFromPool, pool: %d, size: %d invalid pool\n", CmiMyPe(), pool, size);
#ifdef GPU_MEMPOOL_DEBUG
    printf("(%d) num: %d\n", CmiMyPe(), memPoolFreeBufs[pool].num);
#endif
    abort();
  }
  else if (memPoolFreeBufs[pool].head == NULL){
    Header *hd;
    cudaChk(cudaMallocHost((void **)&hd, sizeof(Header)+memPoolFreeBufs[pool].size));
#ifdef GPU_MEMPOOL_DEBUG
    printf("(%d) getBufferFromPool, pool: %d, size: %d expand by 1\n", CmiMyPe(), pool, size);
#endif
    if(hd == NULL){
      abort();
    }
    hd->slot = pool;
    return (void *)(hd+1);
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
  void *buf;
#ifdef GPU_DUMMY_MEMPOOL
  cudaChk(cudaMallocHost((void **)&buf, size+sizeof(Header)));
  if(pool < 0 || pool >= memPoolSize.length()){
    printf("(%d) need to create up to pool %d; malloc size: %d\n", CmiMyPe(), pool, size);
    abort();
  }
  memPoolSize[pool]++;
  if(memPoolSize[pool] > memPoolMax[pool]){
    memPoolMax[pool] = memPoolSize[pool];
  }
  Header *hdr = (Header *)buf;
  hdr->slot = pool;
  hdr = hdr+1;
  buf = (void *)hdr;
#else
  buf = getBufferFromPool(pool, size);
#endif
  
#ifdef GPU_MEMPOOL_DEBUG
  printf("(%d) hapi_malloc size %d pool %d left %d\n", CmiMyPe(), size, pool, memPoolFreeBufs[pool].num);
#endif
  return buf;
}

void hapi_poolFree(void *ptr){
  Header *hd = ((Header *)ptr)-1;
  int pool = hd->slot;

#ifdef GPU_DUMMY_MEMPOOL
  if(pool < 0 || pool >= memPoolSize.length()){
    printf("(%d) free pool %d\n", CmiMyPe(), pool);
    abort();
  }
  memPoolSize[pool]--;
  delayedFree((void *)hd); 
#else
  returnBufferToPool(pool, hd);
#endif

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
