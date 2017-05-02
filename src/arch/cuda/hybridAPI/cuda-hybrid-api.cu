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
#include "converse.h"
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

/*
 * Important Flags:
 *
 * GPU_MEMPOOL
 * Defined in the default Makefile, this flag instructs the code to
 *  allocate a shared pool of pinned memory in advance to draw
 *  allocations from
 *
 * GPU_TRACE
 * When compiling this file, define the flag GPU_TRACE (i.e. -DGPU_TRACE)
 *  to tell the system to record the time for invocation and
 *  completion of GPU events: memory allocation, transfer and
 *  kernel execution
 *
 * GPU_DEBUG
 * Similarly, define the GPU_DEBUG flag to output more verbose debugging
 *  information during execution
 *
 * GPU_MEMPOOL_DEBUG
 * As above but for information regarding the workings of the mempool
 *  operations
 *
 * GPU_INSTRUMENT_WRS
 * Turn this flag on during compilation to enable recording of
 *  of Work Request Start (WRS) and end times. This includes
 *  time spent in each phase (data in, exec, and data out) and other
 *  relevant data such as the chare that launched the kernel etc
 */

#if defined GPU_MEMPOOL || defined GPU_INSTRUMENT_WRS
#include "cklists.h"
#endif

#define cudaChk(code) cudaErrorDie(code, #code, __FILE__, __LINE__)
inline void cudaErrorDie(cudaError_t retCode, const char* code,
                                              const char* file, int line) {
  if (retCode != cudaSuccess) {
    fprintf(stderr, "Fatal CUDA Error %s at %s:%d.\nReturn value %d from '%s'.",
        cudaGetErrorString(retCode), file, line, retCode, code);
    CmiAbort(" Exiting!\n");
  }
}

#if defined GPU_TRACE || defined GPU_INSTRUMENT_WRS
extern "C" double CmiWallTimer();
#endif

#if defined GPU_TRACE
extern "C" int traceRegisterUserEvent(const char*x, int e);
extern "C" void traceUserBracketEvent(int e, double beginT, double endT);
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

#ifdef GPU_TRACE
typedef struct gpuEventTimer {
  int stage;
  double cmistartTime;
  double cmiendTime;
  int eventType;
  unsigned char *traceName;
} gpuEventTimer;
#endif // GPU_TRACE

class GPUManager {

public:

  workRequestQueue *wrQueue;

/* pending page-locked memory allocation requests */
  int pinnedMemQueueIndex;
  pinnedMemReq pinnedMemQueue[MAX_PINNED_REQ];

  int currentDfr;
  void *delayedFreeReqs[MAX_DELAYED_FREE_REQS];

#ifdef GPU_MEMPOOL
#define GPU_MEMPOOL_NUM_SLOTS 20 // Update for new row, again this shouldn't be hard coded!
// pre-allocated buffers will be at least this big
#define GPU_MEMPOOL_MIN_BUFFER_SIZE 256
// Scale the amount of memory each node pins
#define GPU_MEMPOOL_SCALE 1.0
// Largest number of bytes that will initially be allocated for the GPU Mempool
#define GPU_MEMPOOL_MAX_INIT_SIZE static_cast<size_t>(1 << 30) // 1 << 30 = 1073741824 bytes = 1 GiB

  CkVec<BufferPool> memPoolFreeBufs;
  CkVec<size_t> memPoolBoundaries;

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
  void **hostBuffers;

  /* device buffers */
  void **devBuffers;

  /* used to assign bufferIDs automatically by the system if the user
   specifies an invalid bufferID */
  int nextBuffer;

  /* There are separate CUDA streams for kernel execution, data transfer
   * into the device, and data transfer out. This allows prefetching of
   * data for a subsequent kernel while the previous kernel is
   * executing and transferring data out of the device.
   */
  cudaStream_t kernel_stream;
  cudaStream_t data_in_stream;
  cudaStream_t data_out_stream;

  int runningKernelIndex;
  int dataSetupIndex;
  int dataCleanupIndex;

#ifdef GPU_TRACE
  gpuEventTimer gpuEvents[QUEUE_SIZE_INIT * 3];
  int timeIndex;
#endif

#ifdef GPU_INSTRUMENT_WRS
  CkVec<CkVec<CkVec<RequestTimeInfo> > > avgTimes;
  bool initialized_instrument;
#endif
  CmiNodeLock bufferlock;
  CmiNodeLock queuelock;
  CmiNodeLock progresslock;
  CmiNodeLock pinlock;
  CmiNodeLock dfrlock;

  void initHybridAPIHelper();
  void gpuProgressFnHelper();
  GPUManager(){
    wrQueue = NULL;
    devBuffers = NULL;
    hostBuffers = NULL;
    wrqueue::initWRqueue(&wrQueue);
    nextBuffer = NUM_BUFFERS;
    pinnedMemQueueIndex = 0;
    currentDfr = 0;
    runningKernelIndex = 0;
    dataSetupIndex = 0;
    dataCleanupIndex = 0;
    bufferlock = CmiCreateLock();
    queuelock = CmiCreateLock();
    progresslock = CmiCreateLock();
    pinlock   = CmiCreateLock();
    dfrlock   = CmiCreateLock();
#ifdef GPU_TRACE
    timeIndex = 0;
#endif
    initHybridAPIHelper();
  }

  void runKernel(workRequest *wr);
};
CsvDeclare(GPUManager, gpuManager);

#ifdef GPU_INSTRUMENT_WRS
bool initializedInstrument();
#endif

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

void enqueue(workRequest* wr) {
  CmiLock(CsvAccess(gpuManager).queuelock);
#ifdef GPU_DEBUG
  printf("wrQueue Size = %d", CsvAccess(gpuManager).wrQueue->size);
#endif
  wrqueue::enqueue(CsvAccess(gpuManager).wrQueue, wr);
  CmiUnlock(CsvAccess(gpuManager).queuelock);
}

// Keep API consistent but use shared queue
void enqueue(workRequestQueue* q, workRequest* wr) {
  enqueue(wr);
}

cudaStream_t getKernelStream() {
  return CsvAccess(gpuManager).kernel_stream;
}

void** gethostBuffers() {
  return CsvAccess(gpuManager).hostBuffers;
}

void** getdevBuffers() {
  return CsvAccess(gpuManager).devBuffers;
}

int getNextPinnedIndex() {
  CmiLock(CsvAccess(gpuManager).pinlock);
  int pinnedIndex = CsvAccess(gpuManager).pinnedMemQueueIndex++;
  CmiUnlock(CsvAccess(gpuManager).pinlock);
  if (pinnedIndex == MAX_PINNED_REQ) {
    printf("Error: pinned memory request buffer is overflowing\n");
  }
  return pinnedIndex;
}

int getNextDelayedFreeReqIndex() {
  CmiLock(CsvAccess(gpuManager).dfrlock);
  int dfrIndex = CsvAccess(gpuManager).currentDfr;
  int flag = ((dfrIndex == MAX_DELAYED_FREE_REQS) ? 1 : 0);
  CsvAccess(gpuManager).currentDfr++;
  CmiUnlock(CsvAccess(gpuManager).dfrlock);
  if (flag) {
    CmiAbort("Ran out of DFR queue space. Increase MAX_DELAYED_FREE_REQS. Exiting!\n");
  }
  return dfrIndex;
}

/* pinnedMallocHost
 *
 * schedules a pinned memory allocation so that it does not impede
 * concurrent asynchronous execution
 *
 */
void pinnedMallocHost(pinnedMemReq* reqs) {
  if ( (cudaStreamQuery(CsvAccess(gpuManager).kernel_stream) == cudaSuccess) &&
       (cudaStreamQuery(CsvAccess(gpuManager).data_in_stream) == cudaSuccess) &&
       (cudaStreamQuery(CsvAccess(gpuManager).data_out_stream) == cudaSuccess) ) {
    for (int i = 0; i < reqs->nBuffers; i++) {
      cudaChk(cudaMallocHost((void **) reqs->hostPtrs[i], reqs->sizes[i]));
    }

    free(reqs->hostPtrs);
    free(reqs->sizes);

    CUDACallbackManager(reqs->callbackFn);
    gpuProgressFn();
  }
  else {
    CsvAccess(gpuManager).pinnedMemQueue[getNextPinnedIndex()] = *reqs;
  }
}

void delayedFree(void* ptr){
  CsvAccess(gpuManager).delayedFreeReqs[getNextDelayedFreeReqIndex()] = ptr;
}

void flushDelayedFrees(){
  CmiLock(CsvAccess(gpuManager).dfrlock);
  for(int i = 0; i < CsvAccess(gpuManager).currentDfr; i++){
    if(CsvAccess(gpuManager).delayedFreeReqs[i] == NULL){
      CmiAbort("Encountered NULL pointer in delayedFree(). Exiting!\n");
    }
    cudaChk(cudaFreeHost(CsvAccess(gpuManager).delayedFreeReqs[i]));
  }
  CsvAccess(gpuManager).currentDfr = 0;
  CmiUnlock(CsvAccess(gpuManager).dfrlock);
}

/* flushPinnedMemQueue
 *
 * executes pending pinned memory allocation requests
 *
 */
void flushPinnedMemQueue() {
  CmiLock(CsvAccess(gpuManager).pinlock);
  for (int i = 0; i < CsvAccess(gpuManager).pinnedMemQueueIndex; i++) {
    pinnedMemReq *req = &CsvAccess(gpuManager).pinnedMemQueue[i];

    for (int j = 0; j < req->nBuffers; j++) {
      cudaChk(cudaMallocHost((void **) req->hostPtrs[j], req->sizes[j]));
    }
    free(req->hostPtrs);
    free(req->sizes);
    CUDACallbackManager(CsvAccess(gpuManager).pinnedMemQueue[i].callbackFn);
  }
  CsvAccess(gpuManager).pinnedMemQueueIndex = 0;
  CmiUnlock(CsvAccess(gpuManager).pinlock);
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
  dataInfo* bufferInfo = wr->bufferInfo;

  if (bufferInfo != NULL) {
    for (int i = 0; i < wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID;
      int size = bufferInfo[i].size;

      // if index value is invalid, use an available ID
      if (index < 0 || index >= NUM_BUFFERS) {
        bool isFound = false;
        for (int j = CsvAccess(gpuManager).nextBuffer; j < NUM_BUFFERS * 2; j++) {
          if (CsvAccess(gpuManager).devBuffers[j] == NULL) {
            index = j;
            isFound = true;
            break;
          }
        }

        // if no index was found, try to search for a value at the
        // beginning of the system addressed space
        if (!isFound) {
          for (int j = NUM_BUFFERS; j < CsvAccess(gpuManager).nextBuffer; j++) {
            if (CsvAccess(gpuManager).devBuffers[j] == NULL) {
              index = j;
              isFound = true;
              break;
            }
          }
        }

        if (!isFound) {
          printf("Error: devBuffers is full \n");
        }

        CsvAccess(gpuManager).nextBuffer = index+1;
        if (CsvAccess(gpuManager).nextBuffer == NUM_BUFFERS * 2) {
          CsvAccess(gpuManager).nextBuffer = NUM_BUFFERS;
        }

        bufferInfo[i].bufferID = index;
      }

      if (CsvAccess(gpuManager).devBuffers[index] == NULL && size > 0) {
#ifdef GPU_PRINT_BUFFER_ALLOCATE
        double mil = 1e3;
        printf("*** ALLOCATE buffer 0x%x (%d) size %f kb\n", CsvAccess(gpuManager).devBuffers[index], index, size / mil);
#endif

        cudaChk(cudaMalloc((void **) &CsvAccess(gpuManager).devBuffers[index], size));

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
  dataInfo* bufferInfo = wr->bufferInfo;

  if (bufferInfo != NULL) {
    for (int i = 0; i < wr->nBuffers; i++) {
      int index = bufferInfo[i].bufferID;
      int size = bufferInfo[i].size;
      CsvAccess(gpuManager).hostBuffers[index] = bufferInfo[i].hostBuffer;

      if (bufferInfo[i].transferToDevice && size > 0) {
        cudaChk(cudaMemcpyAsync(CsvAccess(gpuManager).devBuffers[index],
                CsvAccess(gpuManager).hostBuffers[index], size,
                cudaMemcpyHostToDevice, CsvAccess(gpuManager).data_in_stream));
#ifdef GPU_DEBUG
        printf("transferToDevice bufId: %d at time %.2f size: %d "
               "error string: %s\n", index, cutGetTimerValue(timerHandle),
               size, cudaGetErrorString(cudaGetLastError()));
#endif
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

    for (int i = 0; i < nBuffers; i++) {
      int index = bufferInfo[i].bufferID;
      int size = bufferInfo[i].size;

      if (bufferInfo[i].transferFromDevice && size > 0) {
#ifdef GPU_DEBUG
        printf("transferFromDevice: %d at time %.2f size: %d "
            "error string: %s\n", index, cutGetTimerValue(timerHandle),
            size, cudaGetErrorString(cudaGetLastError()));
#endif

        cudaChk(cudaMemcpyAsync(CsvAccess(gpuManager).hostBuffers[index],
              CsvAccess(gpuManager).devBuffers[index], size, cudaMemcpyDeviceToHost,
              CsvAccess(gpuManager).data_out_stream));
      }
    }
  }
}

/* frees GPU memory for buffers specified by the user; also frees the
 *  work request's bufferInfo array
 */
void freeMemory(workRequest *wr) {
  dataInfo* bufferInfo = wr->bufferInfo;
  int nBuffers = wr->nBuffers;
  if (bufferInfo != NULL) {
    for (int i = 0; i < nBuffers; i++) {
      int index = bufferInfo[i].bufferID;
      if (bufferInfo[i].freeBuffer) {
#ifdef GPU_PRINT_BUFFER_ALLOCATE
        printf("*** FREE buffer 0x%x (%d)\n", CsvAccess(gpuManager).devBuffers[index], index);
#endif

#ifdef GPU_DEBUG
        printf("buffer %d freed at time %.2f error string: %s\n",
            index, cutGetTimerValue(timerHandle),
            cudaGetErrorString(cudaGetLastError()));
#endif
        cudaChk(cudaFree(CsvAccess(gpuManager).devBuffers[index]));
        CsvAccess(gpuManager).devBuffers[index] = NULL;
      }
    }
    free(bufferInfo);
  }
}

/*
 * Run the user's kernel for this work request.
 *
 * WAS: a switch statement defined by the user to allow the library to execute
 * the correct kernel
 */
void GPUManager::runKernel(workRequest *wr) {
	if (wr->runKernel) {
		wr->runKernel(wr,kernel_stream,devBuffers);
	}
	// else, might be only for data transfer (or might be a bug?)
}

#ifdef GPU_MEMPOOL
void createPool(int *nbuffers, int nslots, CkVec<BufferPool> &pools);
#endif // GPU_MEMPOOL

void initHybridAPI() {
  CsvInitialize(GPUManager, gpuManager);

#ifdef GPU_MEMPOOL

#ifndef GPU_DUMMY_MEMPOOL
  int sizes[GPU_MEMPOOL_NUM_SLOTS];
        /*256*/ sizes[0]  =  4;
        /*512*/ sizes[1]  =  2;
       /*1024*/ sizes[2]  =  2;
       /*2048*/ sizes[3]  =  4;
       /*4096*/ sizes[4]  =  2;
       /*8192*/ sizes[5]  =  6;
      /*16384*/ sizes[6]  =  5;
      /*32768*/ sizes[7]  =  2;
      /*65536*/ sizes[8]  =  1;
     /*131072*/ sizes[9]  =  1;
     /*262144*/ sizes[10] =  1;
     /*524288*/ sizes[11] =  1;
    /*1048576*/ sizes[12] =  1;
    /*2097152*/ sizes[13] =  2;
    /*4194304*/ sizes[14] =  2;
    /*8388608*/ sizes[15] =  2;
   /*16777216*/ sizes[16] =  2;
   /*33554432*/ sizes[17] =  1;
   /*67108864*/ sizes[18] =  1;
  /*134217728*/ sizes[19] =  7;
  createPool(sizes, GPU_MEMPOOL_NUM_SLOTS, CsvAccess(gpuManager).memPoolFreeBufs);

#ifdef GPU_MEMPOOL_DEBUG
  printf("[%d] done creating buffer pool\n", CmiMyPe());
#endif

#endif // GPU_DUMMY_MEMPOOL

#endif // GPU_MEMPOOL
}

inline int getMyCudaDevice(int myPe) {
  int deviceCount;
  cudaChk(cudaGetDeviceCount(&deviceCount));
  return myPe % deviceCount;
}

/* initHybridAPIHelper
 * Moved old initHybridAPI inside the helper function
 * This function is called from GPU Manager's constructor to set the member variables
 * initializes the work request queue, host/device buffer pointer
 * arrays, and CUDA streams
 */
void GPUManager::initHybridAPIHelper() {
  cudaChk(cudaSetDevice(getMyCudaDevice(CmiMyPe())));

  /* allocate host/device buffers array (both user and
     system-addressed) */
  hostBuffers = (void **)malloc(NUM_BUFFERS * 2 * sizeof(void *));
  devBuffers  = (void **)malloc(NUM_BUFFERS * 2 * sizeof(void *));

  /* initialize device array to NULL */
  for (int i = 0; i < NUM_BUFFERS * 2; i++) {
    devBuffers[i] = NULL;
  }

  cudaChk(cudaStreamCreate(&kernel_stream));
  cudaChk(cudaStreamCreate(&data_in_stream));
  cudaChk(cudaStreamCreate(&data_out_stream));


#ifdef GPU_TRACE
  traceRegisterUserEvent("GPU Memory Setup", GpuMemSetup);
  traceRegisterUserEvent("GPU Kernel Execution", GpuKernelExec);
  traceRegisterUserEvent("GPU Memory Cleanup", GpuMemCleanup);
#endif

#ifdef GPU_MEMPOOL
  int nslots = GPU_MEMPOOL_NUM_SLOTS;

#ifdef GPU_DUMMY_MEMPOOL
  memPoolMax.reserve(nslots);
  memPoolMax.length() = nslots;
  memPoolSize.reserve(nslots);
  memPoolSize.length() = nslots;
#endif

  memPoolBoundaries.reserve(GPU_MEMPOOL_NUM_SLOTS);
  memPoolBoundaries.length() = GPU_MEMPOOL_NUM_SLOTS;

  size_t bufSize = GPU_MEMPOOL_MIN_BUFFER_SIZE;
  for(int i = 0; i < GPU_MEMPOOL_NUM_SLOTS; i++){
    memPoolBoundaries[i] = bufSize;
    bufSize = bufSize << 1;
#ifdef GPU_DUMMY_MEMPOOL
    memPoolSize[i] =  0;
    memPoolMax[i]  = -1;
#endif
  }

#endif // GPU_MEMPOOL

#ifdef GPU_INSTRUMENT_WRS
  initialized_instrument = false;
#endif
}

void gpuProgressFn(){
  CmiLock(CsvAccess(gpuManager).progresslock);
  CsvAccess(gpuManager).gpuProgressFnHelper();
  CmiUnlock(CsvAccess(gpuManager).progresslock);
}

inline void gpuEventStart(workRequest *wr, int *index, WorkRequestStage event, ProfilingStage stage){
#ifdef GPU_TRACE
  gpuEventTimer* shared_gpuEvents = CsvAccess(gpuManager).gpuEvents;
  int shared_timeIndex = CsvAccess(gpuManager).timeIndex++;
  shared_gpuEvents[shared_timeIndex].cmistartTime = CmiWallTimer();
  shared_gpuEvents[shared_timeIndex].eventType = event;
  shared_gpuEvents[shared_timeIndex].traceName = wr->traceName;
  *index = shared_timeIndex;
  shared_gpuEvents[shared_timeIndex].stage = stage;
#ifdef GPU_DEBUG
  printf("Start Event Name = %d \t Stage Name=%d workRequest Id= %d\n", event,stage, wr->id);
#endif
#endif // GPU_TRACE
}

inline void gpuEventEnd(int index){
#ifdef GPU_TRACE
  CsvAccess(gpuManager).gpuEvents[index].cmiendTime = CmiWallTimer();
  traceUserBracketEvent(CsvAccess(gpuManager).gpuEvents[index].stage, CsvAccess(gpuManager).gpuEvents[index].cmistartTime, CsvAccess(gpuManager).gpuEvents[index].cmiendTime);
#ifdef GPU_DEBUG
  printf("End Event Name = %d\t Stage Name=%d workRequest Id=%s\n",CsvAccess(gpuManager).gpuEvents[index].eventType, CsvAccess(gpuManager).gpuEvents[index].stage, CsvAccess(gpuManager).gpuEvents[index].traceName);
#endif
#endif
}

inline void workRequestStartTime(workRequest *wr){
#ifdef GPU_INSTRUMENT_WRS
  wr->phaseStartTime = CmiWallTimer();
#endif
}

inline void profileWorkRequestEvent(workRequest* wr, WorkRequestStage event){
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
        printf("Error: Invalid event during profileWorkRequestEvent\n");
    }
  }
#endif
}

/* gpuProgressFnHelper
 *  This function is called after achieving the queuelock.
 *  Old gpuProgressFn now becomes a wrapper for gpuProgressFnHelper for
 *  exclusive access to threads.
 *  Called periodically to monitor work request progress, and perform
 *  the prefetch of data for a subsequent work request.
 */
void GPUManager::gpuProgressFnHelper() {
  bool isHeadFinished = true;
  int returnVal = -1;
  while (isHeadFinished) {
    isHeadFinished = false;

    if (wrQueue == NULL) {
      printf("Error: work request queue not initialized\n");
      break;
    }

    if (wrqueue::isEmpty(wrQueue)) {
      flushPinnedMemQueue();
      flushDelayedFrees();
      break;
    }

    CmiLock(CsvAccess(gpuManager).queuelock);
    workRequest *head   = wrqueue::firstElement(wrQueue);
    workRequest *second = wrqueue::secondElement(wrQueue);
    workRequest *third  = wrqueue::thirdElement(wrQueue);

    if (head->state == QUEUED) {
#ifdef GPU_DEBUG
      printf("wrQueue Size = %d head.state = QUEUED", wrQueue->size);
#endif
      gpuEventStart(head, &dataSetupIndex, DataSetup, GpuMemSetup);
      workRequestStartTime(head);
      allocateBuffers(head);
      setupData(head);
      head->state = TRANSFERRING_IN;
    }
    if (head->state == TRANSFERRING_IN) {
#ifdef GPU_DEBUG
      printf("wrQueue Size = %d head.state = Transferring", wrQueue->size);
#endif
      if ((returnVal = cudaStreamQuery(data_in_stream)) == cudaSuccess) {
        gpuEventEnd(dataSetupIndex);
        profileWorkRequestEvent(head, DataSetup);
        if (second != NULL) {
          allocateBuffers(second);
        }
        gpuEventStart(head, &runningKernelIndex, KernelExecution, GpuKernelExec);
        workRequestStartTime(head);
        flushDelayedFrees();
        runKernel(head);

        head->state = EXECUTING;
        if (second != NULL) {
          gpuEventStart(second, &dataSetupIndex, DataSetup, GpuMemSetup);
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
#ifdef GPU_DEBUG
      printf("wrQueue Size = %d head.state = EXECUTING", CsvAccess(gpuManager).wrQueue->size);
#endif
      if ((returnVal = cudaStreamQuery(CsvAccess(gpuManager).kernel_stream)) == cudaSuccess) {
        gpuEventEnd(runningKernelIndex);
        profileWorkRequestEvent(head, KernelExecution);
        if (second != NULL && second->state == QUEUED) {
          gpuEventStart(second, &dataSetupIndex, DataSetup, GpuMemSetup);
          workRequestStartTime(second);
          allocateBuffers(second);
          setupData(second);
          second->state = TRANSFERRING_IN;
        }
        if (second != NULL && second->state == TRANSFERRING_IN) {
          if (cudaStreamQuery(data_in_stream) == cudaSuccess) {
            gpuEventEnd(dataSetupIndex);
            profileWorkRequestEvent(second, DataSetup);
            if (third != NULL /*&& (third->state == QUEUED)*/) {
              allocateBuffers(third);
            }
            gpuEventStart(second, &runningKernelIndex, KernelExecution, GpuKernelExec);
            workRequestStartTime(second);
            flushDelayedFrees();
            runKernel(second);
            second->state = EXECUTING;
            if (third != NULL) {
              gpuEventStart(third, &dataSetupIndex, DataSetup, GpuMemSetup);
              workRequestStartTime(third);
              setupData(third);
              third->state = TRANSFERRING_IN;
            }
          }
        }
        gpuEventStart(head, &dataCleanupIndex, DataCleanup, GpuMemCleanup);
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
#ifdef GPU_DEBUG
      printf("wrQueue Size = %d head.state= Transferring out", CsvAccess(gpuManager).wrQueue->size);
#endif
      if (cudaStreamQuery(data_out_stream) == cudaSuccess) {
        freeMemory(head);
        gpuEventEnd(dataCleanupIndex);
        profileWorkRequestEvent(head, DataCleanup);
        wrqueue::dequeue(wrQueue);
        isHeadFinished = true;
      }
    }
    CmiUnlock(CsvAccess(gpuManager).queuelock);
    if (isHeadFinished)
      CUDACallbackManager(head->callbackFn);
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
  CmiDestroyLock(CsvAccess(gpuManager).bufferlock);
  CmiDestroyLock(CsvAccess(gpuManager).queuelock);
  CmiDestroyLock(CsvAccess(gpuManager).progresslock);
  CmiDestroyLock(CsvAccess(gpuManager).pinlock);
  CmiDestroyLock(CsvAccess(gpuManager).dfrlock);
#ifdef GPU_MEMPOOL

#ifndef GPU_DUMMY_MEMPOOL
  releasePool(CsvAccess(gpuManager).memPoolFreeBufs);
#else
  for(int i = 0; i < CsvAccess(gpuManager).memPoolBoundaries.length(); i++){
    printf("(%d) slot %d size: %d max: %d\n", CmiMyPe(), i, CsvAccess(gpuManager).memPoolBoundaries[i], CsvAccess(gpuManager).memPoolMax[i]);
  }

  if(CsvAccess(gpuManager).memPoolBoundaries.length() != CsvAccess(gpuManager).memPoolMax.length()){
    CmiAbort("Error while exiting: memPoolBoundaries and memPoolMax sizes do not match!\n");
  }
#endif

#endif // GPU_MEMPOOL

  wrqueue::deleteWRqueue(CsvAccess(gpuManager).wrQueue);
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).kernel_stream));
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).data_in_stream));
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).data_out_stream));

#ifdef GPU_TRACE
  for (int i=0; i<CsvAccess(gpuManager).timeIndex; i++) {
    switch (CsvAccess(gpuManager).gpuEvents[i].eventType) {
    case DataSetup:
      printf("Kernel %s data setup", CsvAccess(gpuManager).gpuEvents[i].traceName);
      break;
    case DataCleanup:
      printf("Kernel %s data cleanup", CsvAccess(gpuManager).gpuEvents[i].traceName);
      break;
    case KernelExecution:
      printf("Kernel %s execution", CsvAccess(gpuManager).gpuEvents[i].traceName);
      break;
    default:
      printf("Error, invalid timer identifier\n");
    }
    printf(" %.2f:%.2f\n", CsvAccess(gpuManager).gpuEvents[i].cmistartTime-CsvAccess(gpuManager).gpuEvents[0].cmistartTime, CsvAccess(gpuManager).gpuEvents[i].cmiendTime-CsvAccess(gpuManager).gpuEvents[0].cmistartTime);
  }

#endif

}

#ifdef GPU_MEMPOOL
void releasePool(CkVec<BufferPool> &pools){
  for(int i = 0; i < pools.length(); i++){
    Header *hdr = pools[i].head;
    if (hdr != NULL){
      cudaChk(cudaFreeHost((void *)hdr));
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
  // Handle
  CkVec<size_t> memPoolBoundariesHandle = CsvAccess(gpuManager).memPoolBoundaries;

  // Initialize pools
  pools.reserve(nslots);
  pools.length() = nslots;
  for (int i = 0; i < nslots; i++) {
    pools[i].size = memPoolBoundariesHandle[i];
    pools[i].head = NULL;
  }

  // Get GPU memory size
  cudaDeviceProp prop;
  cudaChk(cudaGetDeviceProperties(&prop, getMyCudaDevice(CmiMyPe())));

  // Divide by #pes and multiply by #pes in nodes
  size_t maxMemAlloc = std::min(GPU_MEMPOOL_MAX_INIT_SIZE, prop.totalGlobalMem);
  size_t availableMemory = maxMemAlloc / CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyPe()))
                            * CmiMyNodeSize() * GPU_MEMPOOL_SCALE;

  // Pre-calculate memory per size
  int maxBuffers = *std::max_element(nbuffers, nbuffers+nslots);
  int nBuffersToAllocate[nslots];
  memset(nBuffersToAllocate, 0, sizeof(nBuffersToAllocate));
  size_t bufSize;
  while (availableMemory >= memPoolBoundariesHandle[0] + sizeof(Header)) {
    for (int i = 0; i < maxBuffers; i++) {
      for (int j = nslots - 1; j >= 0; j--) {
        bufSize = memPoolBoundariesHandle[j] + sizeof(Header);
        if (i < nbuffers[j] && bufSize <= availableMemory) {
          nBuffersToAllocate[j]++;
          availableMemory -= bufSize;
        }
      }
    }
  }


  // Pin the host memory
  for (int i = 0; i < nslots; i++) {
    bufSize = memPoolBoundariesHandle[i] + sizeof(Header);
    int numBuffers = nBuffersToAllocate[i];

    Header* hd;
    Header* previous = NULL;

    // Pin host memory in a contiguous block for a slot
    void* pinnedChunk;
    cudaChk(cudaMallocHost(&pinnedChunk, bufSize * numBuffers));

    // Initialize header structs
    for (int j = numBuffers - 1; j >= 0; j--) {
      hd = reinterpret_cast<Header*>(reinterpret_cast<unsigned char*>(pinnedChunk) + bufSize * j);
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
  int boundaryArrayLen = CsvAccess(gpuManager).memPoolBoundaries.length();
  if (size <= CsvAccess(gpuManager).memPoolBoundaries[0]) {
    return 0;
  }
  else if (size > CsvAccess(gpuManager).memPoolBoundaries[boundaryArrayLen-1]) {
    // create new slot
    CsvAccess(gpuManager).memPoolBoundaries.push_back(size);
#ifdef GPU_DUMMY_MEMPOOL
    CsvAccess(gpuManager).memPoolMax.push_back(-1);
    CsvAccess(gpuManager).memPoolSize.push_back(0);
#endif

    BufferPool newpool;
    cudaChk(cudaMallocHost((void **)&newpool.head, size+sizeof(Header)));
    if (newpool.head == NULL) {
      printf("(%d) findPool: failed to allocate newpool %d head, size %d\n", CmiMyPe(), boundaryArrayLen, size);
      CmiAbort("Exiting after failed newpool allocation!\n");
    }
    newpool.size = size;
#ifdef GPU_MEMPOOL_DEBUG
    newpool.num = 1;
#endif
    CsvAccess(gpuManager).memPoolFreeBufs.push_back(newpool);

    Header* hd = newpool.head;
    hd->next = NULL;
    hd->slot = boundaryArrayLen;

    return boundaryArrayLen;
  }
  for (int i = 0; i < CsvAccess(gpuManager).memPoolBoundaries.length()-1; i++) {
    if (CsvAccess(gpuManager).memPoolBoundaries[i] < size && size <= CsvAccess(gpuManager).memPoolBoundaries[i+1]) {
      return (i + 1);
    }
  }
  return -1;
}

void* getBufferFromPool(int pool, int size){
  Header* ret;
  if (pool < 0 || pool >= CsvAccess(gpuManager).memPoolFreeBufs.length()) {
    printf("(%d) getBufferFromPool, pool: %d, size: %d invalid pool\n", CmiMyPe(), pool, size);
#ifdef GPU_MEMPOOL_DEBUG
    printf("(%d) num: %d\n", CmiMyPe(), CsvAccess(gpuManager).memPoolFreeBufs[pool].num);
#endif
    CmiAbort("Exiting after invalid pool!\n");
  }
  else if (CsvAccess(gpuManager).memPoolFreeBufs[pool].head == NULL) {
    Header* hd;
    cudaChk(cudaMallocHost((void **)&hd, sizeof(Header) + CsvAccess(gpuManager).memPoolFreeBufs[pool].size));
#ifdef GPU_MEMPOOL_DEBUG
    printf("(%d) getBufferFromPool, pool: %d, size: %d expand by 1\n", CmiMyPe(), pool, size);
#endif
    if (hd == NULL) {
      CmiAbort("Exiting after NULL hd from pool!\n");
    }
    hd->slot = pool;
    return (void *)(hd + 1);
  }
  else {
    ret = CsvAccess(gpuManager).memPoolFreeBufs[pool].head;
    CsvAccess(gpuManager).memPoolFreeBufs[pool].head = ret->next;
#ifdef GPU_MEMPOOL_DEBUG
    ret->size = size;
    CsvAccess(gpuManager).memPoolFreeBufs[pool].num--;
#endif
    return (void *)(ret + 1);
  }
  return NULL;
}

void returnBufferToPool(int pool, Header* hd) {
  hd->next = CsvAccess(gpuManager).memPoolFreeBufs[pool].head;
  CsvAccess(gpuManager).memPoolFreeBufs[pool].head = hd;
#ifdef GPU_MEMPOOL_DEBUG
  CsvAccess(gpuManager).memPoolFreeBufs[pool].num++;
#endif
}

void* hapi_poolMalloc(int size) {
  CmiLock(CsvAccess(gpuManager).bufferlock);
  int pool = findPool(size);
  void* buf;
#ifdef GPU_DUMMY_MEMPOOL
  cudaChk(cudaMallocHost((void **)&buf, size+sizeof(Header)));
  if(pool < 0 || pool >= CsvAccess(gpuManager).memPoolSize.length()){
    printf("(%d) need to create up to pool %d; malloc size: %d\n", CmiMyPe(), pool, size);
    CmiAbort("Exiting after need to create bigger pool!\n");
  }
  CsvAccess(gpuManager).memPoolSize[pool]++;
  if(CsvAccess(gpuManager).memPoolSize[pool] > CsvAccess(gpuManager).memPoolMax[pool]){
    CsvAccess(gpuManager).memPoolMax[pool] = CsvAccess(gpuManager).memPoolSize[pool];
  }
  Header *hdr = (Header *)buf;
  hdr->slot = pool;
  hdr = hdr+1;
  buf = (void *)hdr;
#else
  buf = getBufferFromPool(pool, size);
#endif

#ifdef GPU_MEMPOOL_DEBUG
  printf("(%d) hapi_malloc size %d pool %d left %d\n", CmiMyPe(), size, pool, CsvAccess(gpuManager).memPoolFreeBufs[pool].num);
#endif
  CmiUnlock(CsvAccess(gpuManager).bufferlock);
  return buf;
}

void hapi_poolFree(void* ptr) {
  Header* hd = ((Header *)ptr) - 1;
  int pool = hd->slot;

#ifdef GPU_MEMPOOL_DEBUG
  int size = hd->size;
#endif // GPU_MEMPOOL_DEBUG

  /* FIXME: We should get rid of code under GPU_DUMMY_MEMPOOL because
   *  a) we don't use this flag
   *  b) code under this flag does nothing useful.
   */
  CmiLock(CsvAccess(gpuManager).bufferlock);
#ifdef GPU_DUMMY_MEMPOOL
  if(pool < 0 || pool >= CsvAccess(gpuManager).memPoolSize.length()){
    printf("(%d) free pool %d\n", CmiMyPe(), pool);
    CmiAbort("Exiting after freeing out of bounds pool!\n");
  }
  CsvAccess(gpuManager).memPoolSize[pool]--;
  delayedFree((void *)hd);
#else
  returnBufferToPool(pool, hd);
#endif
  CmiUnlock(CsvAccess(gpuManager).bufferlock);

#ifdef GPU_MEMPOOL_DEBUG
  printf("(%d) hapi_free size %d pool %d left %d\n", CmiMyPe(), size, pool, CsvAccess(gpuManager).memPoolFreeBufs[pool].num);
#endif
}


#endif // GPU_MEMPOOL

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
