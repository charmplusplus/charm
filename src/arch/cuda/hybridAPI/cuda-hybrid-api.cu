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
#include "mempool.h"
#include "cklists.h"
#endif

void cudaErrorDie(int err, const char* code, const char* file, int line) {
  fprintf(stderr, "Fatal CUDA Error %s at %s:%d.\nReturn value %d from '%s'.",
      cudaGetErrorString((cudaError_t) err), file, line, err, code);
  CmiAbort(" Exiting!\n");
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
  int ID;
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
// Scale the amount of memory each node pins
#define GPU_MEMPOOL_SCALE 1.0

  // Pinned host memory pool (from cudaMallocHost)
  mempool_type *mp_pinned;

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
 * a switch statement defined by the user to allow the library to execute
 * the correct kernel
 */
void kernelSelect(workRequest *wr);
#ifdef GPU_MEMPOOL

/* Underlying functions that grab large chunks of 
   pinned memory for the pool. */

#define mempool_type_cudaMallocHost 123
#define mempool_type_systemMalloc 125
static void * mempool_gpu_alloc(size_t *size, mem_handle_t *mem_hndl, int expand_flag)
{
    void *mem=0;
    cudaMallocHost(&mem,*size); // <- no cudaChk, we check manually
    if (mem!=0) { // cudaMallocHost worked
        *mem_hndl=mempool_type_cudaMallocHost;
    } else { 
        // cudaMallocHost failed--fall back to non-pinned system malloc?
        mem=malloc(*size);
        if (mem==0) { // uh, what?
            CmiError("HybridAPI mempool_gpu_alloc of %zu bytes failed both cudaMallocHost and system malloc\n",
                *size);
            CmiAbort("HybridAPI mempool_gpu_alloc failure");
        }
        *mem_hndl=mempool_type_systemMalloc;
    }
#ifdef GPU_TRACE
    CkPrintf("mempool_gpumalloc: size %ld, pointer %p (expand=%s)\n",
        *size,mem,expand_flag?"true":"false");
#endif
    return mem;
}
static void mempool_gpu_free(void *ptr, mem_handle_t mem_hndl)
{
#ifdef GPU_TRACE
    CkPrintf("mempool_gpu_free: pointer %p\n",ptr);
#endif
    switch (mem_hndl) {
    case mempool_type_cudaMallocHost:
        cudaChk(cudaFreeHost(ptr));
        break;
    case mempool_type_systemMalloc:
        free(ptr);
        break;
    default:
        CmiError("HybridAPI mempool_gpu_free of pointer %p has invalid mem_hndl type %d\n",
            ptr,(int)mem_hndl);
        CmiAbort("HybridAPI mempool_gpu_free failure");
        break;
    };
}

#endif

inline int getMyCudaDevice(int myPe) {
  int deviceCount;
  cudaChk(cudaGetDeviceCount(&deviceCount));
  return myPe % deviceCount;
}

void initHybridAPI() {
  CsvInitialize(GPUManager, gpuManager);

#ifdef GPU_MEMPOOL
  // Get GPU memory size
  cudaDeviceProp prop;
  cudaChk(cudaGetDeviceProperties(&prop, getMyCudaDevice(CmiMyPe())));

  // Divide by #pes and multiply by #pes in nodes
  size_t availableMemory = prop.totalGlobalMem / CmiNumPesOnPhysicalNode(CmiPhysicalNodeID(CmiMyPe()))
                            * CmiMyNodeSize() * GPU_MEMPOOL_SCALE;

  CsvAccess(gpuManager).mp_pinned = mempool_init(availableMemory,mempool_gpu_alloc,mempool_gpu_free,0);

#ifdef GPU_MEMPOOL_DEBUG
  printf("[%d] done creating buffer pool\n", CmiMyPe());
#endif

#endif
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
  shared_gpuEvents[shared_timeIndex].ID = wr->id;
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
  printf("End Event Name = %d\t Stage Name=%d workRequest Id=%d\n",CsvAccess(gpuManager).gpuEvents[index].eventType, CsvAccess(gpuManager).gpuEvents[index].stage, CsvAccess(gpuManager).gpuEvents[index].ID);
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
        kernelSelect(head);

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
            kernelSelect(second);
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
  mempool_destroy(CsvAccess(gpuManager).mp_pinned);
#endif // GPU_MEMPOOL

  wrqueue::deleteWRqueue(CsvAccess(gpuManager).wrQueue);
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).kernel_stream));
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).data_in_stream));
  cudaChk(cudaStreamDestroy(CsvAccess(gpuManager).data_out_stream));

#ifdef GPU_TRACE
  for (int i=0; i<CsvAccess(gpuManager).timeIndex; i++) {
    switch (CsvAccess(gpuManager).gpuEvents[i].eventType) {
    case DataSetup:
      printf("Kernel %d data setup", CsvAccess(gpuManager).gpuEvents[i].ID);
      break;
    case DataCleanup:
      printf("Kernel %d data cleanup", CsvAccess(gpuManager).gpuEvents[i].ID);
      break;
    case KernelExecution:
      printf("Kernel %d execution", CsvAccess(gpuManager).gpuEvents[i].ID);
      break;
    default:
      printf("Error, invalid timer identifier\n");
    }
    printf(" %.2f:%.2f\n", CsvAccess(gpuManager).gpuEvents[i].cmistartTime-CsvAccess(gpuManager).gpuEvents[0].cmistartTime, CsvAccess(gpuManager).gpuEvents[i].cmiendTime-CsvAccess(gpuManager).gpuEvents[0].cmistartTime);
  }

#endif

}


#ifdef GPU_MEMPOOL

void* hapi_poolMalloc(size_t size) {
  CmiLock(CsvAccess(gpuManager).bufferlock);
  void *buf=mempool_malloc(CsvAccess(gpuManager).mp_pinned,size,1);
  CmiUnlock(CsvAccess(gpuManager).bufferlock);
  return buf;
}

void hapi_poolFree(void* ptr) {
  CmiLock(CsvAccess(gpuManager).bufferlock);
  mempool_free(CsvAccess(gpuManager).mp_pinned,ptr);
  CmiUnlock(CsvAccess(gpuManager).bufferlock);
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
