/*
 * wr.h
 *
 * by Lukasz Wesolowski
 * 06.02.2008
 *
 * header containing declarations needed by the user of the API
 *
 */

#ifndef __WR_H__
#define __WR_H__

/* struct pinnedMemReq
 *
 * a structure for submitting page-locked memory allocation requests;
 * passed as input into pinnedMallocHost
 *
 */
typedef struct pinnedMemReq {
  void*** hostPtrs;
  size_t* sizes;
  int nBuffers;
  void* callbackFn;
} pinnedMemReq;


/**
 Do a CudaFreeHost on this host pinned memory,
 but only when no kernels are runnable.
*/
void delayedFree(void* ptr);


#ifdef CUDA_USE_CUDAMALLOCHOST /* <- user define */
#  ifdef CUDA_MEMPOOL
#    define hapi_hostFree hapi_poolFree
#  else
#    define hapi_hostFree delayedFree
#  endif
#else
#  define hapi_hostFree free
#endif


/* pinnedMallocHost
 *
 * schedules a pinned memory allocation so that it does not impede
 * concurrent asynchronous execution
 *
 */
void pinnedMallocHost(pinnedMemReq* reqs);


/* struct bufferInfo
 *
 * purpose:
 * structure to indicate which actions the runtime system should
 * perform in relation to the buffer
 *
 * usage:
 *
 * the user associates an array of dataInfo structures with each
 * submitted work request; device memory will be allocated if there is
 * no buffer in use for that index
 *
 */
typedef struct dataInfo {
  // ID of buffer in the runtime system's buffer table
  int bufferID;

  // flags to indicate if the buffer should be transferred
  int transferToDevice;
  int transferFromDevice;

  // flag to indicate if the device buffer memory should be freed
  // after  execution of work request
  int freeBuffer;

  // pointer to host data buffer
  void* hostBuffer;

  // size of buffer in bytes
  size_t size;

  dataInfo(int _bufferID = -1): bufferID(_bufferID) {}

} dataInfo;



/* struct workRequest
 *
 * purpose:
 * structure for organizing work units for execution on the GPU
 *
 * usage model:
 * 1. declare a pointer to a workRequest
 * 2. allocate dynamic memory for the work request
 * 3. define the data members for the work request
 * 4. enqueue the work request
 */


typedef struct workRequest {
  // The following parameters need to be set by the user

  // parameters for kernel execution
  dim3 dimGrid;
  dim3 dimBlock;
  int smemSize;

  // array of dataInfo structs containing buffer information for the
  // execution of the work request
  dataInfo* bufferInfo;

  // number of buffers used by the work request
  int nBuffers;

  // a Charm++ callback function (cast to a void *) to be called after
  // the kernel finishes executing on the GPU
  void* callbackFn;

  // Short identifier used for tracing and logging
  const char *traceName;

  /**
    Host-side function to run this kernel (0 if no kernel to run)
      kernelStream is the cuda stream to run the kernel in.
      deviceBuffers is an array of device pointers, indexed by bufferInfo -> bufferID.
  */
  void (*runKernel)(struct workRequest *wr, cudaStream_t kernelStream, void **deviceBuffers);

  // The following flag is used for control by the system
  int state;

  // user data, may be used to pass scalar values to kernel calls
  void* userData;

#ifdef GPU_INSTRUMENT_WRS
  double phaseStartTime;
  int chareIndex;
  char compType;
  char compPhase;
#endif

  workRequest()
  	:dimGrid(0), dimBlock(0), smemSize(0),
  	 bufferInfo(0), nBuffers(0), callbackFn(0),
  	 traceName(""), runKernel(0), state(0),
  	 userData(0)
  {
#ifdef GPU_INSTRUMENT_WRS
    chareIndex = -1;
#endif
  }

} workRequest;


/* struct workRequestQueue
 *
 * purpose: container for GPU work requests
 *
 * usage model:
 * 1. declare a workRequestQueue
 * 2. call init to allocate memory for the queue and initialize
 *    bookkeeping variables
 * 3. enqueue each work request which needs to be
 *    executed on the GPU
 * 4. the runtime system will be invoked periodically to
 *    handle the details of executing the work request on the GPU
 *
 * implementation notes:
 * the queue is implemented using a circular array; if the array fills
 * up, requests are transferred to a queue having additional
 * QUEUE_EXPANSION_SIZE slots, and the memory for the old queue is freed
 */
typedef struct {
  // array of requests
  workRequest* requests;

  // array index for the logically first item in the queue
  int head;

  // array index for the last item in the queue
  int tail;

  // number of work requests in the queue
  int size;

  // size of the array of work request
  int capacity;

} workRequestQueue;

/* enqueue
 *
 * add a work request to the queue to be later executed on the GPU
 *
 */
void enqueue(workRequestQueue* q, workRequest* wr);
void enqueue(workRequest* wr);
void setWRCallback(workRequest* wr, void* cb);

#ifdef GPU_MEMPOOL
/**
 Return host pinned memory, just like delayedFree,
 but with the option of recycling to future poolMallocs.
*/
void hapi_poolFree(void*);

/**
 Allocate host pinned memory from the pool.
*/
void *hapi_poolMalloc(int size);
#endif // GPU_MEMPOOL
/* external declarations needed by the user */

void** getdevBuffers();
cudaStream_t getKernelStream();

#ifdef GPU_INSTRUMENT_WRS
struct RequestTimeInfo {
  double transferTime;
  double kernelTime;
  double cleanupTime;
  int n;

  RequestTimeInfo(){
    transferTime = 0.0;
    kernelTime = 0.0;
    cleanupTime = 0.0;
    n = 0;
  }
};

void hapi_initInstrument(int nchares, char ntypes);
RequestTimeInfo *hapi_queryInstrument(int chare, char type, char phase);
#endif

#endif // __WR_H__

