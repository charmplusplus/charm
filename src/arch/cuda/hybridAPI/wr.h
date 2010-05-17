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
  void ***hostPtrs; 
  size_t *sizes;
  int nBuffers;
  void *callbackFn; 
} pinnedMemReq;

void delayedFree(void *ptr);


/* pinnedMallocHost
 *
 * schedules a pinned memory allocation so that it does not impede
 * concurrent asynchronous execution 
 *
 */
void pinnedMallocHost(pinnedMemReq *reqs);


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

  /* ID of buffer in the runtime system's buffer table*/
  int bufferID; 

  /* flags to indicate if the buffer should be transferred */
  int transferToDevice; 
  int transferFromDevice; 
  
  /* flag to indicate if the device buffer memory should be freed
     after  execution of work request */
  int freeBuffer; 

  /* pointer to host data buffer */
  void *hostBuffer; 

  /* size of buffer in bytes */
  size_t size; 

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
  /* The following parameters need to be set by the user */

  /* parameters for kernel execution */
  dim3 dimGrid; 
  dim3 dimBlock; 
  int smemSize;
  
  /* array of dataInfo structs containing buffer information for the
     execution of the work request */ 
  dataInfo *bufferInfo; 
  
  /* number of buffers used by the work request */ 
  int nBuffers; 

  /* a Charm++ callback function (cast to a void *) to be called after
     the kernel finishes executing on the GPU */ 
  void *callbackFn; 
 
  /* id to select the correct kernel in kernelSelect */
  int id; 

  /* The following flag is used for control by the system */
  int state; 

  /* user data, may be used to pass scalar values to kernel calls */
  void *userData; 

#ifdef GPU_INSTRUMENT_WRS
  double phaseStartTime;
  int chareIndex;
  char compType;
  char compPhase;

  workRequest(){
    chareIndex = -1;
  }
#endif

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

  /* array of requests */
  workRequest* requests; 

  /* array index for the logically first item in the queue */
  int head; 

  /* array index for the last item in the queue */ 
  int tail; 

  /* number of work requests in the queue */
  int size; 

  /* size of the array of work requests */ 
  int capacity; 

} workRequestQueue; 

/* enqueue
 *
 * add a work request to the queue to be later executed on the GPU
 *
 */
void enqueue(workRequestQueue *q, workRequest *wr); 
void setWRCallback(workRequest *wr, void *cb);

#ifdef GPU_MEMPOOL
void hapi_poolFree(void *);
void *hapi_poolMalloc(int size);
#endif
/* external declarations needed by the user */

extern workRequestQueue *wrQueue; 
extern void **devBuffers; 
extern cudaStream_t kernel_stream; 

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

#endif


