/*
 * wrqueue.h
 *
 * by Lukasz Wesolowski
 * 04.12.2008
 *
 * a simple FIFO queue for GPU work requests
 *
 */

#ifndef __WR_QUEUE_H__
#define __WR_QUEUE_H__

/* number of work requests the queue is initialized to handle */
#define QUEUE_SIZE_INIT 10

/* if the queue is filled, it will be expanded by this many additional 
   units */ 
#define QUEUE_EXPANSION_SIZE 10

/* struct workRequest
 * 
 * purpose:  
 * structure for holding information about work requests on the GPU
 *
 * usage model: 
 * 1. declare a pointer to a workRequest 
 * 2. allocate dynamic memory for the work request
 * 3. call setupMemory to copy over the data to the GPU 
 * 4. enqueue the work request by using addWorkRequest
 */

typedef struct workRequest {

  /* parameters for kernel execution */

  dim3 dimGrid; 
  dim3 dimBlock; 
  int smemSize;
  
  /* pointers to queues and their lengths on the device(gpu) and
     host(cpu)  */

  void *readWriteDevicePtr;
  void *readWriteHostPtr; 
  int readWriteLen; 

  void *readOnlyDevicePtr; 
  void *readOnlyHostPtr; 
  int readOnlyLen; 

  void *writeOnlyDevicePtr;
  void *writeOnlyHostPtr; 
  int writeOnlyLen; 

  /* to be called after the kernel finishes executing on the GPU */ 

  void (*callbackFn)(); 

  /* to select the correct kernel in the switch statement */

  int switchNo; 

  /* event which will be polled to check if kernel has finished
     execution */

  cudaEvent_t completionEvent;  

  /* flags */

  int executing; 

} workRequest; 


/* struct workRequestQueue 
 *
 * purpose: container/mechanism for GPU work requests 
 *
 * usage model: 
 * 1. declare a workRequestQueue
 * 2. call init to allocate memory for the queue and initialize
 *    bookkeeping variables
 * 3. call enqueue for each request which needs to be 
 *    executed on the GPU
 * 4. in the hybrid API gpuProgressFn will execute periodically to
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

/* init_wrqueue
 *
 * allocate memory for the queue and initialize bookkeeping variables
 *
 */
void init_wrqueue(workRequestQueue *q); 

/* enqueue
 *
 * add a work request to the queue to be later executed on the GPU
 *
 */

void enqueue(workRequestQueue *q, workRequest *wr); 

/* dequeue
 *
 * delete the head entry in the queue
 * assumes memory buffers have previously been freed or will be reused
 *
 */
void dequeue(workRequestQueue *q); 

/* delete_wrqueue
 *
 * if queue is nonempty, return -1  
 * if queue is empty, delete the queue, freeing dynamically allocated 
 * memory, and return 0
 *
 *
 */

int delete_wrqueue(workRequestQueue *q); 

/* head
 * 
 * returns the first element in the queue 
 *
 */

workRequest * head(workRequestQueue *q);

/*
 * isEmpty
 *
 * returns:
 * 1 if queue has no pending requests stored
 * 0 otherwise
 */

int isEmpty(workRequestQueue *q);

#endif
