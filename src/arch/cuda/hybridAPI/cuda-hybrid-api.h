/* 
 * cuda-hybrid-api.h
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

#ifndef __CUDA_HYBRID_API_H__
#define __CUDA_HYBRID_API_H__

#ifdef GPU_MEMPOOL
#include "cklists.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* initHybridAPI
   initializes the work request queue
*/
void initHybridAPI(int myPe); 

/* gpuProgressFn
   called periodically to check if the current kernel has completed,
   and invoke subsequent kernel */
void gpuProgressFn();

/* exitHybridAPI
   cleans up and deletes memory allocated for the queue
*/
void exitHybridAPI(); 


#ifdef GPU_MEMPOOL
// data and metadata reside in same chunk of memory
typedef struct _header{
  //void *buf;
  struct _header *next;
  int slot;
}Header;

typedef struct _bufferPool{
  Header *head;
  //bool expanded;
  int size;
}BufferPool;

// pre-allocated buffers will be at least this big
#define GPU_MEMPOOL_MIN_BUFFER_SIZE 1024

void createPool(int *nbuffers, int nslots, CkVec<BufferPool> &pools);

#endif

#ifdef __cplusplus
}
#endif

#endif
